"""
BioInsight Agentic Radar — API Fetcher Tools
=============================================
Decoupled, LangGraph-compatible Tool wrappers for NIH RePORTER v2
and PubMed (via metapub).

Design Philosophy:
  - Each fetcher is a pure function decorated with @tool so LangGraph
    can invoke it as a conditional edge from the Library Checker Node.
  - Fetchers return a list[BioInsightRecord] with embeddings=[]
    (the EmbeddingService fills those before ChromaDB ingestion).
  - Granularity is inferred from keyword heuristics; callers can override.
  - All network calls are retry-wrapped with exponential back-off to
    survive intermittent NIH / NCBI rate limiting.

Usage (standalone):
    from fetcher_tools import fetch_nih_reporter, fetch_pubmed

    grants = fetch_nih_reporter.invoke({
        "domain": "parkinsons",
        "fiscal_year": 2026,
        "max_results": 200,
    })
    papers = fetch_pubmed.invoke({
        "domain": "autism",
        "query": "autism MAPK pathway 2024",
        "max_results": 100,
    })
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests
from langchain_core.tools import tool
from metapub import PubMedFetcher

from bioinsight.chroma_manager import BioInsightRecord, GranularityType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_GRANULARITY_KEYWORDS: dict[GranularityType, list[str]] = {
    "protein": [
        "protein",
        "kinase",
        "receptor",
        "enzyme",
        "LRRK2",
        "MAPK",
        "alpha-synuclein",
        "tau",
        "amyloid",
        "BDNF",
        "mTOR",
    ],
    "pathway": [
        "pathway",
        "signaling",
        "cascade",
        "network",
        "circuit",
        "mechanism",
        "axis",
        "crosstalk",
        "transduction",
    ],
    "field": [],  # catch-all
}


def _infer_granularity(text: str) -> GranularityType:
    """Heuristic granularity tag from title/abstract text."""
    text_lower = text.lower()
    for level in ("protein", "pathway"):  # most specific first
        if any(kw.lower() in text_lower for kw in _GRANULARITY_KEYWORDS[level]):
            return level
    return "field"


def _retry(fn, retries: int = 3, base_delay: float = 2.0):
    """Simple exponential-back-off wrapper."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            wait = base_delay * (2**attempt)
            logger.warning(
                "Attempt %d failed (%s) — retrying in %.1fs", attempt + 1, exc, wait
            )
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Tool 1 — NIH RePORTER v2
# ---------------------------------------------------------------------------

NIH_REPORTER_URL = "https://api.reporter.nih.gov/v2/projects/search"
NIH_PAGE_SIZE = 500  # max allowed by the API


def _nih_reporter_page(
    domain: str,
    fiscal_year: int,
    offset: int,
    limit: int,
) -> list[dict]:
    """Fetch one page of RePORTER results and return raw project dicts."""
    payload = {
        "criteria": {
            "advanced_text_search": {
                "operator": "and",
                "search_field": "all",
                "search_text": domain,
            },
            "fiscal_years": [fiscal_year],
        },
        "offset": offset,
        "limit": limit,
        "include_fields": [
            "ApplId",
            "ProjectTitle",
            "AbstractText",
            "PrincipalInvestigators",
            "OrgName",
            "FiscalYear",
        ],
    }
    resp = requests.post(NIH_REPORTER_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


@tool
def fetch_nih_reporter(
    domain: str,
    fiscal_year: int,
    max_results: int = 500,
) -> list[BioInsightRecord]:
    """
    Fetch NIH RePORTER grants for a given domain and fiscal year.

    Args:
        domain: Disease or research area keyword (e.g. "parkinsons").
        fiscal_year: NIH fiscal year (e.g. 2026).
        max_results: Upper bound on records returned (default 500).

    Returns:
        List of BioInsightRecord objects with embedding=[] ready for
        EmbeddingService processing.
    """
    records: list[BioInsightRecord] = []
    offset = 0
    limit = min(NIH_PAGE_SIZE, max_results)

    logger.info(
        "NIH RePORTER fetch  |  domain=%s  year=%d  max=%d",
        domain,
        fiscal_year,
        max_results,
    )

    while len(records) < max_results:
        batch = _retry(lambda: _nih_reporter_page(domain, fiscal_year, offset, limit))
        if not batch:
            break

        for proj in batch:
            appl_id = str(proj.get("appl_id", ""))
            title = proj.get("project_title", "") or ""
            abstract = proj.get("abstract_text", "") or ""
            text = f"{title}\n\n{abstract}".strip()

            if not text or not appl_id:
                continue

            # Extract PI name (RePORTER returns a list of PI dicts)
            pis: list[dict] = proj.get("principal_investigators", []) or []
            pi_name = pis[0].get("full_name", "Unknown PI") if pis else "Unknown PI"

            records.append(
                BioInsightRecord(
                    source="nih_reporter",
                    domain=domain.lower(),
                    year=proj.get("fiscal_year", fiscal_year),
                    pi_name=pi_name,
                    org_name=proj.get("org_name", "Unknown Org"),
                    external_id=appl_id,
                    granularity=_infer_granularity(text),
                    text=text,
                    title=title,
                )
            )

        fetched_this_page = len(batch)
        offset += fetched_this_page
        if fetched_this_page < limit:
            break  # last page

    logger.info(
        "NIH RePORTER: fetched %d records for domain=%s year=%d",
        len(records),
        domain,
        fiscal_year,
    )
    return records[:max_results]


# ---------------------------------------------------------------------------
# Tool 2 — PubMed via metapub
# ---------------------------------------------------------------------------


def _build_pubmed_query(domain: str, query: Optional[str], year: Optional[int]) -> str:
    """Compose an Entrez search string from components."""
    parts = []
    if query:
        parts.append(query)
    else:
        parts.append(domain)
    if year:
        parts.append(f"{year}[pdat]")
    return " AND ".join(parts)


@tool
def fetch_pubmed(
    domain: str,
    query: Optional[str] = None,
    year: Optional[int] = None,
    max_results: int = 200,
    email: str = "bioinsight@example.com",  # Entrez etiquette: set to your real email
) -> list[BioInsightRecord]:
    """
    Fetch PubMed abstracts for a given domain or query string.

    Args:
        domain: Disease / research area keyword — used as default query term
                and as the metadata domain tag.
        query: Optional explicit Entrez query (overrides domain-only search).
               Example: "autism MAPK pathway 2024[pdat]"
        year: Optional publication year filter (e.g. 2024).
        max_results: Upper bound on records returned (default 200).
        email: Email for NCBI rate-limit courtesy header.

    Returns:
        List of BioInsightRecord objects with embedding=[] ready for
        EmbeddingService processing.
    """
    from metapub import PubMedFetcher  # lazy import — keeps startup fast

    fetch = PubMedFetcher(email=email)
    search_str = _build_pubmed_query(domain, query, year)

    logger.info("PubMed fetch  |  query='%s'  max=%d", search_str, max_results)

    pmids: list[str] = _retry(
        lambda: fetch.pmids_for_query(search_str, retmax=max_results)
    )

    if not pmids:
        logger.warning("PubMed returned 0 PMIDs for query: %s", search_str)
        return []

    records: list[BioInsightRecord] = []

    for pmid in pmids[:max_results]:
        try:
            article = _retry(lambda: fetch.article_by_pmid(pmid))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping PMID %s: %s", pmid, exc)
            continue

        if article is None:
            continue

        title = article.title or ""
        abstract = article.abstract or ""
        text = f"{title}\n\n{abstract}".strip()

        if not text:
            continue

        pub_year = year or (article.year or 0)
        try:
            pub_year = int(pub_year)
        except (TypeError, ValueError):
            pub_year = 0

        # Lead author as PI analogue
        authors = article.authors or []
        pi_name = authors[0] if authors else "Unknown Author"

        records.append(
            BioInsightRecord(
                source="pubmed",
                domain=domain.lower(),
                year=pub_year,
                pi_name=pi_name,
                org_name=article.journal or "Unknown Journal",
                external_id=str(pmid),
                granularity=_infer_granularity(text),
                text=text,
                title=title,
            )
        )

    logger.info("PubMed: fetched %d records for query='%s'", len(records), search_str)
    return records


# ---------------------------------------------------------------------------
# Tool 3 — EmbeddingService (Hugging Face Inference API → BioBERT)
# ---------------------------------------------------------------------------

HF_INFERENCE_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/"
    "dmis-lab/biobert-v1.1"
)


@tool
def embed_records(
    records: list[BioInsightRecord],
    hf_token: str,
    batch_size: int = 32,
) -> list[BioInsightRecord]:
    """
    Populate BioInsightRecord.embedding via HuggingFace Inference API (BioBERT).

    Mutates records in-place and also returns the list for chaining.

    Args:
        records: List of records with empty embeddings.
        hf_token: HuggingFace API token (store in env var HF_TOKEN).
        batch_size: Number of texts per API call (default 32).

    Returns:
        Same list with .embedding populated on each record.
    """
    if not records:
        return records

    headers = {"Authorization": f"Bearer {hf_token}"}
    texts = [r.text for r in records]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        payload = {"inputs": batch_texts, "options": {"wait_for_model": True}}

        response = _retry(
            lambda: requests.post(
                HF_INFERENCE_URL, headers=headers, json=payload, timeout=60
            )
        )
        response.raise_for_status()
        batch_vecs = response.json()

        # HF feature-extraction returns [n_texts, seq_len, 768].
        # CLS-token pooling (index 0) gives the sentence-level embedding.
        for vec in batch_vecs:
            if isinstance(vec[0], list):
                all_embeddings.append(vec[0])  # CLS token
            else:
                all_embeddings.append(vec)  # already pooled

        logger.info(
            "Embedded batch %d–%d / %d",
            i + 1,
            min(i + batch_size, len(texts)),
            len(texts),
        )
        time.sleep(0.2)  # polite rate limiting

    for rec, emb in zip(records, all_embeddings):
        rec.embedding = emb

    logger.info("Embedding complete: %d records → 768-D BioBERT vectors.", len(records))
    return records
