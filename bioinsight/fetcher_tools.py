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
  - Records are split into sentence-level passages at ingest time.
  - Passage IDs follow the format: {source}__{parent_id}__s{sentence_index}
  - All network calls are retry-wrapped with exponential back-off to
    survive intermittent NIH / NCBI rate limiting.
"""

from __future__ import annotations
import logging
import re
import time
from typing import Optional
import os
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv
import spacy

load_dotenv()

from metapub import PubMedFetcher
from bioinsight.chroma_manager import BioInsightRecord, GranularityType

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")

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
    "field": [],
}


def split_into_passages(text: str, parent_id: str, source: str) -> list[dict]:
    """Split text into sentence-level passages with stable IDs."""
    doc = nlp(text)
    passages = []
    for i, sent in enumerate(doc.sents):
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        passages.append(
            {
                "passage_id": f"{source}__{parent_id}__s{i}",
                "text": sent_text,
                "sentence_index": i,
                "parent_id": parent_id,
            }
        )
    return passages


def _infer_granularity(text: str) -> GranularityType:
    """Heuristic granularity tag from title/abstract text."""
    text_lower = text.lower()
    for level in ("protein", "pathway"):
        if any(kw.lower() in text_lower for kw in _GRANULARITY_KEYWORDS[level]):
            return level
    return "field"


def _retry(fn, retries: int = 3, base_delay: float = 2.0):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if "Invalid ID" in str(exc):
                logger.warning("Skipping invalid ID: %s", exc)
                return None
            if attempt == retries - 1:
                raise
            wait = base_delay * (2**attempt)
            logger.warning(
                "Attempt %d failed (%s) — retrying in %.1fs", attempt + 1, exc, wait
            )
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Corpus size probes — cheap count calls before committing to full fetches
# ---------------------------------------------------------------------------

NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NIH_REPORTER_URL = "https://api.reporter.nih.gov/v2/projects/search"
NIH_PAGE_SIZE = 500


def count_pubmed(query: str, year: int, email: str = "bioinsight@example.com") -> int:
    """Return total PubMed hits for a query+year without downloading any records."""
    full_query = _build_pubmed_query(query, None, year)
    params: dict = {
        "db": "pubmed",
        "term": full_query,
        "retmax": 0,
        "rettype": "count",
        "email": email,
    }
    api_key = os.environ.get("NCBI_API_KEY", "")
    if api_key:
        params["api_key"] = api_key
    try:
        resp = requests.get(NCBI_ESEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        m = re.search(r"<Count>(\d+)</Count>", resp.text)
        return int(m.group(1)) if m else 0
    except Exception as exc:
        logger.warning("count_pubmed failed for '%s' year=%d: %s", query, year, exc)
        return 0


def count_nih_reporter(
    domain: str,
    fiscal_year: int,
) -> int:
    """Return total NIH Reporter hits without downloading project records."""
    criteria: dict = {
        "advanced_text_search": {
            "operator": "advanced",
            "search_field": "all",
            "search_text": domain,
        },
        "fiscal_years": [fiscal_year],
    }

    payload = {
        "criteria": criteria,
        "offset": 0,
        "limit": 1,
        "include_fields": ["ApplId"],
    }
    try:
        resp = requests.post(NIH_REPORTER_URL, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # NIH Reporter v2: total is inside the "meta" sub-object, not at root level
        total = data.get("meta", {}).get("total", data.get("total", 0))
        logger.info("NIH count: year=%d query='%s' → %d", fiscal_year, domain[:80], total)
        return total
    except Exception as exc:
        logger.warning(
            "count_nih_reporter failed for '%s' year=%d: %s", domain, fiscal_year, exc
        )
        return 0


def _nih_reporter_page(
    domain: str,
    fiscal_year: int,
    offset: int,
    limit: int,
) -> list[dict]:
    # "advanced" operator supports full boolean: AND, OR, NOT, parentheses,
    # and quoted phrases (e.g. "Parkinson Disease" AND (LRRK2 OR GBA)).
    # search_field="all" searches titles, abstracts, and terms — needed because
    # grants often use "autism" / "ASD" rather than the full phrase in abstracts.
    criteria: dict = {
        "advanced_text_search": {
            "operator": "advanced",
            "search_field": "all",
            "search_text": domain,
        },
        "fiscal_years": [fiscal_year],
    }

    payload = {
        "criteria": criteria,
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
    return resp.json().get("results", [])


@tool
def fetch_nih_reporter(
    domain: str,
    fiscal_year: int,
    max_results: int = 1000,
    offset_start: int = 0,
) -> list[BioInsightRecord]:
    """
    Fetch NIH RePORTER grants for a given domain and fiscal year.
    Each grant abstract is split into sentence-level passages at ingest.

    Args:
        domain: Boolean search string using "advanced" operator syntax.
        fiscal_year: NIH fiscal year (e.g. 2024).
        max_results: Upper bound on records returned (default 500).
        offset_start: Starting offset into the result set (for stratified sampling).

    Returns:
        List of BioInsightRecord objects with embedding=[] ready for
        EmbeddingService processing.
    """
    records: list[BioInsightRecord] = []
    offset = offset_start
    limit = min(NIH_PAGE_SIZE, max_results)

    logger.info(
        "NIH RePORTER fetch  |  domain=%s  year=%d",
        domain,
        fiscal_year,
    )

    while len(records) < max_results:
        batch = _retry(
            lambda: _nih_reporter_page(domain, fiscal_year, offset, limit)
        )
        if not batch:
            break

        for proj in batch:
            appl_id = str(proj.get("appl_id", ""))
            title = proj.get("project_title", "") or ""
            abstract = proj.get("abstract_text", "") or ""
            text = f"{title}\n\n{abstract}".strip()

            if not text or not appl_id:
                continue

            pis: list[dict] = proj.get("principal_investigators", []) or []
            pi_name = pis[0].get("full_name", "Unknown PI") if pis else "Unknown PI"
            org_name = proj.get("org_name", "Unknown Org")
            year = proj.get("fiscal_year", fiscal_year)

            passages = split_into_passages(
                text=text,
                parent_id=appl_id,
                source="nih_reporter",
            )

            for passage in passages:
                records.append(
                    BioInsightRecord(
                        source="nih_reporter",
                        domain=domain.lower(),
                        year=year,
                        pi_name=pi_name,
                        org_name=org_name,
                        external_id=passage["passage_id"],
                        granularity=_infer_granularity(passage["text"]),
                        text=passage["text"],
                        title=title,
                    )
                )

        fetched_this_page = len(batch)
        offset += fetched_this_page
        if fetched_this_page < limit:
            break

    logger.info(
        "NIH RePORTER: fetched %d passage records for domain=%s year=%d",
        len(records),
        domain,
        fiscal_year,
    )
    return records


# ---------------------------------------------------------------------------
# Tool 2 — PubMed via metapub
# ---------------------------------------------------------------------------


def _build_pubmed_query(domain: str, query: Optional[str], year: Optional[int]) -> str:
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
    max_results: int = 500,
    email: str = "bioinsight@example.com",
) -> list[BioInsightRecord]:
    """
    Fetch PubMed abstracts for a given domain or query string.
    Each abstract is split into sentence-level passages at ingest.
    max_results controls how many abstracts to fetch (not passages).

    Args:
        domain: Disease / research area keyword.
        query: Optional explicit Entrez query (overrides domain-only search).
        year: Optional publication year filter (e.g. 2024).
        max_results: Upper bound on abstracts fetched (default 500).
        email: Email for NCBI rate-limit courtesy header.

    Returns:
        List of sentence-level BioInsightRecord objects with embedding=[].
    """
    fetch = PubMedFetcher(email=email, api_key=os.environ.get("NCBI_API_KEY"))
    search_str = _build_pubmed_query(domain, query, year)

    logger.info(
        "PubMed fetch  |  query='%s'  max_abstracts=%d", search_str, max_results
    )

    pmids: list[str] = _retry(
        lambda: fetch.pmids_for_query(search_str, retmax=max_results)
    )

    if not pmids:
        logger.warning("PubMed returned 0 PMIDs for query: %s", search_str)
        return []

    records: list[BioInsightRecord] = []

    for pmid in pmids:
        try:
            article = _retry(lambda: fetch.article_by_pmid(pmid))
        except Exception as exc:
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

        authors = article.authors or []
        pi_name = authors[0] if authors else "Unknown Author"
        journal = article.journal or "Unknown Journal"

        # Extract publication type
        pub_types = article.publication_types or []
        pub_type = (
            "review"
            if any("review" in pt.lower() for pt in pub_types)
            else (
                "letter"
                if any("letter" in pt.lower() for pt in pub_types)
                else (
                    "editorial"
                    if any("editorial" in pt.lower() for pt in pub_types)
                    else "journal_article"
                )
            )
        )

        raw_mesh = getattr(article, "mesh_headings", None) or []
        mesh_terms = "|".join(str(t) for t in raw_mesh if t)

        passages = split_into_passages(
            text=text,
            parent_id=str(pmid),
            source="pubmed",
        )

        for passage in passages:
            records.append(
                BioInsightRecord(
                    source="pubmed",
                    domain=domain.lower(),
                    year=pub_year,
                    pi_name=pi_name,
                    org_name=journal,
                    external_id=passage["passage_id"],
                    granularity=_infer_granularity(passage["text"]),
                    text=passage["text"],
                    title=title,
                    pub_type=pub_type,
                    mesh_terms=mesh_terms,
                )
            )

    logger.info(
        "PubMed: fetched %d passage records from %d abstracts for query='%s'",
        len(records),
        len(pmids),
        search_str,
    )
    return records
