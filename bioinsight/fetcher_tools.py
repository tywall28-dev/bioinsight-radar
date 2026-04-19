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
# Tool 1 — NIH RePORTER v2
# ---------------------------------------------------------------------------

NIH_REPORTER_URL = "https://api.reporter.nih.gov/v2/projects/search"
NIH_PAGE_SIZE = 500


def _nih_reporter_page(
    domain: str,
    fiscal_year: int,
    offset: int,
    limit: int,
) -> list[dict]:
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
    return resp.json().get("results", [])


@tool
def fetch_nih_reporter(
    domain: str,
    fiscal_year: int,
    max_results: int = 500,
) -> list[BioInsightRecord]:
    """
    Fetch NIH RePORTER grants for a given domain and fiscal year.
    Each grant abstract is split into sentence-level passages at ingest.

    Args:
        domain: Disease or research area keyword (e.g. "Parkinson Disease").
        fiscal_year: NIH fiscal year (e.g. 2024).
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
    return records[:max_results]


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
    max_results: int = 200,
    email: str = "bioinsight@example.com",
) -> list[BioInsightRecord]:
    """
    Fetch PubMed abstracts for a given domain or query string.
    Each abstract is split into sentence-level passages at ingest.

    Args:
        domain: Disease / research area keyword.
        query: Optional explicit Entrez query (overrides domain-only search).
        year: Optional publication year filter (e.g. 2024).
        max_results: Upper bound on records returned (default 200).
        email: Email for NCBI rate-limit courtesy header.

    Returns:
        List of BioInsightRecord objects with embedding=[] ready for
        EmbeddingService processing.
    """
    fetch = PubMedFetcher(email=email, api_key=os.environ.get("NCBI_API_KEY"))
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
                )
            )

    logger.info(
        "PubMed: fetched %d passage records for query='%s'",
        len(records),
        search_str,
    )
    return records
