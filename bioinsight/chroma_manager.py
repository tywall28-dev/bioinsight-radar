"""
BioInsight Agentic Radar — ChromaDB Manager
============================================
Manages the persistent vector library for PubMed and NIH RePORTER records.

Design Philosophy:
  - "Embed once, model at query-time."
  - Every record carries the 7-variable metadata guard so downstream
    UMAP/HDBSCAN can operate on clean, filtered subsets — never noisy
    full-corpus projections.
  - Vectors are stored as 768-D BioBERT embeddings (dmis-lab/biobert-v1.1).

Usage:
    from chroma_manager import BioInsightChromaManager

    manager = BioInsightChromaManager(persist_dir="./bioinsight_db")
    manager.upsert_records(records)
    subset = manager.get_subset(filters={"domain": "parkinsons", "year": 2026})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  The 7-Variable Metadata Schema
# ---------------------------------------------------------------------------

SourceType = Literal["pubmed", "nih_reporter"]
GranularityType = Literal["field", "protein", "pathway"]


@dataclass
class BioInsightRecord:
    """
    A single record destined for ChromaDB.

    All seven metadata fields are required.  The `text` field is the raw
    prose that will be embedded (abstract, title + aims, etc.).
    The `embedding` field is populated by the EmbeddingService before ingestion.
    """

    # --- Required metadata (7-variable guard) ---
    source: SourceType
    domain: str  # e.g. "parkinsons", "autism"
    year: int  # publication year or fiscal year
    pi_name: str  # lead author or principal investigator
    org_name: str  # institution / university
    external_id: str  # PMID or RePORTER ApplID (string)
    granularity: GranularityType  # scope of the record

    # --- Content fields ---
    text: str  # prose to embed
    title: str = ""  # human-readable label

    # --- Populated at ingestion time ---
    embedding: list[float] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Raise ValueError on missing or malformed required fields."""
        if not self.external_id:
            raise ValueError("external_id is required and must be non-empty.")
        if self.source not in ("pubmed", "nih_reporter"):
            raise ValueError(f"Invalid source: '{self.source}'")
        if self.granularity not in ("field", "protein", "pathway"):
            raise ValueError(f"Invalid granularity: '{self.granularity}'")
        if not self.text.strip():
            raise ValueError("text field must contain embeddable prose.")

    def chroma_id(self) -> str:
        """Stable, unique Chroma document ID."""
        return f"{self.source}__{self.external_id}"

    def metadata_dict(self) -> dict:
        """Return only the 7-variable metadata slice (no text / embedding)."""
        return {
            "source": self.source,
            "domain": self.domain,
            "year": self.year,
            "pi_name": self.pi_name,
            "org_name": self.org_name,
            "external_id": self.external_id,
            "granularity": self.granularity,
            "title": self.title,
        }


# ---------------------------------------------------------------------------
# 2.  ChromaDB Manager
# ---------------------------------------------------------------------------

COLLECTION_NAME = "bioinsight_library"


class BioInsightChromaManager:
    """
    Persistent ChromaDB collection manager.

    Parameters
    ----------
    persist_dir : str
        Directory where ChromaDB will store its DuckDB + Parquet files.
        Create it once; reuse it across sessions for true persistence.
    """

    def __init__(self, persist_dir: str = "./bioinsight_db") -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for BioBERT
        )
        logger.info(
            "ChromaDB collection '%s' ready  |  persist_dir=%s  |  count=%d",
            COLLECTION_NAME,
            persist_dir,
            self._collection.count(),
        )

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #

    def upsert_records(self, records: list[BioInsightRecord]) -> int:
        """
        Upsert a batch of fully-embedded BioInsightRecords.

        Validates each record, then writes ids / embeddings / documents /
        metadata in a single Chroma call.  Existing records with the same
        chroma_id are silently overwritten (idempotent).

        Returns the number of records successfully upserted.
        """
        if not records:
            logger.warning("upsert_records called with empty list — skipping.")
            return 0

        ids, embeddings, documents, metadatas = [], [], [], []

        for rec in records:
            try:
                rec.validate()
            except ValueError as exc:
                logger.error("Skipping record %s: %s", rec.external_id, exc)
                continue

            if not rec.embedding:
                logger.error(
                    "Skipping record %s: embedding is empty.  "
                    "Run EmbeddingService first.",
                    rec.external_id,
                )
                continue

            ids.append(rec.chroma_id())
            embeddings.append(rec.embedding)
            documents.append(rec.text)
            metadatas.append(rec.metadata_dict())

        if not ids:
            logger.error("No valid records to upsert after validation.")
            return 0

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Upserted %d records into '%s'.", len(ids), COLLECTION_NAME)
        return len(ids)

    # ------------------------------------------------------------------ #
    # Read path
    # ------------------------------------------------------------------ #
    def get_domain_subset(
        self,
        domain: str,
        years: list[int],
        source_filter: Optional[str] = None,
        include_embeddings: bool = True,
    ) -> dict:
        """
        Retrieve a filtered subset of the library for downstream modeling.

        Parameters
        ----------
        domain : str
            e.g. "parkinsons", "autism"
        years : list[int]
            Publication years or fiscal years to include.
        source_filter : str, optional
            "pubmed", "nih_reporter", or None for both.
        include_embeddings : bool
            Set True (default) when passing to UMAP/HDBSCAN.
            Set False for lightweight metadata inspection.
        Returns
        -------
        dict with keys: ids, documents, metadatas[, embeddings]
        """
        filters = {
            "$and": [
                {"domain": domain},
                {"year": {"$in": years}},
            ]
        }
        if source_filter in ("pubmed", "nih_reporter"):
            filters["$and"].append({"source": source_filter})

        return self.get_subset(filters=filters, include_embeddings=include_embeddings)

    def get_subset(
        self,
        filters: Optional[dict] = None,
        include_embeddings: bool = True,
    ) -> dict:
        """
        Retrieve a filtered subset of the library for downstream modeling.

        Parameters
        ----------
        filters : dict, optional
            ChromaDB $where clause. All keys must be valid metadata fields.
            Examples:
                {"domain": "parkinsons"}
                {"$and": [{"domain": "autism"}, {"year": {"$gte": 2024}}]}
        include_embeddings : bool
            Set True (default) when passing to UMAP/HDBSCAN.
            Set False for lightweight metadata inspection.

        Returns
        -------
        dict with keys: ids, documents, metadatas[, embeddings]
        """
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")

        kwargs: dict = {"include": include}
        if filters:
            kwargs["where"] = filters

        results = self._collection.get(**kwargs)
        count = len(results.get("ids", []))
        logger.info("get_subset returned %d records  |  filters=%s", count, filters)
        return results

    def semantic_search(
        self,
        query_embedding: list[float],
        k: int = 50,
        filters: Optional[dict] = None,
    ) -> dict:
        """
        Cosine-similarity search for Micro / RAG workflows.

        Parameters
        ----------
        query_embedding : list[float]
            768-D BioBERT vector of the query string.
        k : int
            Number of nearest neighbours to return.
        filters : dict, optional
            Pre-filter the corpus before ANN search.
        """
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        results = self._collection.query(**kwargs)
        logger.info("semantic_search  k=%d  |  filters=%s", k, filters)
        return results

    # ------------------------------------------------------------------ #
    # Introspection helpers (used by Library Checker Node)
    # ------------------------------------------------------------------ #

    def count(self, filters: Optional[dict] = None) -> int:
        """Return record count, optionally scoped to a metadata filter."""
        if not filters:
            return self._collection.count()
        result = self._collection.get(where=filters, include=[])
        return len(result.get("ids", []))

    def semantic_search_by_year(
        self,
        query_vector: list[float],
        year: int,
        min_records: int = 10,
        min_similarity: float = 0.5,
    ) -> bool:
        """Check if the library has at least `min_records` relevant to the query vector for a given year.

        Parameters
        ----------
        query_vector : list[float]
            768-D BioBERT embedding of the query.
        year : int
            Year to filter records by.
        min_records : int, default 10
            Minimum number of relevant records required.
        min_similarity : float, default 0.5
            Minimum cosine similarity threshold (0.0 to 1.0). Records with similarity
            below this threshold are not counted.
        """
        filters = {"year": year}
        # Get more results than needed to account for filtering
        results = self.semantic_search(
            query_embedding=query_vector,
            k=min_records * 2,  # Get more to filter
            filters=filters,
        )

        # Filter by similarity threshold (cosine distance < 1 - min_similarity)
        distances = results.get("distances", [[]])[0]
        max_distance = 1.0 - min_similarity
        relevant_count = sum(1 for d in distances if d <= max_distance)

        logger.info(
            "Library check: year=%d  found=%d  threshold=%d  meets_threshold=%s  (min_similarity=%.2f)",
            year,
            relevant_count,
            min_records,
            relevant_count >= min_records,
            min_similarity,
        )
        return relevant_count >= min_records

    def list_domains(self) -> list[str]:
        """Return sorted list of all unique domain values in the library."""
        results = self._collection.get(include=["metadatas"])
        domains = sorted({m["domain"] for m in results.get("metadatas", []) if m})
        return domains

    def collection_summary(self) -> dict:
        """High-level stats for logging / dashboard display."""
        results = self._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", []) or []
        total = len(metadatas)
        sources = {}
        domains = {}
        years = {}
        for m in metadatas:
            if not m:
                continue
            sources[m.get("source", "unknown")] = (
                sources.get(m.get("source", "unknown"), 0) + 1
            )
            domains[m.get("domain", "unknown")] = (
                domains.get(m.get("domain", "unknown"), 0) + 1
            )
            years[str(m.get("year", "unknown"))] = (
                years.get(str(m.get("year", "unknown")), 0) + 1
            )
        return {
            "total_records": total,
            "by_source": sources,
            "by_domain": domains,
            "by_year": years,
        }
