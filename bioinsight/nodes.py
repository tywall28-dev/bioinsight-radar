import io
import os
import json
import re
from collections import Counter
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from bioinsight.state import AgentState
from bioinsight.chroma_manager import BioInsightChromaManager
from bioinsight.fetcher_tools import fetch_pubmed, fetch_nih_reporter
from bioinsight.embedder import BioInsightEmbedder
import umap
import hdbscan
import numpy as np
import logging
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001", temperature=0
)  # fast, for routing, extracting, and verifying
synthesis_llm = ChatAnthropic(model="claude-sonnet-4-6")  # powerful, for reports
chroma = BioInsightChromaManager(persist_dir="./bioinsight_db")
embedder = BioInsightEmbedder(model_name="dmis-lab/biobert-v1.1")


_TERM_STOPWORDS = {
    "the", "a", "an", "of", "in", "and", "or", "for", "to", "with", "by",
    "on", "is", "are", "that", "this", "from", "at", "as", "its", "into",
    "via", "role", "using", "based", "study", "studies", "research", "new",
    "analysis", "effects", "effect", "related", "associated", "between",
}


def _extract_title_terms(title: str) -> list[str]:
    """Extract informative words from a grant/paper title (NIH fallback)."""
    words = title.split()
    out = []
    for w in words:
        w = w.strip(".,;:()[]\"'-")
        if len(w) >= 4 and w.lower() not in _TERM_STOPWORDS and not w.isdigit():
            out.append(w)
    return out


def router_node(state: AgentState) -> dict:
    prompt = f"""You are parsing a biomedical research question to query PubMed and NIH Reporter.

    Your tasks:
    1. Enrich the query for vector embedding (no years, just concepts and keywords)
    2. Extract 2-5 specific biomedical search terms suitable for PubMed/NIH APIs.
       - Each term must be 1-3 words maximum — a real MeSH heading or standard biomedical keyword.
       - Good: ["LRRK2", "Parkinson Disease", "alpha-synuclein"] Bad: ["Parkinson genetic risk factors", "dopamine pathway signaling cascade"]
       - First term must be the primary disease/topic anchor (e.g. "Parkinson Disease").
       - Terms must be about the BIOMEDICAL TOPIC only. Never include "grants", "funding", "publications", "research".
    3. Extract structured metadata

    Return ONLY valid JSON with this exact structure:
    {{
        "enriched_query": "...",
        "search_terms": ["keyword1", "keyword2"],
        "domain": "...",
        "years": [...],
        "entity": "...",
        "specificity": 1-5,
        "source_filter": "pubmed|nih_reporter|both"
    }}

    User question: {state["user_query"]}"""

    response = llm.invoke(prompt)
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()
    parsed = json.loads(content)

    try:
        query_vector = embedder.embed_query(parsed["enriched_query"])
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        query_vector = [0.0] * 768

    return {
        "query_vector": query_vector,
        "search_terms": parsed.get("search_terms", []),
        "enriched_query": parsed["enriched_query"],
        "domain": parsed["domain"],
        "years": parsed["years"],
        "entity": parsed.get("entity"),
        "specificity": parsed["specificity"],
        "source_filter": parsed["source_filter"],
    }


def library_checker_node(state: AgentState) -> dict:
    years = state["years"]
    query_vector = state.get("query_vector")
    specificity = state.get("specificity", 3)
    min_similarity = 0.2 + (specificity * 0.08)

    test_limit = int(os.environ.get("BIOINSIGHT_TEST_LIMIT", 0))
    required_docs = test_limit if test_limit > 0 else 50

    if specificity >= 4 and state.get("fetch_attempts", 0) == 0:
        return {"library_has_data": False}

    library_has_data = all(
        chroma.semantic_search_by_year(
            query_vector, year, min_records=required_docs, min_similarity=min_similarity
        )
        for year in years
    )

    return {"library_has_data": library_has_data}


def _build_query_pairs(search_terms: list, fallback: str) -> list[tuple[str, str]]:
    """
    Turn a list of search terms into (pubmed_query, nih_query) pairs.

    Strategy: run the anchor term alone for broad coverage, then pair each
    secondary term with the anchor for focused depth. This avoids the massive
    OR chain that confuses both APIs and produces irrelevant results.

    PubMed uses Entrez boolean syntax: "Parkinson Disease AND neuroinflammation"
    NIH Reporter uses space-separated keywords (all must appear): "Parkinson Disease neuroinflammation"
    """
    if not search_terms:
        return [(fallback, fallback)]

    anchor = search_terms[0]
    pairs = [(anchor, anchor)]  # broad anchor-only pass

    for term in search_terms[1:]:
        pubmed_q = f"{anchor} AND {term}"
        nih_q = f"{anchor} {term}"
        pairs.append((pubmed_q, nih_q))

    return pairs


def fetcher_node(state: AgentState) -> dict:
    search_terms = state.get("search_terms", [])
    years = state["years"]
    source_filter = state.get("source_filter")

    test_limit = int(os.environ.get("BIOINSIGHT_TEST_LIMIT", 0))
    fetch_limit = test_limit if test_limit > 0 else 100

    query_pairs = _build_query_pairs(
        search_terms, fallback=state.get("enriched_query", "")
    )

    # Spread the fetch budget across queries; anchor gets the full budget,
    # secondary passes get a proportional share (min 20 to stay meaningful).
    n_secondary = max(len(query_pairs) - 1, 1)
    secondary_limit = max(20, fetch_limit // n_secondary)

    logger.info(
        "Fetcher: %d query pairs, anchor_limit=%d, secondary_limit=%d, years=%s",
        len(query_pairs),
        fetch_limit,
        secondary_limit,
        years,
    )

    total_records = 0

    for year in years:
        for i, (pubmed_q, nih_q) in enumerate(query_pairs):
            limit = fetch_limit if i == 0 else secondary_limit
            all_records = []

            if source_filter in ("pubmed", "both", None):
                try:
                    records = fetch_pubmed.invoke(
                        {"domain": pubmed_q, "year": year, "max_results": limit}
                    )
                    all_records += records
                    logger.info("PubMed '%s' → %d passages", pubmed_q, len(records))
                except Exception as e:
                    logger.warning("PubMed fetch failed for '%s': %s", pubmed_q, e)

            if source_filter in ("nih_reporter", "both", None):
                try:
                    records = fetch_nih_reporter.invoke(
                        {"domain": nih_q, "fiscal_year": year, "max_results": limit}
                    )
                    all_records += records
                    logger.info("NIH '%s' → %d passages", nih_q, len(records))
                except Exception as e:
                    logger.warning("NIH fetch failed for '%s': %s", nih_q, e)

            if all_records:
                embedded = embedder.embed_records(all_records)
                chroma.upsert_records(embedded)
                total_records += len(all_records)

    return {
        "records_fetched": total_records,
        "fetch_attempts": state.get("fetch_attempts", 0) + 1,
    }


def material_assessor_node(state: AgentState) -> dict:
    """
    LLM reviews a sample of the fetched data and decides whether it is
    sufficient to proceed or whether additional fetching is warranted.
    Returns a prose assessment, a recommended action, and optional new
    search terms to pursue.
    """
    query_vector = state.get("query_vector")
    enriched_query = state.get("enriched_query", "")
    years = state["years"]
    source_filter = state.get("source_filter")
    records_fetched = state.get("records_fetched", 0)

    year_filter = {"year": {"$in": years}}
    if source_filter in ("pubmed", "nih_reporter"):
        filters = {"$and": [{"year": {"$in": years}}, {"source": source_filter}]}
    else:
        filters = year_filter

    # Fetch a large candidate pool — at sentence-level splitting,
    # 1,000 grants ≈ 15,000 passages, so k=3000 covers the top ~20%
    results = chroma.semantic_search(query_vector, k=3000, filters=filters)
    passages = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    parent_map: dict[str, dict] = {}
    for pid, text, meta in zip(ids, passages, metadatas):
        parent_id = pid.rsplit("__s", 1)[0]
        if parent_id not in parent_map:
            parent_map[parent_id] = {
                "pi_name": meta.get("pi_name", "Unknown"),
                "year": meta.get("year", ""),
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "mesh_terms": meta.get("mesh_terms", ""),
                "sentences": [],
            }
        parent_map[parent_id]["sentences"].append((pid, text))

    total_unique_docs = len(parent_map)
    total_passages_in_scope = chroma.count(filters)

    # Compute structured coverage breakdown
    pubmed_docs = 0
    nih_docs = 0
    year_dist: Counter = Counter()
    term_counter: Counter = Counter()
    search_terms = state.get("search_terms") or []

    for doc in parent_map.values():
        src = doc.get("source", "")
        yr = doc.get("year", "")
        if src == "pubmed":
            pubmed_docs += 1
        elif src == "nih_reporter":
            nih_docs += 1
        if yr:
            year_dist[str(yr)] += 1
        mesh = doc.get("mesh_terms", "")
        if mesh:
            for t in mesh.split("|"):
                t = t.strip()
                if t:
                    term_counter[t] += 1
        else:
            for w in _extract_title_terms(doc.get("title", "")):
                term_counter[w] += 1

    top_terms = [{"term": t, "count": c} for t, c in term_counter.most_common(20)]

    # Which query terms appear in the top-term vocabulary?
    top_term_set = {t["term"].lower() for t in top_terms}
    query_term_coverage = {
        st: any(st.lower() in tt or tt in st.lower() for tt in top_term_set)
        for st in search_terms
    }

    coverage_breakdown = {
        "total_unique_docs": total_unique_docs,
        "total_passages_in_scope": total_passages_in_scope,
        "pubmed_docs": pubmed_docs,
        "nih_docs": nih_docs,
        "year_distribution": dict(sorted(year_dist.items())),
        "top_terms": top_terms,
        "query_term_coverage": query_term_coverage,
    }

    # Show top 30 most relevant parent docs to the LLM
    doc_blocks = []
    for doc in list(parent_map.values())[:30]:
        doc["sentences"].sort(key=lambda x: int(x[0].rsplit("__s", 1)[1]))
        body = " ".join(t for _, t in doc["sentences"])
        doc_blocks.append(
            f"({doc['pi_name']}, {doc['year']}) — {doc['title']}\n{body}"
        )
    sample_text = "\n\n---\n\n".join(doc_blocks)

    prompt = f"""You are a biomedical research intelligence analyst assessing a dataset before deep analysis.

Query: {enriched_query}
Years: {years}
Source: {source_filter}

Library scope stats:
- Total passages matching year/source filter: {total_passages_in_scope}
- Unique source documents retrieved by semantic search: {total_unique_docs}
- Documents shown below (top 30 by relevance): {min(30, total_unique_docs)}

Top 30 most relevant documents:
{sample_text}

Assess this dataset and respond in valid JSON only:
{{
  "assessment": "2-3 sentence qualitative assessment: is the data relevant and deep, or superficial? what topics are well-covered across these {total_unique_docs} documents? what seems missing?",
  "coverage_score": 1-5,
  "action": "proceed" or "fetch_more",
  "reasoning": "one sentence explaining the action recommendation",
  "suggested_terms": ["ShortMeSHTerm1", "ShortMeSHTerm2"]
}}

Action rules:
- "proceed" if the data covers the query topic with reasonable depth across these {total_unique_docs} documents.
- "fetch_more" ONLY if there are clear, specific gaps that different search terms would fill. Be conservative.

Suggested term rules (CRITICAL):
- Each suggested term must be 1-3 words maximum — a real MeSH term or standard biomedical keyword.
- Good examples: "neuroinflammation", "LRRK2", "GBA mutation", "alpha-synuclein", "dopamine transporter"
- Bad examples: "neuroinflammation cytokines IL-6 TNF-alpha", "genetic risk factors GBA PINK1 DJ-1"
- Never suggest multi-word phrases longer than 3 words. Each term runs as its own focused API query."""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content.strip())
        logger.info(
            f"Material assessment: score={parsed.get('coverage_score')}/5  action={parsed.get('action')}"
        )
        return {
            "material_assessment": parsed.get("assessment", ""),
            "assessment_action": parsed.get("action", "proceed"),
            "suggested_terms": parsed.get("suggested_terms") or [],
            "coverage_breakdown": coverage_breakdown,
        }
    except Exception as e:
        logger.error(f"Material assessor failed: {e}")
        return {
            "material_assessment": "Assessment unavailable.",
            "assessment_action": "proceed",
            "suggested_terms": [],
            "coverage_breakdown": coverage_breakdown,
        }


def prelim_report_node(state: AgentState) -> dict:
    """
    Generates a fast preliminary overview of the data — no clustering,
    no extraction — just a direct LLM summary of the top documents.
    Shown to the human before they commit to the full analysis pipeline.
    """
    query_vector = state.get("query_vector")
    enriched_query = state.get("enriched_query", "")
    years = state["years"]
    source_filter = state.get("source_filter")

    year_filter = {"year": {"$in": years}}
    if source_filter in ("pubmed", "nih_reporter"):
        filters = {"$and": [{"year": {"$in": years}}, {"source": source_filter}]}
    else:
        filters = year_filter

    # k=2000 gives a meaningful sample even at thousands-of-grants scale
    results = chroma.semantic_search(query_vector, k=2000, filters=filters)
    passages = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    parent_map: dict[str, dict] = {}
    for pid, text, meta in zip(ids, passages, metadatas):
        parent_id = pid.rsplit("__s", 1)[0]
        if parent_id not in parent_map:
            parent_map[parent_id] = {
                "pi_name": meta.get("pi_name", "Unknown"),
                "year": meta.get("year", ""),
                "title": meta.get("title", ""),
                "sentences": [],
            }
        parent_map[parent_id]["sentences"].append((pid, text))

    total_unique_docs = len(parent_map)

    # Show top 25 most relevant parent docs; Haiku handles this easily
    doc_blocks = []
    for doc in list(parent_map.values())[:25]:
        doc["sentences"].sort(key=lambda x: int(x[0].rsplit("__s", 1)[1]))
        body = " ".join(t for _, t in doc["sentences"])
        doc_blocks.append(f"({doc['pi_name']}, {doc['year']}): {body}")
    sample_text = "\n\n".join(doc_blocks)

    prompt = f"""You are a biomedical research analyst. Write a SHORT preliminary overview (200-250 words) of this dataset for a program officer deciding whether to run a full analysis.

Query: {enriched_query}
Years: {years}
Source: {source_filter}
Total unique source documents in scope: {total_unique_docs} (showing top 25 by relevance below)

Top 25 most relevant documents:
{sample_text}

Write the overview in plain prose with these three parts:
1. **What's here** — the main research themes visible across these {total_unique_docs} documents
2. **Depth** — how focused/deep vs. broad/shallow the coverage appears
3. **Potential gaps** — anything the query asks about that seems absent

Be direct and specific. Name actual topics, proteins, mechanisms, PI names you can see. Do not pad."""

    try:
        response = llm.invoke(prompt)
        return {"prelim_report": response.content.strip()}
    except Exception as e:
        logger.error(f"Prelim report failed: {e}")
        return {"prelim_report": "Preliminary overview unavailable."}


def subset_modeler_node(state: AgentState) -> dict:
    query_vector = state.get("query_vector")
    years = state["years"]
    source_filter = state.get("source_filter")
    specificity = state.get("specificity", 3)

    target_docs = max(50, 200 - (specificity - 1) * 30)
    k = target_docs * 15
    k = min(k, 5000)

    year_filter = {"year": {"$in": years}}
    if source_filter in ("pubmed", "nih_reporter"):
        filters = {"$and": [{"year": {"$in": years}}, {"source": source_filter}]}
    else:
        filters = year_filter

    results = chroma.semantic_search(query_vector, k=k, filters=filters)
    if not results.get("documents") or len(results["documents"][0]) == 0:
        return {"clusters": {}, "error": "No records found for this query."}

    embeddings_matrix = np.array(results["embeddings"][0])

    n_samples = embeddings_matrix.shape[0]
    n_neighbors = min(15, n_samples - 1) if n_samples > 1 else 1
    min_cluster_size = min(5, n_samples) if n_samples > 0 else 2

    umap_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=0.1, n_components=5
    ).fit_transform(embeddings_matrix)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)
    cluster_labels = clusterer.fit_predict(umap_embeddings)

    clusters = {}
    for i, (label, document) in enumerate(zip(cluster_labels, results["documents"][0])):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(
            {
                "id": results["ids"][0][i],
                "pi_name": results["metadatas"][0][i].get("pi_name"),
                "title": results["metadatas"][0][i].get("title"),
                "year": results["metadatas"][0][i].get("year"),
                "text": document,
                "metadata": results["metadatas"][0][i],
            }
        )

    return {"clusters": clusters}


def _group_passages_by_parent(passages: list, max_docs: int = 6) -> list[dict]:
    """
    Reconstruct document-level context from sentence passages.

    Returns a list of dicts, each representing one source document with all
    its sentences concatenated. This gives the LLM paragraph-level context
    instead of isolated sentences, making extraction and verification reliable.
    """
    parent_map: dict[str, dict] = {}
    for p in passages:
        parent_id = p["id"].rsplit("__s", 1)[0]
        if parent_id not in parent_map:
            parent_map[parent_id] = {
                "parent_id": parent_id,
                "pi_name": p["metadata"].get("pi_name", "Unknown Author"),
                "year": p["metadata"].get("year", "Unknown Year"),
                "title": p["metadata"].get("title", ""),
                "source": p["metadata"].get("source", ""),
                "sentences": [],
                "ids": [],
            }
        parent_map[parent_id]["sentences"].append((p["id"], p["text"]))

    docs = []
    for doc in list(parent_map.values())[:max_docs]:
        # Sort sentences by their index so prose reads in order
        doc["sentences"].sort(key=lambda x: int(x[0].rsplit("__s", 1)[1]))
        docs.append(doc)
    return docs


def extraction_node(state: AgentState) -> dict:
    """Extracts raw claims from the clusters, parallelized across clusters."""
    enriched_query = state.get("enriched_query", state.get("domain", ""))
    source_filter = state.get("source_filter")
    clusters = state.get("clusters", {})

    # Build full evidence index up front (fast, no LLM)
    evidence_index = {}
    for label, passages in clusters.items():
        for p in passages:
            evidence_index[p["id"]] = {
                "text": p["text"],
                "source": p["metadata"].get("source"),
                "year": p["metadata"].get("year"),
                "title": p["metadata"].get("title"),
                "pi_name": p["metadata"].get("pi_name"),
                "external_id": p["metadata"].get("external_id"),
                "url": (
                    f"https://pubmed.ncbi.nlm.nih.gov/{p['metadata'].get('external_id').split('__')[1]}/"
                    if p["metadata"].get("source") == "pubmed"
                    else f"https://reporter.nih.gov/project-details/{p['metadata'].get('external_id').split('__')[1]}"
                ),
            }

    def extract_cluster(label: int, passages: list) -> tuple[int, list]:
        docs = _group_passages_by_parent(passages, max_docs=6)

        # Build context: each doc block shows the full reconstructed abstract
        doc_blocks = []
        for doc in docs:
            text_body = " ".join(sent for _, sent in doc["sentences"])
            # Use the first sentence ID as the citation anchor
            anchor_id = doc["sentences"][0][0]
            doc_blocks.append(
                f"[{anchor_id}] ({doc['pi_name']}, {doc['year']}) — {doc['title']}\n{text_body}"
            )
        passages_text = "\n\n---\n\n".join(doc_blocks)

        finding_prompt = f"""You are analyzing biomedical research documents to extract structured findings.
Original query: {enriched_query}
Source: {source_filter}

Documents (each block is one grant/paper; the ID in brackets is the citation anchor):
{passages_text}

Extract 2-3 specific findings that are explicitly stated in the text above.
Rules:
- Only extract claims that are clearly asserted as established facts or confirmed results, NOT research aims, hypotheses, or future plans ("will test", "aims to", "we expect").
- Use the ID in brackets as the evidence_id, and construct citation_text from the Author/Year shown.
- Be specific — avoid vague generalizations.

Return ONLY valid JSON, no other text:
{{"findings": [
    {{
        "claim": "one specific factual claim explicitly stated in the documents",
        "evidence_ids": ["nih_reporter__12345__s0"],
        "citation_text": "[Smith et al., 2024]",
        "claim_type": "finding"
    }}
]}}"""

        response = llm.invoke(finding_prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        try:
            parsed = json.loads(content.strip())
            return label, parsed.get("findings", [])
        except Exception as e:
            logger.warning(f"Failed to parse findings for cluster {label}: {e}")
            return label, []

    cluster_findings = {}
    work = [(label, passages) for label, passages in clusters.items() if label != -1]
    logger.info(f"Extraction: processing {len(work)} clusters sequentially")

    for label, passages in work:
        label, findings = extract_cluster(label, passages)
        cluster_findings[label] = findings

    return {"cluster_findings": cluster_findings, "evidence_index": evidence_index}


def verifier_node(state: AgentState) -> dict:
    """Verifies extracted claims against grouped document context, parallelized."""
    clusters = state.get("clusters", {})
    extracted_findings = state.get("cluster_findings", {})

    def verify_cluster(label: int, passages: list) -> tuple[int, list]:
        claims_to_check = extracted_findings.get(label, [])
        if not claims_to_check:
            return label, []

        # Same grouped context as extraction so the verifier sees full abstracts
        docs = _group_passages_by_parent(passages, max_docs=6)
        doc_blocks = []
        for doc in docs:
            text_body = " ".join(sent for _, sent in doc["sentences"])
            anchor_id = doc["sentences"][0][0]
            doc_blocks.append(f"[{anchor_id}] ({doc['pi_name']}, {doc['year']}): {text_body}")
        source_text = "\n\n---\n\n".join(doc_blocks)

        claims_json_str = json.dumps(claims_to_check, indent=2)

        verifier_prompt = f"""You are a strict, objective fact-checker.
Verify each CLAIM against the SOURCE DOCUMENTS below.

SOURCE DOCUMENTS (each block is one full grant abstract or paper):
{source_text}

CLAIMS TO VERIFY:
{claims_json_str}

Rules:
1. Use ONLY the provided source documents.
2. A claim is supported only if the document text explicitly states it as an established fact or confirmed result.
3. Claims based on research aims ("will test", "aims to", "we hypothesize") are NOT supported — mark them false.
4. Keep the exact same order as the input claims.

Return ONLY a valid JSON array, same length as CLAIMS TO VERIFY:
[
  {{
    "is_supported": true,
    "reasoning": "The document explicitly states..."
  }}
]"""

        try:
            response = llm.invoke(verifier_prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            evaluation_results = json.loads(content.strip())

            approved_claims = []
            for i, eval_result in enumerate(evaluation_results):
                if i >= len(claims_to_check):
                    break
                if eval_result.get("is_supported") is True:
                    approved_claims.append(claims_to_check[i])
                else:
                    logger.warning(
                        f"Dropped (cluster {label}): {claims_to_check[i].get('claim')} | {eval_result.get('reasoning', '')}"
                    )
            return label, approved_claims

        except Exception as e:
            logger.error(f"Failed to verify cluster {label}: {e}")
            return label, []

    verified_findings = {}
    work = [(label, passages) for label, passages in clusters.items() if label != -1 and label in extracted_findings]
    logger.info(f"Verification: verifying {len(work)} clusters sequentially")

    for label, passages in work:
        label, approved = verify_cluster(label, passages)
        verified_findings[label] = approved

    return {"verified_cluster_findings": verified_findings}


def _add_hyperlink(paragraph, text: str, url: str):
    """Insert a clickable hyperlink into a docx paragraph."""
    part = paragraph.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    style = OxmlElement("w:rStyle")
    style.set(qn("w:val"), "Hyperlink")
    rpr.append(style)
    run.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    run.append(t)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def _write_inline(paragraph, text: str, citation_url_map: dict):
    """Write a line of text into a paragraph, turning [Author, Year](url) into hyperlinks."""
    token_pattern = re.compile(r"(\[[^\]]+\])(\(https?://[^\)]+\))?")
    cursor = 0
    for m in token_pattern.finditer(text):
        if m.start() > cursor:
            paragraph.add_run(text[cursor:m.start()])
        citation_text = m.group(1)
        url_part = m.group(2)
        if url_part:
            _add_hyperlink(paragraph, citation_text, url_part[1:-1])
        elif citation_text in citation_url_map:
            _add_hyperlink(paragraph, citation_text, citation_url_map[citation_text])
        else:
            paragraph.add_run(citation_text)
        cursor = m.end()
    if cursor < len(text):
        paragraph.add_run(text[cursor:])


def _flush_table(doc, table_rows: list, citation_url_map: dict):
    """Flush a buffered markdown table into a docx table."""
    if not table_rows:
        return
    num_cols = max(len(r) for r in table_rows)
    t = doc.add_table(rows=len(table_rows), cols=num_cols)
    t.style = "Table Grid"
    for r_idx, row in enumerate(table_rows):
        for c_idx in range(num_cols):
            cell_text = row[c_idx].strip() if c_idx < len(row) else ""
            cell = t.rows[r_idx].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            _write_inline(p, cell_text, citation_url_map)
            if r_idx == 0:
                for run in p.runs:
                    run.bold = True


def _parse_table_row(line: str):
    """Return list of cell strings if line is a markdown table row, else None."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    cells = stripped.strip("|").split("|")
    # Separator row: all cells are dashes/colons only
    if all(re.fullmatch(r"[\s:\-]+", c) for c in cells):
        return []  # empty list = separator, skip
    return cells


def _markdown_to_docx(markdown_text: str, citation_url_map: dict) -> bytes:
    """Convert the markdown report to a .docx file with tables and hyperlinks."""
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    table_buffer: list[list[str]] = []

    def flush():
        if table_buffer:
            _flush_table(doc, table_buffer, citation_url_map)
            table_buffer.clear()

    for line in markdown_text.splitlines():
        raw = line.strip()

        # --- Table rows ---
        table_row = _parse_table_row(raw)
        if table_row is not None:
            if table_row:  # non-separator
                table_buffer.append(table_row)
            # separator row: skip but keep buffering
            continue
        else:
            flush()  # end of table region

        # --- Headings ---
        if raw.startswith("#### "):
            doc.add_heading(raw[5:], level=4)
        elif raw.startswith("### "):
            doc.add_heading(raw[4:], level=3)
        elif raw.startswith("## "):
            doc.add_heading(raw[3:], level=2)
        elif raw.startswith("# "):
            doc.add_heading(raw[2:], level=1)
        elif raw.startswith("---"):
            doc.add_paragraph("─" * 60)
        elif raw.startswith("- ") or raw.startswith("* "):
            p = doc.add_paragraph(style="List Bullet")
            _write_inline(p, raw[2:], citation_url_map)
        elif raw == "":
            doc.add_paragraph("")
        else:
            p = doc.add_paragraph()
            _write_inline(p, raw, citation_url_map)

    flush()  # flush any trailing table

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def report_writer_node(state: AgentState) -> dict:
    """Writes the final report using only verified claims, split by data source."""
    enriched_query = state.get("enriched_query", state.get("domain", ""))
    years = state["years"]
    source_filter = state.get("source_filter")
    specificity = state.get("specificity")
    clusters = state.get("clusters", {})
    verified_findings = state.get("verified_cluster_findings", {})
    evidence_index = state.get("evidence_index", {})

    total_passages = sum(len(passages) for passages in clusters.values())
    unique_docs = len(
        set(
            p["id"].rsplit("__s", 1)[0]
            for passages in clusters.values()
            for p in passages
        )
    )

    # Build citation_text → url map for post-processing
    citation_url_map = {}
    for findings in verified_findings.values():
        for f in findings:
            citation_text = f.get("citation_text")
            evidence_ids = f.get("evidence_ids", [])
            if citation_text and evidence_ids:
                url = evidence_index.get(evidence_ids[0], {}).get("url")
                if url:
                    citation_url_map[citation_text] = url

    # Tag each finding with its data source so the LLM can separate them
    tagged_findings = {}
    pubmed_count = 0
    nih_count = 0
    for label, findings in verified_findings.items():
        tagged = []
        for f in findings:
            evidence_ids = f.get("evidence_ids", [])
            src = "unknown"
            if evidence_ids:
                src = evidence_index.get(evidence_ids[0], {}).get("source", "unknown")
            if src == "pubmed":
                pubmed_count += 1
            elif src == "nih_reporter":
                nih_count += 1
            tagged.append({**f, "data_source": src})
        tagged_findings[label] = tagged

    all_findings_text = json.dumps(tagged_findings, indent=2)

    has_pubmed = pubmed_count > 0
    has_nih = nih_count > 0

    source_context = ""
    if has_pubmed and has_nih:
        source_context = (
            "The findings come from BOTH published literature (data_source=pubmed) "
            "and NIH-funded grants (data_source=nih_reporter). "
            "Write separate sections for each."
        )
    elif has_pubmed:
        source_context = "All findings come from published literature (PubMed)."
    elif has_nih:
        source_context = "All findings come from NIH-funded grant abstracts (NIH Reporter)."

    synthesis_prompt = f"""You are a senior biomedical research analyst writing an intelligence report for a program officer who needs to make funding decisions.

Query: {enriched_query}
Years: {years}
{source_context}

VERIFIED FINDINGS ({len(verified_findings)} clusters, {pubmed_count + nih_count} total findings):
Each finding includes a "data_source" field: "pubmed" = published literature, "nih_reporter" = active/recent grant funding.
{all_findings_text}

---

Write a structured report in EXACTLY this order. Use markdown headings.

# BioInsight Report: {enriched_query}

## Executive Summary
Write 3-5 bullet points answering "so what does this data tell a funder?". Be direct and specific — name actual targets, drugs, mechanisms, or gaps. Each bullet should contain a citation.

## Key Entities at a Glance
Produce a markdown table of the most important named biological entities (proteins, genes, drugs, pathways, biomarkers, cell types) that appear in the verified findings. Only include entities explicitly named in the findings below.

| Entity | Type | Key Finding | Citation |
|--------|------|-------------|----------|
(fill rows)

{f'''## What Published Research Shows
Summarize findings where data_source is "pubmed". Group by theme. After every claim cite using the citation_text exactly.
Aim for 3-5 themes with 1-3 sentences each.

## Where Grant Funding is Going
Summarize findings where data_source is "nih_reporter". Group by theme. After every claim cite using the citation_text exactly.
Highlight where grant activity aligns with or diverges from the published literature above.
Aim for 3-5 themes with 1-3 sentences each.''' if has_pubmed and has_nih else f'''## Research Findings by Theme
Group findings by theme. After every claim cite using the citation_text exactly. Aim for 3-5 themes.'''}

## Research Gaps & White Space
What important aspects of "{enriched_query}" are absent or under-represented across ALL the verified findings? Be specific — name what is missing, not just "more research is needed."

## Strategic Summary
2-3 sentences. What is the single most important insight, and what would a well-targeted next investment look like based on these findings?

---

CRITICAL RULES:
- Use citation_text EXACTLY as it appears in the JSON (e.g., [Smith et al., 2024]). Never paraphrase or reformat it.
- Every claim must have a citation from the findings. No unsupported statements.
- Do NOT use raw system IDs (nih_reporter__, pubmed__, etc.) anywhere in the output.
- For the entity table: only rows for entities you can cite. Leave no uncited rows."""

    response = synthesis_llm.invoke(synthesis_prompt)

    # Post-process: [Author, Year] → [Author, Year](url) for Markdown links
    report_text = response.content
    for citation, url in citation_url_map.items():
        report_text = report_text.replace(citation, f"{citation}({url})")

    diagnostic = f"""---
**Analysis Metadata** — Source: {source_filter} | Years: {years} | Unique docs: {unique_docs} | Passages: {total_passages} | Clusters: {len(verified_findings)}

---

"""
    full_report = diagnostic + report_text

    report_docx = _markdown_to_docx(full_report, citation_url_map)

    sidecar = {
        "query": enriched_query,
        "years": years,
        "source": source_filter,
        "cluster_findings": verified_findings,
        "evidence_index": evidence_index,
    }

    return {
        "final_answer": full_report,
        "report_sidecar": json.dumps(sidecar, indent=2),
        "report_docx": report_docx,
    }


def error_node(state: AgentState) -> dict:
    attempts = state.get("fetch_attempts", 0)
    search_terms = state.get("search_terms", [])
    return {
        "final_answer": f"Could not find sufficient relevant data after {attempts} fetch attempts. "
        f"Search terms used: {search_terms}. "
        f"Try broadening your query, using different terminology, or expanding the date range."
    }
