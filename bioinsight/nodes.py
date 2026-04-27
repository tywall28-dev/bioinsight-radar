import io
import os
import json
import re
from collections import Counter
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from bioinsight.state import AgentState
from bioinsight.chroma_manager import BioInsightChromaManager
from bioinsight.fetcher_tools import (
    fetch_pubmed,
    fetch_nih_reporter,
    count_pubmed,
    count_nih_reporter,
    lookup_mesh_terms,
)
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
    "the",
    "a",
    "an",
    "of",
    "in",
    "and",
    "or",
    "for",
    "to",
    "with",
    "by",
    "on",
    "is",
    "are",
    "that",
    "this",
    "from",
    "at",
    "as",
    "its",
    "into",
    "via",
    "role",
    "using",
    "based",
    "study",
    "studies",
    "research",
    "new",
    "analysis",
    "effects",
    "effect",
    "related",
    "associated",
    "between",
}


# Words that describe the query format, not biomedical content.
# Applied in code after both router and refiner to ensure they never end up
# as search terms that get ANDed into API queries.
_QUERY_META_WORDS = frozenset({
    # Single-word meta descriptors
    "literature", "grants", "grant", "funding", "funded",
    "research", "themes", "theme", "trends", "trend", "landscape",
    "study", "studies", "analysis", "review", "overview", "survey",
    "findings", "finding", "published", "emerging", "current", "recent",
    "priorities", "priority", "advances", "advance", "developments",
    "development", "areas", "topics", "approaches",
    # Publication-type words (users say "pubs", "papers", "articles" to mean literature)
    "publications", "publication", "pubs", "pub", "papers", "paper",
    "articles", "article", "journals", "journal",
    # Known multi-word meta phrases
    "funding trends", "grant themes", "grant priorities", "funding priorities",
    "research themes", "research trends", "research landscape",
    "focus areas", "current trends", "emerging themes", "grant funding",
    "emerging trends", "recent advances", "new developments", "key themes",
    "hot topics", "research areas",
})


def _strip_meta_words(terms: list[str]) -> list[str]:
    def _is_meta(term: str) -> bool:
        t = term.lower().strip()
        if t in _QUERY_META_WORDS:
            return True
        # Compound term where EVERY word is individually a meta-word catches
        # unlisted combinations like "current landscape", "recent trends", etc.
        words = t.split()
        return len(words) > 1 and all(w in _QUERY_META_WORDS for w in words)
    return [t for t in terms if not _is_meta(t)]


def _extract_title_terms(title: str) -> list[str]:
    """Extract informative words from a grant/paper title (NIH fallback)."""
    words = title.split()
    out = []
    for w in words:
        w = w.strip(".,;:()[]\"'-")
        if len(w) >= 4 and w.lower() not in _TERM_STOPWORDS and not w.isdigit():
            out.append(w)
    return out


_NICHE_THRESHOLD = 25  # total records (across all years + sources) below this triggers a user check


def _format_mesh_context(mesh_context: dict) -> str:
    """Format MeSH lookup results into a compact block for injection into the refiner prompt."""
    if not mesh_context:
        return "No MeSH data available."
    lines = []
    for term, info in mesh_context.items():
        if not info.get("found"):
            lines.append(
                f'• "{term}": NOT found in MeSH — likely very specific, informal, or a gene name. '
                "Keep the user's term as-is."
            )
            continue
        preferred = info.get("preferred_heading", "")
        is_preferred = info.get("is_preferred_heading", False)
        is_entry = info.get("is_entry_term", False)
        entry_terms = info.get("entry_terms", [])
        tree_nums = info.get("tree_numbers", [])
        scope = info.get("scope_note", "")
        if is_preferred:
            status = f'✓ IS the MeSH preferred heading "{preferred}"'
        elif is_entry:
            status = f'→ Entry term (synonym) under MeSH heading "{preferred}"'
        else:
            status = f'→ Not in MeSH — closest heading is "{preferred}" (different concept)'
        line = f'• "{term}": {status}'
        if tree_nums:
            line += f"  [Tree: {tree_nums[0]}]"
        if entry_terms and not is_preferred:
            line += f"  | Other synonyms: {', '.join(entry_terms[:4])}"
        if scope:
            line += f'\n  Scope: "{scope}"'
        lines.append(line)
    return "\n".join(lines)


def mesh_enrichment_node(state: AgentState) -> dict:
    """Fetch authoritative MeSH descriptors for anchor + explicit terms before query refining.

    Capped at 5 API calls (~0.6–1.2 s total) to keep latency low.
    The refiner uses this data to make informed decisions about term normalization.
    """
    anchor = (state.get("search_terms") or [""])[0]
    explicit_terms = state.get("explicit_terms") or []

    terms_to_lookup: list[str] = []
    if anchor:
        terms_to_lookup.append(anchor)
    for t in explicit_terms:
        if len(terms_to_lookup) >= 5:
            break
        if t and t != anchor:
            terms_to_lookup.append(t)

    if not terms_to_lookup:
        return {"mesh_context": {}}

    logger.info("MeSH enrichment: looking up %d terms: %s", len(terms_to_lookup), terms_to_lookup)
    try:
        mesh_context = lookup_mesh_terms(terms_to_lookup)
    except Exception as exc:
        logger.warning("MeSH enrichment failed: %s", exc)
        mesh_context = {}

    return {"mesh_context": mesh_context}


def corpus_scout_node(state: AgentState) -> dict:
    """Counts available records per year/source before fetching.
    Sets corpus_is_niche=True when totals are very low so the UI can check with the user."""
    # Skip on re-fetch passes or after user has already approved the small corpus
    if state.get("scout_approved") or state.get("fetch_attempts", 0) > 0:
        return {
            "corpus_is_niche": False,
            "scout_results": state.get("scout_results") or {},
            "scout_total": state.get("scout_total") or 0,
        }

    years = state["years"]
    source_filter = state.get("source_filter", "both")
    pubmed_q = state.get("pubmed_query") or ""
    nih_q = state.get("nih_query") or ""

    if not pubmed_q or not nih_q:
        search_terms = state.get("search_terms") or []
        pq, nq = _build_queries(
            search_terms,
            fallback=state.get("enriched_query", ""),
            explicit_terms=state.get("explicit_terms") or [],
            expansion_terms=state.get("expansion_terms") or [],
        )
        pubmed_q = pubmed_q or pq
        nih_q = nih_q or nq

    scout_results: dict = {}
    total = 0

    for year in years:
        year_counts: dict = {}
        if source_filter in ("pubmed", "both", None):
            try:
                n = count_pubmed(pubmed_q, year)
                year_counts["pubmed"] = n
                total += n
            except Exception:
                year_counts["pubmed"] = 0
        if source_filter in ("nih_reporter", "both", None):
            try:
                n = count_nih_reporter(nih_q, year)
                year_counts["nih_reporter"] = n
                total += n
            except Exception:
                year_counts["nih_reporter"] = 0
        scout_results[str(year)] = year_counts

    corpus_is_niche = total < _NICHE_THRESHOLD
    logger.info("Corpus scout: total=%d niche=%s", total, corpus_is_niche)
    return {
        "scout_results": scout_results,
        "scout_total": total,
        "corpus_is_niche": corpus_is_niche,
    }


def query_approval_node(state: AgentState) -> dict:
    """Pure interrupt point. Query strings were already computed by query_refiner_node."""
    return {}


def router_node(state: AgentState) -> dict:
    import datetime as _dt
    _current_year = _dt.date.today().year

    prompt = f"""You are parsing a biomedical research question to query PubMed and NIH Reporter.

    Your tasks:
    1. Enrich the query for vector embedding (no years, just concepts and keywords)
    2. Identify the anchor term (primary disease/topic) and classify all secondary terms:
       - "explicit_terms": biomedical concepts the USER EXPLICITLY MENTIONED in their question.
         These will be ANDed into the search query — results MUST contain these.
         Example: user asks "Parkinson's and biomarkers" → explicit_terms: ["biomarkers"]
       - "expansion_terms": 0-3 additional MeSH-compatible terms YOU add for better coverage.
         These are ORed — helpful but not required.
         Example: you add ["alpha-synuclein", "LRRK2"] to broaden a general PD query.
       - All terms (anchor + explicit + expansion) go into "search_terms" as a flat list,
         anchor FIRST. Each term must be 1-3 words. No meta-terms like "grants" or "research".
    3. Extract structured metadata

    SPECIFICITY SCALE (controls how aggressively fresh data is fetched):
    1 = Broad field — e.g., "oncology", "neuroscience", "immunology"
    2 = Disease class — e.g., "neurodegeneration", "autoimmune disease", "cancer"
    3 = Named disease — e.g., "Parkinson Disease", "Alzheimer Disease", "Type 2 Diabetes"
    4 = Disease + mechanism or biomarker — e.g., "Parkinson Disease biomarkers", "LRRK2 Parkinson"
    5 = Molecular target or variant — e.g., "LRRK2 G2019S inhibitor", "GBA N370S mutation iPSC"
    Use the highest level that matches the user's question. When in doubt, round up.

    SOURCE FILTER:
    - "pubmed": user asks about findings, mechanisms, clinical results, published science
    - "nih_reporter": user asks about funding trends, grants, NIH investments, what is being studied
    - "both": user asks about the full landscape, or both perspectives are needed

    Return ONLY valid JSON with this exact structure:
    {{
        "enriched_query": "...",
        "anchor": "Primary Disease Term",
        "explicit_terms": ["term user asked about"],
        "expansion_terms": ["term you added for coverage"],
        "search_terms": ["anchor", "explicit1", "expansion1"],
        "domain": "...",
        "years": [...],
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

    anchor = parsed.get("anchor", "")
    explicit = _strip_meta_words(parsed.get("explicit_terms") or [])
    expansion = _strip_meta_words(parsed.get("expansion_terms") or [])
    # Rebuild search_terms authoritatively from classified parts so order is guaranteed
    search_terms = (
        [anchor]
        + [t for t in explicit if t != anchor]
        + [t for t in expansion if t != anchor and t not in explicit]
    )

    # Guard: years must never be empty — ChromaDB $in operator rejects empty lists
    years = parsed.get("years") or []
    if not years:
        years = [_current_year - 1, _current_year]

    return {
        "query_vector": query_vector,
        "search_terms": search_terms or parsed.get("search_terms", []),
        "explicit_terms": explicit,
        "expansion_terms": expansion,
        "enriched_query": parsed["enriched_query"],
        "domain": parsed["domain"],
        "years": years,
        "specificity": parsed["specificity"],
        "source_filter": parsed["source_filter"],
    }


def query_refiner_node(state: AgentState) -> dict:
    """
    Option 1: Validates and refines the router's parsed query before any fetching.
    Option 3: Applies source-aware strategy

    One fast Haiku call. No API fetches. Runs between router and library_checker.
    """
    import datetime

    current_year = datetime.date.today().year

    user_query = state["user_query"]
    anchor = (state.get("search_terms") or [""])[0]
    explicit_terms = state.get("explicit_terms") or []
    expansion_terms = state.get("expansion_terms") or []
    years = state.get("years", [])
    source_filter = state.get("source_filter", "both")
    specificity = state.get("specificity", 3)

    mesh_context = state.get("mesh_context") or {}
    mesh_block = _format_mesh_context(mesh_context)

    prompt = f"""You are a biomedical search query expert. Validate and refine this parsed query.

ORIGINAL USER QUESTION: "{user_query}"

CURRENT PARSED QUERY:
- Anchor (primary disease/topic, always required): {anchor}
- Explicit terms (user mentioned these, will be ANDed — results must contain): {explicit_terms}
- Expansion terms (added for coverage, will be ORed — optional breadth): {expansion_terms}
- Years: {years}
- Source: {source_filter}  ("pubmed" | "nih_reporter" | "both")
- Specificity: {specificity}/5
- Current year: {current_year}

MESH GROUNDING DATA (authoritative NCBI descriptors — use this to make better decisions):
{mesh_block}

How to use MeSH data:
- If a term IS the preferred heading → it's already optimal; use it as-is.
- If a term IS an entry term (synonym) under a preferred heading → it's valid and specific;
  keep the user's version. You may mention the preferred heading in refinement_notes.
- If a term is NOT in MeSH → it may be a gene, protein, or informal term; keep it exactly.
- NEVER substitute one concept for another just to match a MeSH heading. If the preferred
  heading is a broader/different concept (e.g. "infralimbic cortex" → MeSH "Prefrontal Cortex"),
  keep the user's specific term — do NOT replace it with the broader heading.

TASK — check each of the following and correct if needed:

1. ANCHOR TERM — Only fix obvious shorthand or possessive forms. NEVER substitute one
   specific scientific concept for another.
   ALLOWED corrections (common name → standard form):
   - "Parkinson's Disease" → "Parkinson Disease"  (possessive → standard)
   - "Alzheimer's" → "Alzheimer Disease"  (abbreviation → full name)
   - "COVID" → "COVID-19"  (abbreviation → full name)
   NOT allowed: replacing one specific anatomical region, gene, pathway, or biological
   concept with a different one, even if the replacement is "closer" to MeSH.
   Examples of what NOT to do:
   - "infralimbic" → do NOT change to "prelimbic" (different brain region)
   - "ventral striatum" → do NOT change to "nucleus accumbens" (related but distinct)
   - "LRRK2 G2019S" → do NOT simplify to "LRRK2" (user specified the variant)
   If the user's term is specific and scientifically valid, keep it exactly.
   If a closely related MeSH heading exists but differs from the user's term, note it
   in refinement_notes but DO NOT change the term.

2. EXPLICIT TERMS: Are these truly concepts the user explicitly asked about?
   If the user didn't name it directly, move it to expansion_terms instead.
   Apply the same rule as #1 — never substitute one specific term for another.
   Keep each term 1-3 words.

3. EXPANSION TERMS: Are they useful and distinct from explicit? Replace vague or
   redundant ones. 0-3 terms max. Each must be 1-3 words.
   Do not add expansion terms that are close synonyms of the anchor — that narrows
   rather than broadens. Add terms that cover adjacent but distinct aspects.

4. SOURCE FILTER:
   - Question is about published findings, mechanisms, biology, clinical results
     → prefer "pubmed" or "both"
   - Question is about funding trends, grants, what NIH is investing in, emerging
     priorities → prefer "nih_reporter" or "both"
   - Question explicitly mentions both perspectives → "both"
   Adjust if the current filter doesn't match the intent.

5. YEARS: Do the years match the question's time intent?
   - "recent" / "emerging" / "new" → [{current_year - 1}, {current_year}]
   - "trends" / "landscape" / no qualifier → [{current_year - 2}, {current_year - 1}]
   - "established" / "historical" → broader range is fine
   Adjust only if clearly wrong.

Respond in valid JSON only:
{{
  "anchor": "corrected anchor term",
  "explicit_terms": ["..."],
  "expansion_terms": ["..."],
  "years": [...],
  "source_filter": "pubmed|nih_reporter|both",
  "refinement_notes": "1-2 sentences. If you kept a user term that differs from a MeSH heading, say so and name the MeSH heading for reference. If nothing changed, say 'Query looks well-formed — no changes needed.'",
  "clarifying_question": "One focused question to ask the user when the query is ambiguous OR specificity >= 4. Goal: confirm the user's exact intent before fetching. Examples: 'Are you focused on the infralimbic cortex specifically, or the broader medial prefrontal cortex?', 'Do you want recent clinical trial results, animal model studies, or both?' Leave empty string if the query intent is already clear."
}}"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content.strip())

        new_anchor = parsed.get("anchor") or anchor
        new_explicit = _strip_meta_words(parsed.get("explicit_terms") or [])
        new_expansion = _strip_meta_words(parsed.get("expansion_terms") or [])

        # Rebuild search_terms authoritatively from refined parts
        seen = {new_anchor}
        new_search_terms = [new_anchor]
        for t in new_explicit + new_expansion:
            if t and t not in seen:
                seen.add(t)
                new_search_terms.append(t)

        notes = parsed.get("refinement_notes", "")
        clarifying_question = parsed.get("clarifying_question", "")
        new_years = parsed.get("years") or years or [current_year - 1, current_year]
        new_source = parsed.get("source_filter") or source_filter
        logger.info("Query refiner: %s", notes)

        # Compute API query strings now so they're visible at the approval interrupt.
        # query_approval_node runs AFTER the interrupt, so the refiner is the last
        # chance to put these into state before the user sees the approval panel.
        try:
            pubmed_q, nih_q = _build_queries(
                new_search_terms,
                fallback=state.get("enriched_query", ""),
                explicit_terms=new_explicit,
                expansion_terms=new_expansion,
            )
        except Exception:
            pubmed_q, nih_q = "", ""

        return {
            "search_terms": new_search_terms,
            "explicit_terms": new_explicit,
            "expansion_terms": new_expansion,
            "years": new_years,
            "source_filter": new_source,
            "refinement_notes": notes,
            "clarifying_question": clarifying_question,
            "pubmed_query": pubmed_q,
            "nih_query": nih_q,
        }

    except Exception as e:
        logger.error("Query refiner failed: %s", e)
        return {"refinement_notes": "", "clarifying_question": "", "pubmed_query": "", "nih_query": ""}


def library_checker_node(state: AgentState) -> dict:
    years = state["years"]
    query_vector = state.get("query_vector")
    specificity = state.get("specificity", 3)
    fetch_attempts = state.get("fetch_attempts", 0)

    test_limit = int(os.environ.get("BIOINSIGHT_TEST_LIMIT", 0))
    if test_limit > 0:
        required_docs = test_limit
    else:
        fetch_mode = state.get("fetch_mode", "standard")
        mode_limit = _FETCH_MODE_LIMITS.get(fetch_mode)
        required_docs = 500 if mode_limit is None else max(30, int(mode_limit * 0.6))

    # High-specificity queries always need a fresh fetch on the first run
    if specificity >= 4 and fetch_attempts == 0:
        return {"library_has_data": False}

    # Post-fetch: only require ≥1 doc per year — we already fetched everything available,
    # so if the threshold isn't met the topic is simply very niche. Proceed to the
    # material_assessor which will honestly report the limited coverage.
    if fetch_attempts > 0:
        check_docs = 1
        check_similarity = 0.0
    else:
        check_docs = required_docs
        check_similarity = 0.2 + (specificity * 0.08)

    library_has_data = all(
        chroma.semantic_search_by_year(
            query_vector, year, min_records=check_docs, min_similarity=check_similarity
        )
        for year in years
    )

    return {"library_has_data": library_has_data}


def _build_queries(
    search_terms: list,
    fallback: str,
    explicit_terms: list = None,
    expansion_terms: list = None,
) -> tuple[str, str]:
    """
    Build one (pubmed_query, nih_query) pair.

    RETRIEVAL PHILOSOPHY:
    - Anchor (search_terms[0]) is always required.
    - Explicit terms (user asked for) are ANDed — results must contain these.
    - Expansion terms are intentionally EXCLUDED from the retrieval query.
      They over-restrict lexical search, filtering out relevant papers that use
      synonyms. The semantic embedding layer (BioBERT + UMAP clustering) handles
      thematic focus without requiring keyword matches.

    When no explicit terms are set, all secondaries are ORed for breadth,
    so the anchor alone isn't too narrow.

    PubMed: no field tag, no quotes — ATM handles MeSH mapping.
    NIH:    "advanced" Lucene operator; multi-word anchor joined with AND to prevent OR default.
    """
    explicit_terms = explicit_terms or []

    if not search_terms and not fallback:
        return "", ""

    anchor = search_terms[0] if search_terms else fallback

    def _pub(t):
        return f'"{t}"[tiab]' if " " in t else f"{t}[tiab]"

    def _nih(t):
        return f'"{t}"' if " " in t else t

    # PubMed anchor: no quotes, no field tag. ATM maps words to MeSH headings and
    # searches all fields — same behavior as the web UI.
    parts_pub = [anchor]

    # NIH anchor: "advanced" operator uses Lucene parsing where space-separated
    # words default to OR. Joining with explicit AND requires all anchor words
    # to appear somewhere in the grant text, without demanding an exact phrase.
    nih_anchor = " AND ".join(anchor.split()) if " " in anchor else anchor
    parts_nih = [nih_anchor]

    if explicit_terms:
        # Explicit terms → ANDed with [tiab] to keep them as precision filters.
        # The user specifically named these so results should contain them in-text.
        for t in explicit_terms:
            if t and t != anchor:
                parts_pub.append(_pub(t))
                parts_nih.append(_nih(t))

    elif len(search_terms) > 1:
        # No explicit terms — OR expansion terms for PubMed breadth.
        # NIH gets anchor only: complex boolean over-restricts NIH's full-text index,
        # and BioBERT handles topical focus without keyword matching.
        secondary = [t for t in search_terms[1:] if t != anchor]
        if secondary:
            # Expansion terms also use no field tag for maximum breadth
            def _pub_broad(t):
                return f'"{t}"' if " " in t else t
            parts_pub.append(f"({' OR '.join(_pub_broad(t) for t in secondary)})")

    return " AND ".join(parts_pub), " AND ".join(parts_nih)


# Per-mode fetch limits (per year, per source).
# "everything" uses None — the actual count drives the limit, capped by EVERYTHING_CAP.
_FETCH_MODE_LIMITS = {
    "quick": 150,
    "standard": 400,
    "deep": 800,
    "everything": None,
}
_EVERYTHING_CAP = 3000  # hard ceiling per year per source for "everything" mode


def fetcher_node(state: AgentState) -> dict:
    search_terms = state.get("search_terms", [])
    years = state["years"]
    source_filter = state.get("source_filter")
    fetch_mode = state.get("fetch_mode", "standard")

    # TEST_LIMIT env var overrides mode (useful for dev/CI runs)
    test_limit = int(os.environ.get("BIOINSIGHT_TEST_LIMIT", 0))
    mode_limit = (
        test_limit if test_limit > 0 else _FETCH_MODE_LIMITS.get(fetch_mode, 400)
    )

    # Use user-edited query strings if present, otherwise build from structured terms
    if state.get("pubmed_query") or state.get("nih_query"):
        pubmed_q = state.get("pubmed_query") or ""
        nih_q = state.get("nih_query") or ""
        logger.info("Fetcher | mode=%s | using user-edited queries", fetch_mode)
    else:
        pubmed_q, nih_q = _build_queries(
            search_terms,
            fallback=state.get("enriched_query", ""),
            explicit_terms=state.get("explicit_terms") or [],
            expansion_terms=state.get("expansion_terms") or [],
        )
    logger.info("Fetcher | mode=%s | PubMed: '%s'", fetch_mode, pubmed_q)
    logger.info("Fetcher | mode=%s | NIH:    '%s'", fetch_mode, nih_q)

    total_records = 0
    corpus_stats: dict = {}

    for year in years:
        year_stats: dict = {}
        all_records = []

        # ── PubMed ────────────────────────────────────────────────────────
        if source_filter in ("pubmed", "both", None):
            pub_available = count_pubmed(pubmed_q, year)
            year_stats["pub_available"] = pub_available

            if pub_available > 0:
                if mode_limit is None:
                    pub_limit = min(pub_available, _EVERYTHING_CAP)
                else:
                    pub_limit = min(mode_limit, pub_available)

                try:
                    records = fetch_pubmed.invoke(
                        {"domain": pubmed_q, "year": year, "max_results": pub_limit}
                    )
                    all_records += records
                    year_stats["pub_fetched"] = pub_limit
                    logger.info(
                        "PubMed year=%d available=%d fetched=%d → %d passages",
                        year,
                        pub_available,
                        pub_limit,
                        len(records),
                    )
                except Exception as e:
                    logger.warning("PubMed fetch failed year=%d: %s", year, e)
                    year_stats["pub_fetched"] = 0
            else:
                year_stats["pub_fetched"] = 0

        # ── NIH Reporter ─────────────────────────────────────────────────
        if source_filter in ("nih_reporter", "both", None):
            nih_available = count_nih_reporter(nih_q, year)
            year_stats["nih_available"] = nih_available

            if nih_available > 0:
                if mode_limit is None:
                    nih_limit = min(nih_available, _EVERYTHING_CAP)
                else:
                    nih_limit = min(mode_limit, nih_available)

                nih_base = {
                    "domain": nih_q,
                    "fiscal_year": year,
                    "max_results": nih_limit,
                }

                try:
                    # Stratified sampling: if we're fetching < 60% of what's available,
                    # split the budget between offset=0 (most relevant) and offset=middle
                    # (mid-relevance) to get better subtopic diversity.
                    if nih_available > nih_limit * 1.7 and nih_limit >= 40:
                        half = nih_limit // 2
                        records_top = fetch_nih_reporter.invoke(
                            {**nih_base, "max_results": half}
                        )
                        records_mid = fetch_nih_reporter.invoke(
                            {
                                **nih_base,
                                "max_results": half,
                                "offset_start": nih_available // 2,
                            }
                        )
                        records = records_top + records_mid
                        logger.info(
                            "NIH year=%d stratified: top=%d mid=%d → %d passages",
                            year,
                            len(records_top),
                            len(records_mid),
                            len(records),
                        )
                    else:
                        records = fetch_nih_reporter.invoke(nih_base)
                        logger.info(
                            "NIH year=%d available=%d fetched=%d → %d passages",
                            year,
                            nih_available,
                            nih_limit,
                            len(records),
                        )
                    all_records += records
                    year_stats["nih_fetched"] = nih_limit
                except Exception as e:
                    logger.warning("NIH fetch failed year=%d: %s", year, e)
                    year_stats["nih_fetched"] = 0
            else:
                year_stats["nih_fetched"] = 0

        if all_records:
            embedded = embedder.embed_records(all_records)
            chroma.upsert_records(embedded)
            total_records += len(all_records)

        corpus_stats[str(year)] = year_stats

    return {
        "records_fetched": total_records,
        "fetch_attempts": state.get("fetch_attempts", 0) + 1,
        "corpus_stats": corpus_stats,
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
        doc_blocks.append(f"({doc['pi_name']}, {doc['year']}) — {doc['title']}\n{body}")
    sample_text = "\n\n---\n\n".join(doc_blocks)

    user_query = state.get("user_query", "")

    prompt = f"""You are a biomedical research intelligence analyst assessing a dataset before deep analysis.

ORIGINAL USER QUESTION: "{user_query}"
SEARCH SCOPE: {enriched_query}
Years: {years}
Source: {source_filter}

Your job is to assess whether the data below is sufficient to answer the user's specific question — not just whether it covers the topic generally.

Library scope stats:
- Total passages matching year/source filter: {total_passages_in_scope}
- Unique source documents retrieved by semantic search: {total_unique_docs}
- Documents shown below (top 30 by relevance): {min(30, total_unique_docs)}

Top 30 most relevant documents:
{sample_text}

Assess this dataset in the context of the user's question and respond in valid JSON only:
{{
  "assessment": "2-3 sentences: does this data actually answer '{user_query}'? What aspects of the question are well-covered? What is missing that would be needed to answer it fully?",
  "coverage_score": 1-5,
  "action": "proceed" or "fetch_more",
  "reasoning": "one sentence explaining the action recommendation relative to the user's question",
  "suggested_terms": ["ShortMeSHTerm1", "ShortMeSHTerm2"]
}}

Action rules:
- "proceed" if the data covers the user's specific question with reasonable depth.
- "fetch_more" ONLY if there are clear, named gaps in the data that different search terms would fill. Be conservative — default to proceed.

Suggested term rules (CRITICAL):
- Terms must address what is MISSING for the user's question — not repeat what was already searched.
- Each term must be 1-3 words maximum — a real MeSH term or standard biomedical keyword.
- Good: "neuroinflammation", "LRRK2", "GBA mutation", "alpha-synuclein"
- Bad: "neuroinflammation cytokines IL-6 TNF-alpha", "genetic risk factors GBA PINK1 DJ-1"
- Never suggest terms already in the search scope: {state.get("search_terms", [])} or any terms like grant,funding trends, emerging topics that aren't real biomedical concepts."""

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

    user_query = state.get("user_query", "")

    prompt = f"""You are a biomedical research analyst. Write a SHORT preliminary overview (200-250 words) for a program officer deciding whether to run a full analysis.

ORIGINAL USER QUESTION: "{user_query}"
SEARCH SCOPE: {enriched_query}
Years: {years}
Source: {source_filter}
Total unique source documents in scope: {total_unique_docs} (showing top 25 by relevance below)

Top 25 most relevant documents:
{sample_text}

Write in crisp declarative prose with these three parts, always framed around the user's question:
1. **What's here** — which aspects of "{user_query}" are visible in these {total_unique_docs} documents? Name actual topics, proteins, mechanisms, or PI names you can see.
2. **Depth** — is the coverage specific enough to answer the user's question, or is it broad and tangential?
3. **Potential gaps** — what would a researcher need to fully answer "{user_query}" that appears absent here?

Be direct and specific. Do not pad. Do not repeat the user's question verbatim — synthesize what the data shows about it."""

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

        finding_prompt = f"""You are analyzing biomedical research documents to extract high-value findings.
Original user question: {enriched_query}
Source: {source_filter}

Documents (each block is one grant/paper; the ID in brackets is the citation anchor):
{passages_text}

Extract 0-3 findings that are explicitly stated in the text AND directly relevant to the original question.

QUALITY RULES — prefer findings in this order:
1. Quantitative results: specific numbers, effect sizes, percentages, p-values, fold-changes (e.g., "LRRK2 inhibition reduced phospho-S129 α-synuclein by 60% in patient iPSC-neurons")
2. Named mechanism or pathway: a specific molecular interaction, causal relationship, or pathway finding (e.g., "GBA loss-of-function activates NLRP3 inflammasome via lysosomal dysfunction")
3. Named clinical or translational outcome: trial result, biomarker validation, patient subgroup finding
4. General thematic finding: only if nothing more specific exists

EXCLUSION RULES — do NOT extract:
- Research aims, hypotheses, or future plans ("will test", "aims to", "we hypothesize", "we expect", "proposed")
- Background statements or textbook-level facts ("dopamine neurons are lost in Parkinson's")
- Vague generalizations without named entities or measurements

If the documents are off-topic or contain only aims/background, return 0 findings — do not pad with weak claims.

Use the ID in brackets as the evidence_id. Construct citation_text as [Last Name et al., Year].

Return ONLY valid JSON, no other text:
{{"findings": [
    {{
        "claim": "specific, cited factual finding with named entities or measurements",
        "evidence_ids": ["nih_reporter__12345__s0"],
        "citation_text": "[Smith et al., 2024]"
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
    user_query = state.get("user_query", "")
    enriched_query = state.get("enriched_query", "")

    def verify_cluster(label: int, passages: list) -> tuple[int, list]:
        claims_to_check = extracted_findings.get(label, [])
        if not claims_to_check:
            return label, []

        docs = _group_passages_by_parent(passages, max_docs=6)
        doc_blocks = []
        for doc in docs:
            text_body = " ".join(sent for _, sent in doc["sentences"])
            anchor_id = doc["sentences"][0][0]
            doc_blocks.append(
                f"[{anchor_id}] ({doc['pi_name']}, {doc['year']}): {text_body}"
            )
        source_text = "\n\n---\n\n".join(doc_blocks)

        claims_json_str = json.dumps(claims_to_check, indent=2)

        verifier_prompt = f"""You are a strict, objective fact-checker.
Verify each CLAIM against the SOURCE DOCUMENTS below using two independent criteria.

ORIGINAL USER QUESTION: "{user_query}"
SEARCH SCOPE: {enriched_query}

SOURCE DOCUMENTS (each block is one full grant abstract or paper):
{source_text}

CLAIMS TO VERIFY:
{claims_json_str}

For each claim evaluate TWO things independently:

1. FACTUAL SUPPORT — is this claim explicitly stated as an established fact or confirmed result in the source documents?
   - TRUE only if the document text explicitly asserts it as fact/result.
   - FALSE if it is a research aim ("will test", "aims to", "we hypothesize", "we expect", "proposed").
   - FALSE if it is background/textbook knowledge not evidenced by the specific documents shown.

2. QUERY RELEVANCE — does this claim directly address the user's original question?
   - TRUE if the claim provides a specific answer, finding, or insight relevant to what the user asked.
   - FALSE if the claim is factually supported but tangential (e.g., describes methods, animal models, or a different disease/target than asked about).

A claim passes if BOTH criteria are true.

For claims that FAIL, classify the failure in "fail_reason":
- "aim": phrased as a research aim, hypothesis, or proposed direction — not yet established
- "overspecified": the core finding IS in the source but the claim added a clause or detail that is NOT — the source supports part of the claim but not all of it
- "background": general background knowledge not specifically evidenced by these documents
- "irrelevant": factually supported but does not address the user's question

For claims where fail_reason="aim" AND is_relevant=true, provide a repaired version in "repaired_claim":
- Rephrase as an active investigation, preserving ALL named entities, mechanisms, and measurements
- Use phrases like: "Researchers are investigating whether...", "X is under active investigation as...", "Active work is examining the role of..."
- Do NOT generalize or strip specifics — keep protein names, pathways, and drug names intact

For claims where fail_reason="overspecified" AND is_relevant=true, provide a repaired version in "repaired_claim":
- Restate ONLY what the source document explicitly asserts — remove the unsupported clause(s)
- Preserve all named entities, measurements, and mechanistic details that ARE in the source
- Do NOT hedge or soften — if the source states it as a fact, state it as a fact
- Example: claim says "A caused B, supporting C" but source only says "A caused B" → repair is "A caused B"

For all other failures (background, irrelevant, or aim/overspecified where is_relevant=false), set repaired_claim to null.

Return ONLY a valid JSON array, same length as CLAIMS TO VERIFY, in the same order:
[
  {{
    "is_supported": true,
    "is_relevant": true,
    "fail_reason": null,
    "repaired_claim": null,
    "reasoning": "Document explicitly states X; directly addresses the user's question about Y."
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
                supported = eval_result.get("is_supported") is True
                relevant = eval_result.get("is_relevant") is True
                repaired = eval_result.get("repaired_claim")
                fail_reason = eval_result.get("fail_reason")
                reasoning = eval_result.get("reasoning", "")

                if supported and relevant:
                    approved_claims.append(claims_to_check[i])
                elif repaired and relevant:
                    if fail_reason == "overspecified":
                        # Core finding is real — strip the unsupported clause,
                        # keep as a confirmed finding (no confidence flag)
                        repaired_obj = {**claims_to_check[i], "claim": repaired}
                        logger.info(
                            "Corrected (cluster %d) overspecified → confirmed: %s",
                            label,
                            repaired[:80],
                        )
                    else:
                        # Aim-phrased claim repaired as an active investigation
                        repaired_obj = {
                            **claims_to_check[i],
                            "claim": repaired,
                            "confidence": "preliminary",
                        }
                        logger.info(
                            "Repaired (cluster %d) %s → preliminary: %s",
                            label,
                            fail_reason,
                            repaired[:80],
                        )
                    approved_claims.append(repaired_obj)
                else:
                    logger.warning(
                        "Dropped (cluster %d) supported=%s relevant=%s fail_reason=%s: %s | %s",
                        label,
                        supported,
                        relevant,
                        fail_reason,
                        claims_to_check[i].get("claim"),
                        reasoning,
                    )
            return label, approved_claims

        except Exception as e:
            logger.error(f"Failed to verify cluster {label}: {e}")
            return label, []

    verified_findings = {}
    work = [
        (label, passages)
        for label, passages in clusters.items()
        if label != -1 and label in extracted_findings
    ]
    logger.info(f"Verification: verifying {len(work)} clusters sequentially")

    for label, passages in work:
        label, approved = verify_cluster(label, passages)
        verified_findings[label] = approved

    return {"verified_cluster_findings": verified_findings}


def _add_hyperlink(paragraph, text: str, url: str):
    """Insert a clickable hyperlink into a docx paragraph."""
    part = paragraph.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
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
            paragraph.add_run(text[cursor : m.start()])
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
    user_query = state.get("user_query", "")
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

    # Tag each finding with its data source and separate confirmed vs preliminary
    tagged_findings = {}
    preliminary_findings = []
    pubmed_count = 0
    nih_count = 0
    for label, findings in verified_findings.items():
        tagged = []
        for f in findings:
            evidence_ids = f.get("evidence_ids", [])
            src = "unknown"
            if evidence_ids:
                src = evidence_index.get(evidence_ids[0], {}).get("source", "unknown")
            is_preliminary = f.get("confidence") == "preliminary"
            if not is_preliminary:
                if src == "pubmed":
                    pubmed_count += 1
                elif src == "nih_reporter":
                    nih_count += 1
            tagged_f = {**f, "data_source": src}
            if is_preliminary:
                preliminary_findings.append(tagged_f)
            else:
                tagged.append(tagged_f)
        tagged_findings[label] = tagged

    all_findings_text = json.dumps(tagged_findings, indent=2)
    preliminary_count = len(preliminary_findings)
    preliminary_text = json.dumps(preliminary_findings, indent=2) if preliminary_findings else ""

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
        source_context = (
            "All findings come from NIH-funded grant abstracts (NIH Reporter)."
        )

    preliminary_section_instruction = ""
    if preliminary_findings:
        preliminary_section_instruction = f"""
## Research Directions
These are active investigations — aims and hypotheses that have not yet produced confirmed findings.
Use hedged language only: "is under active investigation", "researchers are examining", "preliminary work suggests".
Never present these as established facts.
Group by theme. Each entry must include its citation.

PRELIMINARY CLAIMS ({preliminary_count} items):
{preliminary_text}
"""

    synthesis_prompt = f"""You are a senior biomedical research analyst writing an intelligence report for a program officer who needs to make funding decisions.

ORIGINAL USER QUESTION: "{user_query}"
SEARCH SCOPE: {enriched_query} | Years: {years}
{source_context}

The report must directly answer the user's original question. Every section should be framed around what the user asked — not just what the data happens to contain.

VERIFIED FINDINGS ({len(verified_findings)} clusters, {pubmed_count + nih_count} confirmed findings, {preliminary_count} preliminary):
Each finding includes a "data_source" field: "pubmed" = published literature, "nih_reporter" = active/recent grant funding.
Claims with "confidence": "preliminary" MUST appear ONLY in the Research Directions section — never in confirmed finding sections.
{all_findings_text}

---

Write a structured report in EXACTLY this order. Use markdown headings.

# BioInsight Report: {enriched_query}

## Executive Summary
3-5 bullet points that directly answer: **"{user_query}"**
Each bullet must name a specific finding (target, mechanism, drug, trend, or gap) and include a citation. Lead with the most actionable insight.

## Key Entities at a Glance
A markdown table of the most important named biological entities (proteins, genes, drugs, pathways, biomarkers, cell types) that appear in the verified findings. Only include entities you can directly cite.

| Entity | Type | Role in {enriched_query.split()[0] if enriched_query else 'this area'} | Key Finding | Citation |
|--------|------|---------|-------------|----------|
(fill rows — leave no uncited rows)

{f'''## What Published Research Shows
Start with 1-2 sentences that orient the reader: what is the overall picture emerging from the published literature as it relates to "{user_query}"?

Then group findings by theme. For EACH theme:
- Open with 1-2 sentences introducing what this theme is and why it matters in the context of the user's question.
- Follow with the specific findings and citations.

Aim for 3-5 themes. After every claim, cite using the citation_text exactly.

## Where Grant Funding is Going
Start with 1-2 sentences: how does the NIH grant landscape for this topic relate to what the user asked — is investment aligned with published findings, or diverging toward emerging priorities?

Then group findings by theme. For EACH theme:
- Open with 1-2 sentences introducing what this funding area is and what it signals strategically.
- Follow with the specific findings and citations.

Aim for 3-5 themes. After every claim, cite using the citation_text exactly.''' if has_pubmed and has_nih else f'''## Research Findings by Theme
Start with 1-2 sentences framing the overall picture as it relates to "{user_query}".

For EACH theme:
- Open with 1-2 sentences introducing what this theme covers and its relevance to the user's question.
- Follow with specific findings and citations.

Aim for 3-5 themes.'''}

{preliminary_section_instruction if preliminary_findings else ""}
## Research Gaps & White Space
What specific aspects of "{user_query}" are absent or under-represented in the verified findings? Frame gaps in terms of the user's original question — what did they ask about that the data couldn't answer? Be concrete: name missing targets, populations, mechanisms, or time horizons.

## Strategic Summary
2-3 sentences answering: given this data, what is the single most important insight for someone asking "{user_query}", and what would a well-targeted next action or investment look like?

---

CRITICAL RULES:
- Use citation_text EXACTLY as it appears in the JSON (e.g., [Smith et al., 2024]). Never paraphrase or reformat it.
- Every factual claim must have a citation. No unsupported statements.
- Do NOT use raw system IDs (nih_reporter__, pubmed__, etc.) anywhere in the output.
- Frame the entire report in terms of the original user question — not just the enriched query.
- Claims with "confidence": "preliminary" belong ONLY in Research Directions — never assert them as established results elsewhere."""

    response = synthesis_llm.invoke(synthesis_prompt)

    # Post-process: [Author, Year] → [Author, Year](url) for Markdown links
    report_text = response.content
    for citation, url in citation_url_map.items():
        report_text = report_text.replace(citation, f"{citation}({url})")

    diagnostic = f"""---
**Analysis Metadata** — Source: {source_filter} | Years: {years} | Unique docs: {unique_docs} | Passages: {total_passages} | Clusters: {len(verified_findings)} | Confirmed findings: {pubmed_count + nih_count} | Research directions: {preliminary_count}

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
