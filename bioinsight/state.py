from typing import Optional, TypedDict, Annotated


class AgentState(TypedDict):
    # User input
    user_query: str
    enriched_query: Optional[str]
    query_vector: Optional[list[float]]
    search_terms: Annotated[Optional[list], lambda x, y: y if y is not None else x]

    # Router outputs
    domain: Optional[str]
    years: Optional[list[int]]
    specificity: Optional[int]
    source_filter: Optional[str]  # "pubmed", "nih_reporter", or "both"
    # Terms explicitly in the user's question (ANDed) vs added by router for coverage (ORed)
    explicit_terms: Optional[list[str]]
    expansion_terms: Optional[list[str]]
    # Query refiner outputs
    activity_codes: Optional[
        list[str]
    ]  # NIH grant mechanism filter, e.g. ["R01", "R21"]
    refinement_notes: Optional[str]  # what the refiner changed and why
    clarifying_question: Optional[str]  # follow-up question for user when query is ambiguous

    # Query approval outputs (computed at interrupt, user-editable before fetch)
    pubmed_query: Optional[str]   # final PubMed query string sent to API
    nih_query: Optional[str]      # final NIH Reporter query string sent to API

    # Corpus scout outputs (pre-fetch count check)
    scout_results: Optional[dict]   # {year: {"pubmed": N, "nih_reporter": M}}
    scout_total: Optional[int]      # total records across all years + sources
    corpus_is_niche: Optional[bool] # True when total < threshold
    scout_approved: Optional[bool]  # user explicitly OKed a small-corpus fetch

    # Library Checker outputs
    library_has_data: Optional[bool]

    # Fetcher inputs
    fetch_mode: Optional[str]  # "quick" | "standard" | "deep" | "everything"

    # Library Checker / Fetcher coordination
    years_needing_data: Optional[
        list
    ]  # years the library_checker says still need fetching

    # Fetcher outputs
    records_fetched: Optional[int]
    fetch_attempts: Optional[int]
    corpus_stats: Optional[
        dict
    ]  # {year: {pub_available, nih_available, pub_fetched, nih_fetched}}

    # Material Assessor outputs
    material_assessment: Optional[str]  # LLM prose assessment of data quality
    assessment_action: Optional[str]  # "proceed" | "fetch_more"
    suggested_terms: Optional[list[str]]  # extra terms the LLM wants fetched
    coverage_breakdown: Optional[
        dict
    ]  # structured signals: term freqs, year dist, source split

    # Preliminary Report output (shown to human before full analysis)
    prelim_report: Optional[str]

    # Subset Modeler outputs
    clusters: Optional[dict]

    # Extraction Node outputs (Pass 1)
    cluster_findings: Optional[dict]
    evidence_index: Optional[dict]

    # Verifier Node outputs (Pass 2)
    verified_cluster_findings: Optional[dict]

    # Report Writer outputs (Pass 3)
    report_sidecar: Optional[str]
    report_docx: Optional[bytes]
    final_answer: Optional[str]

    # MeSH enrichment (fetched before query_refiner runs)
    mesh_context: Optional[dict]  # {term: {found, preferred_heading, entry_terms, ...}}

    # Error tracking
    error: Optional[str]
