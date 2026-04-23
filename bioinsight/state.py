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
    entity: Optional[str]
    specificity: Optional[int]
    source_filter: Optional[str]  # "pubmed", "nih_reporter", or "both"

    # Library Checker outputs
    library_has_data: Optional[bool]

    # Fetcher outputs
    records_fetched: Optional[int]
    fetch_attempts: Optional[int]

    # Material Assessor outputs
    material_assessment: Optional[str]   # LLM prose assessment of data quality
    assessment_action: Optional[str]     # "proceed" | "fetch_more"
    suggested_terms: Optional[list[str]] # extra terms the LLM wants fetched
    coverage_breakdown: Optional[dict]   # structured signals: term freqs, year dist, source split

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

    # Error tracking
    error: Optional[str]
