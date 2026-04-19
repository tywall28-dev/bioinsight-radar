from typing import Optional, TypedDict, Annotated


class AgentState(TypedDict):
    # User input
    user_query: str
    enriched_query: Optional[str]  # Enriched query after processing by the router
    query_vector: Optional[list[float]]  # Vector representation of the user query
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

    # Subset Modeler outputs
    clusters: Optional[list]

    # Synthesis outputs
    report_sidecar: Optional[str]
    final_answer: Optional[str]

    # Error tracking
    error: Optional[str]

    # Fetch Attempts
    fetch_attempts: Optional[int]
