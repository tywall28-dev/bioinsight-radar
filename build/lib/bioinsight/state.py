from typing import TypedDict, Optional

class AgentState(TypedDict):
    # User input
    user_query: str
    
    # Router outputs
    domain: Optional[str]
    years: Optional[list[int]]
    query_type: Optional[str]  # "macro" or "micro"
    
    # Library Checker outputs
    library_has_data: Optional[bool]
    
    # Fetcher outputs
    records_fetched: Optional[int]
    
    # Subset Modeler outputs
    clusters: Optional[list]
    
    # Synthesis outputs
    final_answer: Optional[str]

    # Error tracking
    error: Optional[str]