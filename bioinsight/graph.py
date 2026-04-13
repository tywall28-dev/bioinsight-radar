from langgraph.graph import StateGraph, END
from bioinsight.state import AgentState
from bioinsight.nodes import fetcher_node, router_node, library_checker_node

graph = StateGraph(AgentState)
graph.add_node("router", router_node)
graph.add_node("library_checker", library_checker_node)
graph.add_node("fetcher", fetcher_node)


def should_fetch(state: AgentState) -> str:
    if state["library_has_data"]:
        return "subset_modeler"
    elif state.get("fetch_attempts", 0) >= 3:
        return "error"
    else:
        return "fetcher"


graph.add_edge("router", "library_checker")
graph.add_conditional_edges(
    "library_checker",
    should_fetch,
    {
        "fetcher": "fetcher",
        "subset_modeler": "subset_modeler",
        "error": END,
    },
)

graph.set_entry_point("router")
app = graph.compile()
