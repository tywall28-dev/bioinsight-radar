from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from bioinsight.state import AgentState
from langgraph.checkpoint.memory import MemorySaver
from bioinsight.nodes import (
    fetcher_node,
    router_node,
    library_checker_node,
    subset_modeler_node,
    synthesis_node,
    error_node,
)

graph = StateGraph(AgentState)
graph.add_node("router", router_node)
graph.add_node("library_checker", library_checker_node)
graph.add_node("fetcher", fetcher_node)
graph.add_node("subset_modeler", subset_modeler_node)
graph.add_node("synthesis", synthesis_node)
graph.add_node("error", error_node)


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
        "error": "error",
    },
)
graph.add_edge("fetcher", "library_checker")
graph.add_edge("subset_modeler", "synthesis")
graph.add_edge("synthesis", END)
graph.add_edge("error", END)

graph.set_entry_point("router")

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer, interrupt_before=["subset_modeler"])
