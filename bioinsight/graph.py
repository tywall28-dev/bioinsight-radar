from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from bioinsight.state import AgentState
from langgraph.checkpoint.memory import MemorySaver
from bioinsight.nodes import (
    fetcher_node,
    router_node,
    library_checker_node,
    material_assessor_node,
    prelim_report_node,
    subset_modeler_node,
    extraction_node,
    verifier_node,
    report_writer_node,
    error_node,
)

graph = StateGraph(AgentState)
graph.add_node("router", router_node)
graph.add_node("library_checker", library_checker_node)
graph.add_node("fetcher", fetcher_node)
graph.add_node("material_assessor", material_assessor_node)
graph.add_node("prelim_report", prelim_report_node)
graph.add_node("subset_modeler", subset_modeler_node)
graph.add_node("extraction", extraction_node)
graph.add_node("verifier", verifier_node)
graph.add_node("report_writer", report_writer_node)
graph.add_node("error", error_node)


def should_fetch(state: AgentState) -> str:
    if state["library_has_data"]:
        return "material_assessor"
    elif state.get("fetch_attempts", 0) >= 3:
        return "error"
    else:
        return "fetcher"


def should_proceed(state: AgentState) -> str:
    """
    After the material assessor runs, decide whether to fetch more data
    or move on to the preliminary report.
    The assessor can trigger at most one extra fetch; after that we always proceed.
    """
    action = state.get("assessment_action", "proceed")
    fetch_attempts = state.get("fetch_attempts", 0)
    if action == "fetch_more" and fetch_attempts < 4:
        return "fetcher"
    return "prelim_report"


graph.add_edge("router", "library_checker")
graph.add_conditional_edges(
    "library_checker",
    should_fetch,
    {
        "fetcher": "fetcher",
        "material_assessor": "material_assessor",
        "error": "error",
    },
)
graph.add_edge("fetcher", "library_checker")
graph.add_conditional_edges(
    "material_assessor",
    should_proceed,
    {
        "fetcher": "fetcher",
        "prelim_report": "prelim_report",
    },
)
graph.add_edge("prelim_report", "subset_modeler")

# Linear synthesis pipeline
graph.add_edge("subset_modeler", "extraction")
graph.add_edge("extraction", "verifier")
graph.add_edge("verifier", "report_writer")
graph.add_edge("report_writer", END)
graph.add_edge("error", END)

graph.set_entry_point("router")

checkpointer = MemorySaver()
# Interrupt before subset_modeler — human sees prelim report and assessment first
app = graph.compile(checkpointer=checkpointer, interrupt_before=["subset_modeler"])
