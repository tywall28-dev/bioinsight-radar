from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from bioinsight.state import AgentState
from langgraph.checkpoint.memory import MemorySaver
from bioinsight.nodes import (
    fetcher_node,
    router_node,
    mesh_enrichment_node,
    query_refiner_node,
    query_approval_node,
    corpus_scout_node,
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
graph.add_node("mesh_enrichment", mesh_enrichment_node)
graph.add_node("query_refiner", query_refiner_node)
graph.add_node("query_approval", query_approval_node)
graph.add_node("library_checker", library_checker_node)
graph.add_node("corpus_scout", corpus_scout_node)
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
        return "material_assessor"
    else:
        return "corpus_scout"


graph.add_edge("router", "mesh_enrichment")
graph.add_edge("mesh_enrichment", "query_refiner")
graph.add_edge("query_refiner", "query_approval")
graph.add_edge("query_approval", "library_checker")
graph.add_conditional_edges(
    "library_checker",
    should_fetch,
    {
        "corpus_scout": "corpus_scout",
        "material_assessor": "material_assessor",
    },
)
graph.add_edge("corpus_scout", "fetcher")
graph.add_edge("fetcher", "library_checker")
# material_assessor always proceeds to prelim_report.
# The human controls additional fetching via the "Fetch More" button,
# which re-enters the graph at library_checker with a targeted gap-fill query.
graph.add_edge("material_assessor", "prelim_report")
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
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["query_approval", "subset_modeler"],
    interrupt_after=["corpus_scout"],
)
