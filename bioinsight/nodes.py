import os
import json
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from bioinsight.state import AgentState
from bioinsight.chroma_manager import BioInsightChromaManager
from bioinsight.fetcher_tools import fetch_pubmed, fetch_nih_reporter
from bioinsight.embedder import BioInsightEmbedder
import umap
import hdbscan
import numpy as np

load_dotenv()
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
chroma = BioInsightChromaManager(persist_dir="./bioinsight_db")
embedder = BioInsightEmbedder(model="dmis-lab/biobert-v1.1")


def router_node(state: AgentState) -> dict:
    # read from state
    prompt = f"""You are parsing a biomedical research question.
    
        Extract the following and return ONLY valid JSON, no other text:
        {{
            "domain": "the disease or research area (e.g. parkinsons, autism)",
            "years": [list of years mentioned or implied],
            "entity": "the specific entity or concept being researched",
            "specificity": [1 to 5, where 1 is very broad and 5 is very specific],
            "source_filter": "pubmed, nih_reporter, both"
        }}

        User question: {state["user_query"]}"""

    response = llm.invoke(prompt)
    parsed = json.loads(response.content)

    return {
        "domain": parsed["domain"],
        "years": parsed["years"],
        "entity": parsed.get("entity"),
        "specificity": parsed["specificity"],
        "source_filter": parsed["source_filter"],
    }


def library_checker_node(state: AgentState) -> dict:
    # read parameters from state
    domain = state["domain"]
    years = state["years"]

    # call library checker logic
    library_has_data = all(chroma.domain_year_exists(domain, year) for year in years)
    return {"library_has_data": library_has_data}


def fetcher_node(state: AgentState) -> dict:
    # Placeholder for fetcher logic
    # In a real implementation, this would call external APIs to fetch data
    domain = state["domain"]
    years = state["years"]
    total_records = 0

    for year in years:
        # Fetch new records from PubMed and NIH Reporter
        pubmed_records = fetch_pubmed.invoke(
            {"domain": domain, "year": year, "max_results": 100}
        )

        nih_records = fetch_nih_reporter.invoke(
            {"domain": domain, "fiscal_year": year, "max_results": 100}
        )

        all_records = pubmed_records + nih_records
        total_records += len(all_records)
        # Embed and upsert into ChromaDB
        embedded_records = embedder.embed_records(all_records)
        chroma.upsert_records(embedded_records)

    return {
        "records_fetched": total_records,
        "fetch_attempts": state.get("fetch_attempts", 0) + 1,
    }


def subset_modeler_node(state: AgentState) -> dict:
    domain = state["domain"]
    years = state["years"]
    source_filter = state.get("source_filter")
    results = chroma.get_domain_subset(domain, years, source_filter)
    if not results["embeddings"]:
        return {"clusters": {}, "error": "No records found for this query."}

    embeddings_matrix = np.array(results["embeddings"])
    umap_embeddings = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=5
    ).fit_transform(embeddings_matrix)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1)
    cluster_labels = clusterer.fit_predict(umap_embeddings)

    clusters = {}
    for label, document in zip(cluster_labels, results["documents"]):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(document)

    return {"clusters": clusters}
