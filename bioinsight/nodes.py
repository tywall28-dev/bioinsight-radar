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
embedder = BioInsightEmbedder(model_name="dmis-lab/biobert-v1.1")


def router_node(state: AgentState) -> dict:
    # read from state
    prompt = f"""You are parsing a biomedical research question that is aiming to pull either pubmed articles or NIH reporter grants. 
        Your task is to enrish the user query so that it can be embedded and used to query a vector database of biomedical research. You also need to extract the following information from the user query and return it in JSON format:
    
        Extract the following and return ONLY valid JSON, no other text:
        {{
            "enriched_query": "a rewritten version of the user query that is optimized for embedding and retrieval. It should be more specific not include years or any information that could reduce similarity search efficiency and include relevant keywords.",
            "domain": "the disease or research area (e.g. parkinsons, autism)",
            "years": [list of years mentioned or implied],
            "entity": "the specific entity or concept being researched",
            "specificity": [1 to 5, where 1 is very broad and 5 is very specific],
            "source_filter": "pubmed, nih_reporter, both"
        }}

        User question: {state["user_query"]}"""

    response = llm.invoke(prompt)
    content = response.content.strip()
    # Strip markdown code blocks if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()
    parsed = json.loads(content)
    print(f"DEBUG Router output: {parsed}")  # add this

    DOMAIN_MAP = {
        "parkinson's disease": "parkinsons",
        "parkinson disease": "parkinsons",
        "parkinsons disease": "parkinsons",
        "alzheimer's disease": "alzheimers",
        "alzheimer disease": "alzheimers",
        "autistic disorder": "autism",
        "autism spectrum disorder": "autism",
        "amyotrophic lateral sclerosis": "als",
        "als": "als",
        "lou gehrig's disease": "als",
    }

    domain = parsed["domain"].lower()
    domain = DOMAIN_MAP.get(domain, domain)

    # Generate query embedding
    try:
        query_vector = embedder.embed_query(parsed["enriched_query"])
        print(
            f"DEBUG: Query vector generated successfully: {query_vector[:5]}..."
        )  # add this
    except Exception as e:
        print(f"ERROR: Failed to embed query: {e}")
        # Fallback: use a zero vector or skip embedding for now
        query_vector = [0.0] * 768

    return {
        "query_vector": query_vector,
        "enriched_query": parsed["enriched_query"],
        "domain": domain,
        "years": parsed["years"],
        "entity": parsed.get("entity"),
        "specificity": parsed["specificity"],
        "source_filter": parsed["source_filter"],
    }


def library_checker_node(state: AgentState) -> dict:
    # read parameters from state
    years = state["years"]
    query_vector = state.get("query_vector")

    # call library checker logic - check for semantically relevant records
    library_has_data = all(
        chroma.semantic_search_by_year(query_vector, year) for year in years
    )
    return {"library_has_data": library_has_data}


def fetcher_node(state: AgentState) -> dict:
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
    if results["embeddings"] is None or len(results["embeddings"]) == 0:
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


def synthesis_node(state: AgentState) -> dict:
    domain = state["domain"]
    years = state["years"]
    source_filter = state.get("source_filter")
    clusters = state.get("clusters", {})
    specificity = state.get("specificity")
    cluster_summaries = {
        label: docs[:6]
        for label, docs in clusters.items()
        if label != -1  # skip noise cluster
    }

    prompt = f"""You are a biomedical research analyst writing a report for a program officer.

        Domain: {domain}
        Years: {years}
        Source: {source_filter}
        Specificity: {specificity}

        Research clusters identified:
        {json.dumps(cluster_summaries, indent=2)}

        Write a structured report with these sections:
        1. Overview
        2. Major Research Themes (one per cluster)
        3. White Space / Gaps
        4. Rising Signals"""

    response = llm.invoke(prompt)
    # Placeholder for synthesis logic
    # In a real implementation, this would take the clustered data and generate a final answer
    return {"final_answer": response.content}
