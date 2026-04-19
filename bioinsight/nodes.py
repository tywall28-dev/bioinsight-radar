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
import ast

load_dotenv()
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")  # fast, for routing
synthesis_llm = ChatAnthropic(model="claude-sonnet-4-6")  # powerful, for reports
chroma = BioInsightChromaManager(persist_dir="./bioinsight_db")
embedder = BioInsightEmbedder(model_name="dmis-lab/biobert-v1.1")


def router_node(state: AgentState) -> dict:
    # read from state
    prompt = f"""You are parsing a biomedical research question to query PubMed and NIH Reporter.

    Your tasks:
    1. Enrich the query for vector embedding (no years, just concepts and keywords)
    2. Extract 2-5 specific biomedical search terms suitable for PubMed/NIH APIs. Example for a Parkinson's LRRK2 query: ["LRRK2", "Parkinson Disease", "kinase inhibitor"] Do not include years. Use proper MeSH terminology.Search terms should be about the BIOMEDICAL TOPIC only — disease names, genes, proteins, mechanisms.  Never include terms about the data source like "grants", "funding", "publications", "research".
    3. Extract structured metadata

    Return ONLY valid JSON with this exact structure:
    {{
        "enriched_query": "...",
        "search_terms": ["keyword1", "keyword2"],
        "domain": "...",
        "years": [...],
        "entity": "...",
        "specificity": 1-5,
        "source_filter": "pubmed|nih_reporter|both"
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
    print(f"DEBUG full parsed output: {json.dumps(parsed, indent=2)}")

    # Generate query embedding
    try:
        query_vector = embedder.embed_query(parsed["enriched_query"])
    except Exception as e:
        print(f"ERROR: Failed to embed query: {e}")
        # Fallback: use a zero vector or skip embedding for now
        query_vector = [0.0] * 768
    print(f"DEBUG router returning search_terms: {parsed.get('search_terms', [])}")
    return_dict = {
        "query_vector": query_vector,
        "search_terms": parsed.get("search_terms", []),
        "enriched_query": parsed["enriched_query"],
        "domain": parsed["domain"],
        "years": parsed["years"],
        "entity": parsed.get("entity"),
        "specificity": parsed["specificity"],
        "source_filter": parsed["source_filter"],
    }
    print(f"DEBUG router return dict: {return_dict.keys()}")
    return return_dict


def library_checker_node(state: AgentState) -> dict:
    years = state["years"]
    query_vector = state.get("query_vector")
    specificity = state.get("specificity", 3)
    min_similarity = 0.2 + (specificity * 0.08)
    print(f"DEBUG search_terms type: {type(state.get('search_terms'))}")
    print(f"DEBUG search_terms value: {state.get('search_terms')}")

    if specificity >= 4 and state.get("fetch_attempts", 0) == 0:
        return {"library_has_data": False}

    library_has_data = all(
        chroma.semantic_search_by_year(
            query_vector, year, min_similarity=min_similarity
        )
        for year in years
    )
    total = chroma.count()
    print(f"DEBUG library_checker: total records in DB: {total}")
    print(f"DEBUG library_checker: min_similarity: {min_similarity}")
    return {"library_has_data": library_has_data}


def fetcher_node(state: AgentState) -> dict:
    search_terms = state.get("search_terms", [])

    if search_terms:
        main_term = search_terms[0]
        other_terms = " OR ".join(search_terms[1:])
        pubmed_query = f"{main_term} AND ({other_terms})" if other_terms else main_term
        nih_query = search_terms[0]
    else:
        pubmed_query = state.get("enriched_query", "")
        nih_query = state.get("enriched_query", "")

    years = state["years"]
    source_filter = state.get("source_filter")
    total_records = 0

    for year in years:
        all_records = []

        if source_filter in ("pubmed", "both", None):
            pubmed_records = fetch_pubmed.invoke(
                {"domain": pubmed_query, "year": year, "max_results": 100}
            )
            all_records += pubmed_records

        if source_filter in ("nih_reporter", "both", None):
            nih_records = fetch_nih_reporter.invoke(
                {"domain": nih_query, "fiscal_year": year, "max_results": 100}
            )
            all_records += nih_records

        total_records += len(all_records)
        embedded_records = embedder.embed_records(all_records)
        chroma.upsert_records(embedded_records)

    return {
        "records_fetched": total_records,
        "fetch_attempts": state.get("fetch_attempts", 0) + 1,
    }


def subset_modeler_node(state: AgentState) -> dict:
    query_vector = state.get("query_vector")
    years = state["years"]
    source_filter = state.get("source_filter")
    specificity = state.get("specificity", 3)
    k = max(50, 500 - (specificity - 1) * 100)

    year_filter = {"year": {"$in": years}}
    if source_filter in ("pubmed", "nih_reporter"):
        filters = {"$and": [{"year": {"$in": years}}, {"source": source_filter}]}
    else:
        filters = year_filter

    results = chroma.semantic_search(query_vector, k=k, filters=filters)
    if not results.get("documents") or len(results["documents"][0]) == 0:
        return {"clusters": {}, "error": "No records found for this query."}

    embeddings_matrix = np.array(results["embeddings"][0])
    umap_embeddings = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=5
    ).fit_transform(embeddings_matrix)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1)
    cluster_labels = clusterer.fit_predict(umap_embeddings)

    clusters = {}
    for i, (label, document) in enumerate(zip(cluster_labels, results["documents"][0])):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(
            {
                "id": results["ids"][0][i],
                "text": document,
                "metadata": results["metadatas"][0][i],
            }
        )

    return {"clusters": clusters}


def synthesis_node(state: AgentState) -> dict:
    enriched_query = state.get("enriched_query", state.get("domain", ""))
    years = state["years"]
    source_filter = state.get("source_filter")
    clusters = state.get("clusters", {})
    specificity = state.get("specificity")

    # Pass 1: Summarize each cluster with Haiku
    cluster_summaries = {}
    for label, docs in clusters.items():
        if label == -1:
            continue
        docs_text = "\n\n".join([doc["text"][:800] for doc in docs[:10]])
        summary_prompt = f"""Summarize the following biomedical research abstracts in 3-4 sentences.
Focus on: what is being studied, key findings, and therapeutic implications.

Abstracts:
{docs_text}"""
        summary = llm.invoke(summary_prompt)
        cluster_summaries[label] = summary.content

    # Pass 2: Synthesize all cluster summaries with Sonnet
    synthesis_prompt = f"""You are a biomedical research analyst writing a report for a program officer.

Query: {enriched_query}
Years: {years}
Source: {source_filter}
Specificity: {specificity}

Research cluster summaries ({len(cluster_summaries)} clusters identified):
{json.dumps(cluster_summaries, indent=2)}

Write a structured report with these sections:
1. Overview
2. Major Research Themes (one per cluster)
3. White Space / Gaps
4. Rising Signals"""

    response = synthesis_llm.invoke(synthesis_prompt)
    return {"final_answer": response.content}


def error_node(state: AgentState) -> dict:
    attempts = state.get("fetch_attempts", 0)
    search_terms = state.get("search_terms", [])
    return {
        "final_answer": f"Could not find sufficient relevant data after {attempts} fetch attempts. "
        f"Search terms used: {search_terms}. "
        f"Try broadening your query, using different terminology, or expanding the date range."
    }
