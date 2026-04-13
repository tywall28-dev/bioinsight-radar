import os
import json
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from bioinsight.state import AgentState
from bioinsight.chroma_manager import BioInsightChromaManager
from bioinsight.fetcher_tools import fetch_pubmed, fetch_nih_reporter

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
            "query_type": "macro or micro"
        }}

        macro = broad topic landscape questions
        micro = specific mechanism or pathway questions

        User question: {state["user_query"]}"""

    response = llm.invoke(prompt)
    parsed = json.loads(response.content)

    return {
        "domain": parsed["domain"],
        "years": parsed["years"],
        "query_type": parsed["query_type"],
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
