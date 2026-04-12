import os
import json
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from bioinsight.state import AgentState
from bioinsight.chroma_manager import BioInsightChromaManager

load_dotenv()
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
chroma = BioInsightChromaManager(persist_dir="./bioinsight_db")

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
    #read parameters from state
    domain = state["domain"]
    years = state["years"]
    
    # call library checker logic
    library_has_data = all(
        chroma.domain_year_exists(domain, year) 
        for year in years
    )
    
    return {
        "library_has_data": library_has_data
    }