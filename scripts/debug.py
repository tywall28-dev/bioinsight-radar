from dotenv import load_dotenv

load_dotenv()

from bioinsight.graph import app

for chunk in app.stream(
    {
        "user_query": "What are the emerging research themes in LRRK2 Parkinson's disease from 2023 to 2025?"
    }
):
    print(f"Node: {list(chunk.keys())[0]}")
    print(f"State: {list(chunk.values())[0].keys()}")
    print(f"search_terms: {list(chunk.values())[0].get('search_terms')}")
    print("---")
    break  # just first node
