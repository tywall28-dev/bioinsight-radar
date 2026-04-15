from dotenv import load_dotenv

load_dotenv()

from bioinsight.graph import app

result = app.invoke(
    {
        "user_query": "What are the emerging pubmed themes in LRKK2 Parkinson's disease in the past year"
    }
)
print(result["final_answer"])
