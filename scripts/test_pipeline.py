from dotenv import load_dotenv

load_dotenv()

from bioinsight.graph import app

result = app.invoke(
    {
        "user_query": "What are the emerging grant themes in Parkinson's disease research from 2023 to 2026?"
    }
)
print(result["final_answer"])
