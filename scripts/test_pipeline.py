from bioinsight.graph import app

result = app.invoke(
    {
        "user_query": "What are the emerging grant themes in Parkinson's disease from 2023 to 2025?"
    }
)

print(result["final_answer"])
