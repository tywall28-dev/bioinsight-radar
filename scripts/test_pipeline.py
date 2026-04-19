from dotenv import load_dotenv

load_dotenv()

import uuid
from bioinsight.graph import app

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# First run — goes until interrupt
for chunk in app.stream(
    {
        "user_query": "What are the emerging grant themes in Parkinson's disease from 2023 to 2025?"
    },
    config=config,
):
    pass

# Check where we are
state = app.get_state(config)
print(f"Paused at: {state.next}")

# Resume through to completion
for chunk in app.stream(None, config=config):
    pass

state = app.get_state(config)
clusters = state.values.get("clusters", {})
if clusters:
    first_label = list(clusters.keys())[0]
    print(f"Cluster {first_label} first record:")
    print(clusters[first_label][0])

state = app.get_state(config)
print(state.values.get("final_answer", state.values.get("error", "No output")))
