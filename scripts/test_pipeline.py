from dotenv import load_dotenv

load_dotenv()

import json
import uuid
import time
from pathlib import Path
from bioinsight.graph import app
import logging

logging.basicConfig(level=logging.INFO)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

t0 = time.time()

# First run — goes until interrupt before subset_modeler
for chunk in app.stream(
    {
        "user_query": "What are the emerging grant themes in Parkinson's disease for 2025?"
    },
    config=config,
):
    pass

state = app.get_state(config)
print(f"Paused at: {state.next}  ({time.time() - t0:.1f}s)")

# Resume through to completion
for chunk in app.stream(None, config=config):
    pass

elapsed = time.time() - t0
state = app.get_state(config)

clusters = state.values.get("clusters", {})
verified = state.values.get("verified_cluster_findings", {})
print(f"\nDone in {elapsed:.1f}s")
print(
    f"Clusters: {len(clusters)}  |  Clusters with verified findings: {sum(1 for v in verified.values() if v)}"
)

print("\n" + "=" * 60)
print(state.values.get("final_answer", "No report generated"))

sidecar = state.values.get("report_sidecar")
if sidecar:
    data = json.loads(sidecar)
    print(f"\n--- SIDECAR ---")
    print(f"Clusters with findings: {len(data['cluster_findings'])}")
    print(f"Evidence entries: {len(data['evidence_index'])}")

# Save docx
report_docx = state.values.get("report_docx")
if report_docx:
    out_path = Path("scripts/test_report.docx")
    out_path.write_bytes(report_docx)
    print(f"\nWord document saved → {out_path.resolve()}")
else:
    print("\nNo docx generated.")
