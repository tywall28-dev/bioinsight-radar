from bioinsight.chroma_manager import BioInsightRecord, BioInsightChromaManager

record = BioInsightRecord(
    source="pubmed",
    domain="parkinsons",
    year=2024,
    pi_name="Jane Smith",
    org_name="Colorado State University",
    external_id="12345678",
    granularity="field",
    text="This is a fake abstract about dopamine dysregulation.",
    title="A fake paper title",
    embedding=[0.1] * 768
)

manager = BioInsightChromaManager(persist_dir="./bioinsight_db")
manager.upsert_records([record])
print(manager.collection_summary())