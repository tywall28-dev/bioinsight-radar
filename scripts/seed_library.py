from bioinsight.chroma_manager import BioInsightChromaManager
from bioinsight.embedder import BioInsightEmbedder
from bioinsight.fetcher_tools import fetch_pubmed, fetch_nih_reporter

seed_domains = ["parkinsons", "autism"]
seed_years = [2023, 2024, 2025]

def seed_library():
    chroma = BioInsightChromaManager(persist_dir="./bioinsight_db")
    embedder = BioInsightEmbedder(model="dmis-lab/biobert-v1.1")

    for domain in seed_domains:
        for year in seed_years:
            # Fetch new records from PubMed and NIH Reporter
            pubmed_records = fetch_pubmed.invoke({
                "domain": domain,
                "year": year,
                "max_results": 100
            })
            
            nih_records = fetch_nih_reporter.invoke({
                "domain": domain,
                "fiscal_year": year,
                "max_results": 100
            })

            all_records = pubmed_records + nih_records
            print(f"Fetched {len(all_records)} records for {domain} in {year}.")

            # Embed and upsert into ChromaDB
            embedded_records = embedder.embed_records(all_records)
            chroma.upsert_records(embedded_records)
            print(f"Upserted {len(embedded_records)} records for {domain} in {year}.")

if __name__ == "__main__":
    seed_library()