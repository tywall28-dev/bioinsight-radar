
# BioInsight-Radar

**Biomedical research intelligence powered by LangGraph, BioBERT, and Claude.**

BioInsight Radar is an agentic pipeline that turns a natural-language research question into a structured intelligence report. It pulls from PubMed literature and NIH-funded grants, clusters findings by topic, verifies every claim against its source, and produces an annotated Word document with inline citations. Because the pipeline relies on large language models, verification is probabilistic rather than absolute, but source-level citation allows readers to confirm each claim independently.

> **Note on hardware:** Embeddings currently run locally using BioBERT via PyTorch. A GPU is strongly recommended for reasonable ingestion speed. CPU will work but is significantly slower, especially on larger fetch modes.

---

## Demo

<video src="https://github.com/user-attachments/assets/1bc09cbc-7a56-407d-90da-5629cdff7030" width="600" controls></video>

[infralimbic_prefrontal_cortex_research.docx](https://github.com/user-attachments/files/27110909/infralimbic_prefrontal_cortex_research.docx)
---

## What It Does

You type a question like:

> *"What are the emerging treatment strategies in Parkinson's disease for 2024–2025?"*

BioInsight Radar:

1. **Parses and refines** your question into structured search terms, separating what you explicitly asked about from what it should broaden to cover
2. **Checks a persistent local library** — if relevant data already exists, fetching is skipped
3. **Fetches from PubMed and NIH Reporter** with configurable depth (150–3,000 docs/year/source)
4. **Shows a preliminary overview and data coverage panel** so you can decide whether to run the full analysis or fetch more before committing
5. **Clusters the corpus** with UMAP + HDBSCAN on 768-D BioBERT embeddings to find thematic groups without predefining categories
6. **Extracts and verifies findings** — each claim passes two independent gates: factually stated in the source, and directly relevant to your question
7. **Writes a structured report** with inline citations, entity tables, research gaps, and strategic commentary
8. **Exports to `.docx`** with clickable hyperlinks back to every source

---

## Key Features

- **Human-in-the-loop checkpoint** — you see a preliminary overview and data assessment before the full analysis runs, with the option to fetch more data targeting specific gaps
- **Persistent vector library** — ChromaDB stores BioBERT embeddings on disk; repeat queries on the same topic reuse ingested data without re-fetching
- **Sentence-level provenance** — each abstract and grant is split into sentence passages at ingest; citations trace back to the exact source document
- **Dual-source analysis** — query PubMed (published findings) and NIH Reporter (active grant funding) independently or together; the report separates "what research shows" from "where funding is going"
- **Configurable fetch depth** — Quick (~150 docs/yr), Standard (~400), Deep (~800), Everything (all available, cap 3,000)
- **Structured verification** — every extracted claim is independently checked for factual grounding and query relevance before it appears in the report

---

## Architecture

```
User Query
    |
    v
+-------------+    +------------------+
|   Router    |--->|  Query Refiner   |  Claude Haiku
|  (parse +   |    | (validate, clean |  - fast structured
|   embed)    |    |  meta-words)     |    parsing
+-------------+    +--------+---------+
                             |
                    +--------v---------+
                    | Library Checker  |  ChromaDB semantic
                    | (BioBERT cosine  |  search - is there
                    |  similarity)     |  enough relevant data?
                    +--------+---------+
                          +--+--+
                     yes  |     |  no
                          |     v
                          |  +---------+
                          |  | Fetcher |  PubMed + NIH Reporter
                          |  |         |  -> embed -> ChromaDB
                          |  +----+----+
                          |       | (loop until threshold met)
                          +-------+
                             |
                    +--------v----------+
                    | Material Assessor |  Claude Haiku
                    | + Prelim Report   |  - data quality check
                    +--------+----------+
                             |
                   ==========|==========
                    HUMAN CHECKPOINT
                    Proceed / Fetch More
                   ==========|==========
                             |
                    +--------v----------+
                    |  Subset Modeler   |  UMAP (768D -> 5D)
                    |                   |  + HDBSCAN clustering
                    +--------+----------+
                             |
                    +--------v----------+
                    |    Extraction     |  Claude Haiku
                    |  (per cluster)    |  - cite findings
                    +--------+----------+
                             |
                    +--------v----------+
                    |    Verifier       |  Claude Haiku
                    |  (two-gate check) |  - supported + relevant
                    +--------+----------+
                             |
                    +--------v----------+
                    |  Report Writer    |  Claude Sonnet
                    |  + .docx export   |  - full synthesis
                    +-------------------+
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) StateGraph with human-in-the-loop interrupt |
| LLMs | Claude Haiku (routing, extraction, verification) / Claude Sonnet (synthesis) |
| Embeddings | [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1) `dmis-lab/biobert-v1.1` (768-D, runs locally) |
| Vector store | [ChromaDB](https://www.trychroma.com/) persistent, cosine similarity |
| Clustering | UMAP (768-D → 5-D) + HDBSCAN |
| Literature | PubMed via [metapub](https://github.com/metapub/metapub) + NCBI E-utilities |
| Grants | [NIH Reporter API v2](https://api.reporter.nih.gov/) |
| Sentence splitting | spaCy `en_core_web_sm` |
| UI | Streamlit |
| Report export | python-docx with clickable hyperlinks |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/bioinsight-radar.git
cd bioinsight-radar
pip install -e .
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional but recommended — increases PubMed rate limits significantly
NCBI_API_KEY=your_ncbi_api_key

# Optional — limits docs fetched per year, useful for dev/CI
BIOINSIGHT_TEST_LIMIT=50
```

Get an NCBI API key for free at [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/).

### 3. Run the app

```bash
streamlit run streamlit_app.py
```

The ChromaDB library is created at `./bioinsight_db/` on first run and persists across sessions.

---

## Usage

**Ask a research question** in the chat input. Be as specific or broad as you like:

- *"What biomarkers are being studied for early Alzheimer's detection in 2024–2025?"*
- *"Where is NIH funding going for autism spectrum disorder research in the last 3 years?"*
- *"What are the latest LRRK2-targeted therapies in Parkinson's disease?"*

**Choose fetch depth** in the sidebar before querying. Standard is a good starting point; use Deep or Everything for thorough analysis.

**Review the preliminary overview.** The app pauses here and shows you a coverage breakdown panel (document counts by year and source, top terms, query coverage). You can:

- **Run Full Analysis** — proceeds to clustering, extraction, and report writing
- **Fetch More** — targets gaps the assessor identified with additional search terms
- **New Query** — start over

**Download the `.docx`** from the final report. Every citation is a clickable link back to the original PubMed abstract or NIH Reporter grant page.

---

## Project Structure

```
bioinsight-radar/
├── streamlit_app.py          # Streamlit UI + graph interaction
├── bioinsight/
│   ├── graph.py              # LangGraph StateGraph definition
│   ├── state.py              # AgentState TypedDict
│   ├── nodes.py              # All node implementations
│   ├── fetcher_tools.py      # PubMed + NIH Reporter fetchers
│   ├── chroma_manager.py     # ChromaDB read/write layer
│   └── embedder.py           # BioBERT embedding service
├── scripts/
│   ├── test_pipeline.py      # End-to-end pipeline test (no UI)
│   ├── seed_library.py       # Pre-populate the library for a domain
│   └── check_chroma.py       # Inspect ChromaDB collection stats
├── bioinsight_db/            # ChromaDB persistent storage (gitignored)
└── requirements.txt
```

---

## Fetch Modes

| Mode | Docs / year / source | Best for |
|---|---|---|
| Quick | ~150 | Fast exploration of known topics |
| Standard | ~400 | Balanced, good default |
| Deep | ~800 | Thorough analysis of emerging areas |
| Everything | All available (cap 3,000) | Complete corpus coverage |

Availability is checked before each fetch. If fewer records exist for a given year or source, the full available set is used automatically.

---

## How Verification Works

Every extracted finding passes two independent gates before it appears in the report:

1. **Factual support** — is this claim explicitly stated as an established result in the source document? Research aims, hypotheses, and background statements are rejected.
2. **Query relevance** — does this claim directly address what you asked? Methodological details and tangential findings are rejected.

Both gates must pass. Claims that fail either one are dropped and logged with a reason.

---

## License

MIT
