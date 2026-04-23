import streamlit as st
import uuid
from dotenv import load_dotenv

load_dotenv()

from bioinsight.graph import app

st.set_page_config(page_title="BioInsight Radar", page_icon="🔬", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "stage" not in st.session_state:
    st.session_state.stage = "waiting_for_query"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "current_state" not in st.session_state:
    st.session_state.current_state = None
if "projects" not in st.session_state:
    st.session_state.projects = {}
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

def render_coverage_breakdown(bd: dict):
    """Render the structured coverage breakdown panel."""
    if not bd:
        return
    with st.expander("📊 Coverage Breakdown", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Unique Docs", bd.get("total_unique_docs", "—"))
        m2.metric("PubMed", bd.get("pubmed_docs", "—"))
        m3.metric("NIH Grants", bd.get("nih_docs", "—"))
        m4.metric("Passages in Scope", bd.get("total_passages_in_scope", "—"))

        year_dist = bd.get("year_distribution", {})
        if year_dist:
            st.caption("Year distribution")
            st.bar_chart(year_dist)

        top_terms = bd.get("top_terms", [])
        if top_terms:
            pubmed_docs = bd.get("pubmed_docs", 0)
            nih_docs = bd.get("nih_docs", 0)
            term_source = "MeSH terms" if pubmed_docs >= nih_docs else "Title terms"
            st.caption(f"Top topics by frequency ({term_source})")
            st.dataframe(
                {
                    "Term": [t["term"] for t in top_terms],
                    "Count": [t["count"] for t in top_terms],
                },
                use_container_width=True,
                hide_index=True,
            )

        qcov = bd.get("query_term_coverage", {})
        if qcov:
            st.caption("Query term coverage")
            cols = st.columns(min(len(qcov), 4))
            for i, (term, found) in enumerate(qcov.items()):
                cols[i % len(cols)].markdown(f"{'✅' if found else '❌'} `{term}`")


col_chat, col_library = st.columns([2, 1])

with col_chat:
    st.title("🔬 BioInsight Radar")
    st.caption("Biomedical Research Intelligence")

    for message in st.session_state.messages:
        if message["role"] == "docx":
            filename = message.get("query", "report")[:40].replace(" ", "_") + ".docx"
            st.download_button(
                label="⬇️ Download Report (.docx)",
                data=message["content"],
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # ── Stage 1: accept the user's question ───────────────────────────────
    if st.session_state.stage == "waiting_for_query":
        if prompt := st.chat_input("Ask a research question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.stage = "running_pipeline"
            st.session_state.current_query = prompt
            st.rerun()

    # ── Stage 2: run router → library_checker → fetcher(s) →
    #             material_assessor → prelim_report → pause ─────────────────
    elif st.session_state.stage == "running_pipeline":
        with st.spinner("Searching library, assessing data, generating preview..."):
            thread_id = str(uuid.uuid4())
            st.session_state.thread_id = thread_id
            config = {"configurable": {"thread_id": thread_id}}

            try:
                for chunk in app.stream(
                    {"user_query": st.session_state.current_query}, config=config
                ):
                    pass
                state = app.get_state(config)
                st.session_state.current_state = state.values
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.session_state.stage = "waiting_for_query"
                st.stop()

        s = st.session_state.current_state
        action = s.get("assessment_action", "proceed")
        coverage_indicator = "✅ Good coverage" if action == "proceed" else "⚠️ Limited coverage — more data recommended"

        preview_msg = f"""**Query interpreted as:**
Search terms: `{s.get('search_terms', [])}`  |  Years: `{s.get('years', [])}`  |  Source: `{s.get('source_filter', 'both')}`

---

**Data Assessment:** {coverage_indicator}

{s.get('material_assessment', '')}

---

**Preliminary Overview**

{s.get('prelim_report', '')}

---

*Run the full analysis to get clustered themes, verified findings, and inline citations.*"""

        st.session_state.messages.append({"role": "assistant", "content": preview_msg})
        st.session_state.stage = "waiting_for_decision"
        st.rerun()

    # ── Stage 3: human decides ────────────────────────────────────────────
    elif st.session_state.stage == "waiting_for_decision":
        s = st.session_state.current_state
        suggested = s.get("suggested_terms") or []

        render_coverage_breakdown(s.get("coverage_breakdown") or {})

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔬 Run Full Analysis", use_container_width=True):
                st.session_state.stage = "running_analysis"
                st.rerun()
        with col2:
            fetch_label = f"📥 Fetch More ({', '.join(suggested[:2])})" if suggested else "📥 Fetch More Data"
            if st.button(fetch_label, use_container_width=True):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                existing = s.get("search_terms") or []
                existing_set = {t.lower() for t in existing}
                # Keep the anchor (existing[0]) + only add genuinely new terms.
                # The fetcher runs anchor-only + one focused query per new term,
                # so don't re-add terms that were already fetched.
                new_terms = [t for t in suggested if t.lower() not in existing_set]
                anchor = existing[:1]  # preserve the primary disease anchor
                update = {
                    "library_has_data": False,
                    "search_terms": list(dict.fromkeys(anchor + new_terms)),
                }
                app.update_state(config, update, as_node="library_checker")
                st.session_state.stage = "running_pipeline_resume"
                st.rerun()
        with col3:
            if st.button("✏️ New Query", use_container_width=True):
                st.session_state.messages = []
                st.session_state.stage = "waiting_for_query"
                st.rerun()

    # ── Stage 3b: resume after human requests more data ───────────────────
    elif st.session_state.stage == "running_pipeline_resume":
        with st.spinner("Fetching additional data and re-assessing..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            for chunk in app.stream(None, config=config):
                pass
            state = app.get_state(config)
            st.session_state.current_state = state.values

        s = st.session_state.current_state
        action = s.get("assessment_action", "proceed")
        coverage_indicator = "✅ Good coverage" if action == "proceed" else "⚠️ Still limited — consider broadening the query"

        preview_msg = f"""**Updated Data Assessment:** {coverage_indicator}

{s.get('material_assessment', '')}

---

**Updated Preliminary Overview**

{s.get('prelim_report', '')}"""

        st.session_state.messages.append({"role": "assistant", "content": preview_msg})
        st.session_state.stage = "waiting_for_decision"
        st.rerun()

    # ── Stage 4: run full analysis ────────────────────────────────────────

    elif st.session_state.stage == "running_analysis":
        with st.spinner("Running topic modeling, extracting and verifying findings... this takes a few minutes."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            try:
                for chunk in app.stream(None, config=config):
                    pass
                state = app.get_state(config)
                st.session_state.current_state = state.values
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.session_state.stage = "waiting_for_query"
                st.stop()

        final_answer = st.session_state.current_state.get("final_answer")
        if not final_answer:
            st.error("No report generated.")
            st.session_state.stage = "waiting_for_query"
            st.stop()

        st.session_state.messages.append({"role": "assistant", "content": final_answer})

        report_docx = st.session_state.current_state.get("report_docx")
        if report_docx:
            st.session_state.messages.append({
                "role": "docx",
                "content": report_docx,
                "query": st.session_state.current_state.get("enriched_query", "report"),
            })

        s = st.session_state.current_state
        project_name = s.get("enriched_query", "Research Project")[:40]
        st.session_state.projects[project_name] = {
            "record_count": s.get("records_fetched", 0),
            "years": s.get("years", []),
            "source": s.get("source_filter", "both"),
        }

        st.session_state.stage = "waiting_for_query"
        st.rerun()

with col_library:
    st.subheader("Research Library")
    if st.session_state.projects:
        tabs = st.tabs(list(st.session_state.projects.keys()))
        for i, (project, data) in enumerate(st.session_state.projects.items()):
            with tabs[i]:
                st.metric("Records", data.get("record_count", 0))
                st.write(f"Years: {data.get('years', [])}")
                st.write(f"Source: {data.get('source', 'both')}")
    else:
        st.info("No research projects yet. Ask a question to get started.")
