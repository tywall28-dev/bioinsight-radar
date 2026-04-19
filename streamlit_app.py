import streamlit as st
import uuid
from dotenv import load_dotenv

load_dotenv()

import os
from bioinsight.graph import app
from bioinsight.chroma_manager import BioInsightChromaManager

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

col_chat, col_library = st.columns([2, 1])

with col_chat:
    st.title("🔬 BioInsight Radar")
    st.caption("Biomedical Research Intelligence")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.stage == "waiting_for_query":
        if prompt := st.chat_input("Ask a research question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.stage = "running_router"
            st.session_state.current_query = prompt
            st.rerun()

    elif st.session_state.stage == "running_router":
        with st.spinner("Interpreting your query..."):
            thread_id = str(uuid.uuid4())
            st.session_state.thread_id = thread_id
            config = {"configurable": {"thread_id": thread_id}}

            # Clear DB for fresh analysis
            chroma = BioInsightChromaManager(persist_dir="./bioinsight_db")
            chroma._collection.delete(
                where={"source": {"$in": ["pubmed", "nih_reporter"]}}
            )

            # Run router — graph pauses at subset_modeler interrupt
            for chunk in app.stream(
                {"user_query": st.session_state.current_query}, config=config
            ):
                pass

            state = app.get_state(config)
            st.session_state.current_state = state.values

        s = st.session_state.current_state
        confirmation_msg = f"""I interpreted your query as:

**Search terms:** {s.get('search_terms', [])}
**Years:** {s.get('years', [])}
**Source:** {s.get('source_filter', 'both')}
**Specificity:** {s.get('specificity', 3)}/5

**Enriched query:** {s.get('enriched_query', '')}

Does this look right? Confirm to proceed or tell me what to change."""

        st.session_state.messages.append(
            {"role": "assistant", "content": confirmation_msg}
        )
        st.session_state.stage = "waiting_for_confirmation"
        st.rerun()

    elif st.session_state.stage == "waiting_for_confirmation":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm — Search & Fetch", use_container_width=True):
                st.session_state.stage = "running_fetch"
                st.rerun()
        with col2:
            if st.button("✏️ Edit query", use_container_width=True):
                st.session_state.stage = "waiting_for_query"
                st.session_state.messages = []
                st.rerun()

    elif st.session_state.stage == "running_fetch":
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        state = app.get_state(config)
        st.write(f"Before fetch - next nodes: {state.next}")
        st.write(f"Thread ID: {st.session_state.thread_id}")
        with st.spinner("Searching library and fetching records..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            # Resume — runs library_checker, fetcher cycles, stops at subset_modeler
            for chunk in app.stream(None, config=config):
                pass

            state = app.get_state(config)
            st.session_state.current_state = state.values

        s = st.session_state.current_state
        records = s.get("records_fetched", 0)

        fetch_msg = f"""I searched the library and fetched data.

**Records found:** {records}
**Years covered:** {s.get('years', [])}
**Source:** {s.get('source_filter', 'both')}

Ready to run topic modeling and generate your report?"""

        st.session_state.messages.append({"role": "assistant", "content": fetch_msg})
        st.session_state.stage = "waiting_for_analysis_confirmation"
        st.rerun()

    elif st.session_state.stage == "waiting_for_analysis_confirmation":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔬 Run Analysis", use_container_width=True):
                st.session_state.stage = "running_analysis"
                st.rerun()
        with col2:
            if st.button("📥 Fetch More Records", use_container_width=True):
                st.session_state.stage = "running_fetch"
                st.rerun()

    elif st.session_state.stage == "running_analysis":
        with st.spinner(
            "Running topic modeling and generating report... this may take a minute."
        ):
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
            st.error(f"No report generated. Next nodes: {app.get_state(config).next}")
            st.write(f"State keys: {list(st.session_state.current_state.keys())}")
            st.session_state.stage = "waiting_for_query"
            st.stop()

        st.session_state.messages.append({"role": "assistant", "content": final_answer})

        # Add to projects library
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
