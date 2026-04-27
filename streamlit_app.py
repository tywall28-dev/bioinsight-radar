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
if "fetch_mode" not in st.session_state:
    st.session_state.fetch_mode = "standard"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Fetch Settings")
    _MODE_LABELS = {
        "quick": "Quick  (~150 docs/yr) — fast exploration",
        "standard": "Standard  (~400 docs/yr) — balanced",
        "deep": "Deep  (~800 docs/yr) — thorough analysis",
        "everything": "Everything  (all available) — complete corpus",
    }
    selected = st.radio(
        "Data coverage",
        options=list(_MODE_LABELS.keys()),
        index=list(_MODE_LABELS.keys()).index(st.session_state.fetch_mode),
        format_func=lambda m: _MODE_LABELS[m],
    )
    st.session_state.fetch_mode = selected
    st.caption(
        "Counts are **per year per source** before deduplication. "
        "Availability is checked before each fetch — if fewer records exist the full set is used."
    )
    if selected == "everything":
        st.warning("Everything mode can be slow for broad queries. Hard cap: 3,000 docs/yr/source.")


def render_coverage_breakdown(bd: dict):
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
                width=600,
                hide_index=True,
            )

        qcov = bd.get("query_term_coverage", {})
        if qcov:
            st.caption("Query term coverage")
            cols = st.columns(min(len(qcov), 4))
            for i, (term, found) in enumerate(qcov.items()):
                cols[i % len(cols)].markdown(f"{'✅' if found else '❌'} `{term}`")


def _parse_csv(s: str) -> list[str]:
    return [t.strip() for t in s.split(",") if t.strip()]


def _source_label(source: str) -> str:
    return {
        "pubmed": "PubMed (published research)",
        "nih_reporter": "NIH Reporter (grant funding)",
        "both": "PubMed + NIH Reporter",
    }.get(source, source)


def _build_scout_message(s: dict) -> str:
    scout_results = s.get("scout_results") or {}
    total = s.get("scout_total", 0)
    source = s.get("source_filter", "both")
    years = sorted(s.get("years") or [])
    pubmed_q = s.get("pubmed_query", "")
    nih_q = s.get("nih_query", "")

    lines = [
        f"⚠️ **Only {total} records found across all years and sources before fetching.**",
        "",
        "Here's what's available:",
        "",
    ]
    # Table header
    if source == "both":
        lines += ["| Year | PubMed | NIH Reporter |", "|------|--------|--------------|"]
    elif source == "pubmed":
        lines += ["| Year | PubMed |", "|------|--------|"]
    else:
        lines += ["| Year | NIH Reporter |", "|------|--------------|"]

    for yr in years:
        yr_data = scout_results.get(str(yr), {})
        if source == "both":
            lines.append(f"| {yr} | {yr_data.get('pubmed', 0)} | {yr_data.get('nih_reporter', 0)} |")
        elif source == "pubmed":
            lines.append(f"| {yr} | {yr_data.get('pubmed', 0)} |")
        else:
            lines.append(f"| {yr} | {yr_data.get('nih_reporter', 0)} |")

    lines += [""]
    if pubmed_q and source in ("pubmed", "both"):
        lines.append(f"PubMed query used: `{pubmed_q}`")
    if nih_q and source in ("nih_reporter", "both"):
        lines.append(f"NIH Reporter query used: `{nih_q}`")

    lines += [
        "",
        "A small corpus means the analysis may be narrow or incomplete. Options:",
        "- **Proceed** — fetch what's there and see what the data shows",
        "- **New Question** — try broader terms or a wider year range",
    ]
    return "\n".join(lines)


def _render_fetch_stats(corpus_stats: dict, status_widget):
    """Write per-year fetch results into a status widget."""
    for yr, yst in sorted(corpus_stats.items()):
        parts = []
        if "pub_available" in yst:
            pub_a = yst["pub_available"]
            pub_f = yst.get("pub_fetched", 0)
            parts.append(
                "PubMed: no articles found"
                if pub_a == 0
                else f"PubMed: **{pub_f:,}** of {pub_a:,} available articles fetched"
            )
        if "nih_available" in yst:
            nih_a = yst["nih_available"]
            nih_f = yst.get("nih_fetched", 0)
            parts.append(
                "NIH Reporter: no grants found"
                if nih_a == 0
                else f"NIH Reporter: **{nih_f:,}** of {nih_a:,} available grants fetched"
            )
        if parts:
            status_widget.write(f"**{yr}** → " + "  |  ".join(parts))


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

    # ── Stage 1: accept the user's question ───────────────────────────────────
    if st.session_state.stage == "waiting_for_query":
        if prompt := st.chat_input("Ask a research question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.stage = "running_router"
            st.session_state.current_query = prompt
            st.rerun()

    # ── Stage 1b: run router + refiner + query_approval, pause ───────────────
    elif st.session_state.stage == "running_router":
        with st.spinner("Parsing your question..."):
            thread_id = str(uuid.uuid4())
            st.session_state.thread_id = thread_id
            config = {"configurable": {"thread_id": thread_id}}
            try:
                for _ in app.stream(
                    {"user_query": st.session_state.current_query, "fetch_mode": st.session_state.fetch_mode},
                    config=config,
                ):
                    pass
                state = app.get_state(config)
                st.session_state.current_state = state.values
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.session_state.stage = "waiting_for_query"
                st.stop()

        s = st.session_state.current_state
        anchor = (s.get("search_terms") or [""])[0]
        explicit = s.get("explicit_terms") or []
        expansion = s.get("expansion_terms") or []
        years = s.get("years") or []
        source = s.get("source_filter", "both")
        refinement = s.get("refinement_notes", "")
        clarifying = s.get("clarifying_question", "")
        mesh_context = s.get("mesh_context") or {}

        # Save original computed raw queries for change-detection at approval time
        st.session_state.original_pubmed_q = s.get("pubmed_query", "")
        st.session_state.original_nih_q = s.get("nih_query", "")

        lines = [
            "**Here's how I interpreted your question — review and approve before I search:**",
            "",
            f"- **Main topic:** `{anchor}`",
        ]
        if explicit:
            lines.append(f"- **Required terms** (you mentioned these — results must contain them): `{', '.join(explicit)}`")
        if expansion:
            lines.append(f"- **Broadening terms** (added to improve coverage, optional): `{', '.join(expansion)}`")
        lines += [
            f"- **Years:** `{', '.join(str(y) for y in years)}`",
            f"- **Sources:** {_source_label(source)}",
        ]
        if refinement:
            lines += ["", f"> 🔍 *{refinement}*"]

        # MeSH grounding block — show what NCBI says about each term
        if mesh_context:
            lines += ["", "**🔬 MeSH Terminology Check** (NCBI authoritative headings):"]
            for term, info in mesh_context.items():
                if not info.get("found"):
                    lines.append(f"  - `{term}` — not in MeSH (may be a gene name or very specific term — kept as-is)")
                    continue
                preferred = info.get("preferred_heading", "")
                is_preferred = info.get("is_preferred_heading", False)
                is_entry = info.get("is_entry_term", False)
                scope = info.get("scope_note", "")
                scope_snippet = f' · *"{scope}"*' if scope else ""
                if is_preferred:
                    lines.append(f"  - `{term}` → ✅ MeSH preferred heading{scope_snippet}")
                elif is_entry:
                    lines.append(f"  - `{term}` → entry term (synonym) under **{preferred}**{scope_snippet}")
                elif preferred:
                    lines.append(f"  - `{term}` → not in MeSH; closest heading **{preferred}** (different concept — kept your term)")
                else:
                    lines.append(f"  - `{term}` → not found in MeSH — kept your term as-is")

        if clarifying:
            lines += ["", f"**Before I search — one question:** {clarifying}",
                      "", "*Answer in the edit panel below, or just edit the fields directly and approve.*"]

        approval_msg = "\n".join(lines)
        st.session_state.messages.append({"role": "assistant", "content": approval_msg})
        st.session_state.stage = "waiting_for_query_approval"
        st.rerun()

    # ── Stage 2: let user approve or edit the parsed query ────────────────────
    elif st.session_state.stage == "waiting_for_query_approval":
        s = st.session_state.current_state
        anchor = (s.get("search_terms") or [""])[0]
        explicit = s.get("explicit_terms") or []
        expansion = s.get("expansion_terms") or []
        years = s.get("years") or []
        source = s.get("source_filter", "both")
        enriched = s.get("enriched_query", "")
        orig_pubmed_q = st.session_state.get("original_pubmed_q", "")
        orig_nih_q = st.session_state.get("original_nih_q", "")

        clarifying = s.get("clarifying_question", "")
        if clarifying:
            st.info(f"**To help refine your search:** {clarifying}")
            st.caption("Answer by editing the fields below, then approve.")

        with st.expander("✏️ Edit search terms", expanded=bool(clarifying)):
            new_anchor = st.text_input("Main topic (anchor)", value=anchor)
            new_explicit_str = st.text_input(
                "Required terms — comma-separated (must appear in results)",
                value=", ".join(explicit),
                help="Terms you explicitly asked about. Results are filtered to contain all of these.",
            )
            new_expansion_str = st.text_input(
                "Broadening terms — comma-separated (optional coverage)",
                value=", ".join(expansion),
                help="Added by the system to improve coverage. These are OR'd — results don't need to contain them.",
            )
            new_years_str = st.text_input(
                "Years (comma-separated)", value=", ".join(str(y) for y in years)
            )
            new_source = st.radio(
                "Sources",
                options=["pubmed", "nih_reporter", "both"],
                index=["pubmed", "nih_reporter", "both"].index(source),
                format_func=_source_label,
                horizontal=True,
            )

        with st.expander("🔬 Edit raw API queries (advanced)", expanded=False):
            st.caption(
                "These are the exact strings sent to each API. "
                "Editing here overrides the search terms above. "
                "Leave unchanged to have them rebuilt automatically from your edits above."
            )
            new_enriched = st.text_area(
                "Enriched query (used for semantic / vector search)",
                value=enriched,
                height=68,
                help="Free-text description of your intent — used to find semantically similar passages in the library via BioBERT embeddings.",
            )
            new_pubmed_q = ""
            new_nih_q = ""
            if source in ("pubmed", "both"):
                new_pubmed_q = st.text_area(
                    "PubMed query string",
                    value=orig_pubmed_q,
                    height=80,
                    help='Full PubMed query with field tags, e.g. "infralimbic cortex"[tiab] AND "fear extinction"[tiab]',
                )
            if source in ("nih_reporter", "both"):
                new_nih_q = st.text_area(
                    "NIH Reporter query string",
                    value=orig_nih_q,
                    height=80,
                    help='Full NIH Reporter query, e.g. "infralimbic cortex" AND "fear extinction"',
                )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve & Search", use_container_width=True, type="primary"):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                new_explicit = _parse_csv(new_explicit_str)
                new_expansion = _parse_csv(new_expansion_str)
                new_years = [int(y) for y in _parse_csv(new_years_str) if y.isdigit()]
                seen = {new_anchor}
                new_search_terms = [new_anchor]
                for t in new_explicit + new_expansion:
                    if t and t not in seen:
                        seen.add(t)
                        new_search_terms.append(t)

                # Honor raw query edits only if the user actually changed them
                final_pubmed_q = new_pubmed_q if new_pubmed_q != orig_pubmed_q else None
                final_nih_q = new_nih_q if new_nih_q != orig_nih_q else None

                update = {
                    "search_terms": new_search_terms,
                    "explicit_terms": new_explicit,
                    "expansion_terms": new_expansion,
                    "years": new_years or years,
                    "source_filter": new_source,
                    "enriched_query": new_enriched or enriched,
                    "pubmed_query": final_pubmed_q,
                    "nih_query": final_nih_q,
                }
                app.update_state(config, update)
                st.session_state.current_state = {**s, **update}
                st.session_state.stage = "running_pipeline"
                st.rerun()
        with col2:
            if st.button("✏️ New Question", use_container_width=True):
                st.session_state.messages = []
                st.session_state.stage = "waiting_for_query"
                st.rerun()

    # ── Stage 3: resume from query_approval → library_checker → prelim_report ─
    elif st.session_state.stage == "running_pipeline":
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        s = st.session_state.current_state
        source = s.get("source_filter", "both")
        years = s.get("years") or []

        scout_niche_found = False
        try:
            with st.status("Searching and fetching data...", expanded=True) as pipeline_status:
                pipeline_status.write("🔍 Checking local library for existing data...")

                while True:
                    for chunk in app.stream(None, config=config):
                        node_name = list(chunk.keys())[0]
                        node_output = chunk[node_name]

                        if node_name == "library_checker":
                            has_data = node_output.get("library_has_data", False)
                            if has_data:
                                pipeline_status.write("✅ Library already has relevant data — no API fetch needed.")
                            else:
                                pubmed_q = s.get("pubmed_query") or ""
                                nih_q = s.get("nih_query") or ""
                                pipeline_status.write(
                                    f"📭 Library doesn't have enough data. Checking availability in"
                                    f" **{_source_label(source)}**..."
                                )
                                if pubmed_q and source in ("pubmed", "both"):
                                    pipeline_status.write(f"PubMed query: `{pubmed_q}`")
                                if nih_q and source in ("nih_reporter", "both"):
                                    pipeline_status.write(f"NIH Reporter query: `{nih_q}`")

                        elif node_name == "corpus_scout":
                            pass  # handled after stream pauses below

                        elif node_name == "fetcher":
                            corpus_stats = node_output.get("corpus_stats", {})
                            _render_fetch_stats(corpus_stats, pipeline_status)
                            total = node_output.get("records_fetched", 0)
                            pipeline_status.write(f"✅ Fetch complete — **{total:,}** text passages added to library.")

                        elif node_name == "material_assessor":
                            pipeline_status.write("📊 Assessing data coverage and quality...")

                        elif node_name == "prelim_report":
                            pipeline_status.write("📝 Generating preliminary overview...")

                    # Stream paused — determine why
                    graph_state = app.get_state(config)
                    st.session_state.current_state = graph_state.values
                    next_nodes = list(graph_state.next or [])

                    if "fetcher" in next_nodes:
                        # interrupt_after corpus_scout fired
                        is_niche = graph_state.values.get("corpus_is_niche", False)
                        scout_total = graph_state.values.get("scout_total", 0)
                        scout_results = graph_state.values.get("scout_results") or {}
                        if is_niche:
                            pipeline_status.update(
                                label=f"⚠️ Only {scout_total} records available — checking with you before fetching.",
                                state="running",
                            )
                            scout_niche_found = True
                            break
                        else:
                            # Not niche — show counts and auto-continue
                            for yr_str, yr_data in sorted(scout_results.items()):
                                parts = []
                                if "pubmed" in yr_data:
                                    parts.append(f"PubMed: **{yr_data['pubmed']:,}** available")
                                if "nih_reporter" in yr_data:
                                    parts.append(f"NIH Reporter: **{yr_data['nih_reporter']:,}** available")
                                if parts:
                                    pipeline_status.write(f"**{yr_str}** → " + "  |  ".join(parts))
                            continue  # resume stream through fetcher

                    elif "subset_modeler" in next_nodes:
                        pipeline_status.update(label="Ready for your review.", state="complete")
                        break

                    else:
                        pipeline_status.update(label="Done.", state="complete")
                        break

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.session_state.stage = "waiting_for_query"
            st.stop()

        if scout_niche_found:
            scout_msg = _build_scout_message(st.session_state.current_state)
            st.session_state.messages.append({"role": "assistant", "content": scout_msg})
            st.session_state.stage = "scout_review"
            st.rerun()

        s = st.session_state.current_state
        if s.get("material_assessment"):
            action = s.get("assessment_action", "proceed")
            coverage_indicator = (
                "✅ Good coverage"
                if action == "proceed"
                else "⚠️ Limited coverage — more data recommended"
            )
        else:
            coverage_indicator = "⚠️ Very limited data — this topic may have few indexed records"

        activity_codes = s.get("activity_codes") or []
        nih_filter = f"  |  NIH mechanisms: `{activity_codes}`" if activity_codes else ""
        explicit = s.get("explicit_terms") or []
        expansion = s.get("expansion_terms") or []
        terms_display = ""
        if explicit:
            terms_display += f"  Required: `{explicit}`"
        if expansion:
            terms_display += f"  Broadened by: `{expansion}`"
        if not terms_display:
            terms_display = f"  Terms: `{s.get('search_terms', [])}`"

        refinement = s.get("refinement_notes", "")
        refinement_block = f"\n> 🔍 *{refinement}*\n" if refinement else ""

        corpus_stats = s.get("corpus_stats") or {}
        corpus_lines = []
        for yr, yst in sorted(corpus_stats.items()):
            parts = []
            if "pub_available" in yst:
                parts.append(f"PubMed {yst.get('pub_fetched', 0):,} / {yst['pub_available']:,} available")
            if "nih_available" in yst:
                parts.append(f"NIH {yst.get('nih_fetched', 0):,} / {yst['nih_available']:,} available")
            if parts:
                corpus_lines.append(f"**{yr}**: " + "  |  ".join(parts))
        corpus_block = "\n".join(corpus_lines) if corpus_lines else ""

        preview_msg = f"""**Query searched as:**
{terms_display}  |  Years: `{s.get('years', [])}`  |  Source: `{s.get('source_filter', 'both')}`{nih_filter}
{refinement_block}
{corpus_block}

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

    # ── Stage 3b: niche corpus detected — check with user before fetching ────
    elif st.session_state.stage == "scout_review":
        s = st.session_state.current_state
        total = s.get("scout_total", 0)
        years = s.get("years") or []
        source = s.get("source_filter", "both")

        st.info(
            f"**{total} records found** across {len(years)} year(s) in {_source_label(source)}. "
            "This is a very small corpus — results may be narrow or incomplete. "
            "You can proceed and see what the data shows, or start fresh with different terms."
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Proceed with limited data", use_container_width=True, type="primary"):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                app.update_state(config, {"scout_approved": True})
                st.session_state.current_state = {**s, "scout_approved": True}
                st.session_state.stage = "running_pipeline"
                st.rerun()
        with col2:
            if st.button("✏️ New Question", use_container_width=True):
                st.session_state.messages = []
                st.session_state.stage = "waiting_for_query"
                st.rerun()

    # ── Stage 4: human decides ────────────────────────────────────────────────
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
            fetch_label = (
                f"📥 Fetch More ({', '.join(suggested[:2])})" if suggested else "📥 Fetch More Data"
            )
            if st.button(fetch_label, use_container_width=True):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                existing = s.get("search_terms") or []
                existing_set = {t.lower() for t in existing}
                new_terms = [t for t in suggested if t.lower() not in existing_set]
                anchor = existing[:1]
                update = {
                    "library_has_data": False,
                    "search_terms": list(dict.fromkeys(anchor + new_terms)),
                    "explicit_terms": [],
                    "expansion_terms": new_terms,
                    "fetch_attempts": 0,
                    "years_needing_data": s.get("years") or [],
                    "pubmed_query": None,  # rebuild from new gap terms
                    "nih_query": None,
                    "scout_approved": True,  # user already saw the assessment; skip niche re-check
                }
                app.update_state(config, update, as_node="library_checker")
                st.session_state.current_state = {**s, **update}
                st.session_state.stage = "running_pipeline_resume"
                st.rerun()
        with col3:
            if st.button("✏️ New Query", use_container_width=True):
                st.session_state.messages = []
                st.session_state.stage = "waiting_for_query"
                st.rerun()

    # ── Stage 4b: resume after "Fetch More" ───────────────────────────────────
    elif st.session_state.stage == "running_pipeline_resume":
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        s = st.session_state.current_state
        source = s.get("source_filter", "both")
        years = s.get("years") or []
        expansion = s.get("expansion_terms") or []

        try:
            with st.status("Fetching additional data...", expanded=True) as resume_status:
                resume_status.write(
                    f"🔍 Targeting gaps: `{', '.join(expansion)}`"
                    f" from **{_source_label(source)}**"
                    f" (years: {', '.join(str(y) for y in years)})..."
                )

                while True:
                    for chunk in app.stream(None, config=config):
                        node_name = list(chunk.keys())[0]
                        node_output = chunk[node_name]

                        if node_name == "fetcher":
                            corpus_stats = node_output.get("corpus_stats", {})
                            _render_fetch_stats(corpus_stats, resume_status)
                            total = node_output.get("records_fetched", 0)
                            resume_status.write(f"✅ Fetch complete — **{total:,}** passages added.")

                        elif node_name == "library_checker":
                            if node_output.get("library_has_data", False):
                                resume_status.write("✅ Library updated — re-assessing coverage...")

                        elif node_name == "material_assessor":
                            resume_status.write("📊 Re-assessing data coverage...")

                        elif node_name == "prelim_report":
                            resume_status.write("📝 Updating preliminary overview...")

                    graph_state = app.get_state(config)
                    st.session_state.current_state = graph_state.values
                    next_nodes = list(graph_state.next or [])

                    if "fetcher" in next_nodes:
                        # corpus_scout fired — scout_approved=True so auto-continue
                        continue
                    elif "subset_modeler" in next_nodes:
                        resume_status.update(label="Update complete.", state="complete")
                        break
                    else:
                        resume_status.update(label="Done.", state="complete")
                        break

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.session_state.stage = "waiting_for_query"
            st.stop()

        s = st.session_state.current_state
        action = s.get("assessment_action", "proceed")
        coverage_indicator = (
            "✅ Good coverage"
            if action == "proceed"
            else "⚠️ Still limited — consider broadening the query"
        )

        corpus_stats = s.get("corpus_stats") or {}
        corpus_lines = []
        for yr, yst in sorted(corpus_stats.items()):
            parts = []
            if "pub_available" in yst:
                parts.append(f"PubMed {yst.get('pub_fetched', 0):,} / {yst['pub_available']:,} available")
            if "nih_available" in yst:
                parts.append(f"NIH {yst.get('nih_fetched', 0):,} / {yst['nih_available']:,} available")
            if parts:
                corpus_lines.append(f"**{yr}**: " + "  |  ".join(parts))
        corpus_block = "\n".join(corpus_lines) if corpus_lines else ""

        preview_msg = f"""**Updated Data Assessment:** {coverage_indicator}

{s.get('material_assessment', '')}

{corpus_block}

---

**Updated Preliminary Overview**

{s.get('prelim_report', '')}"""

        st.session_state.messages.append({"role": "assistant", "content": preview_msg})
        st.session_state.stage = "waiting_for_decision"
        st.rerun()

    # ── Stage 5: run full analysis ────────────────────────────────────────────
    elif st.session_state.stage == "running_analysis":
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        try:
            with st.status("Running full analysis...", expanded=True) as analysis_status:
                analysis_status.write("🧬 Running topic modeling (UMAP + HDBSCAN)...")

                for chunk in app.stream(None, config=config):
                    node_name = list(chunk.keys())[0]
                    node_output = chunk[node_name]

                    if node_name == "subset_modeler":
                        n_clusters = len(node_output.get("clusters") or {})
                        analysis_status.write(
                            f"🗂️ Found **{n_clusters}** topic clusters — extracting findings..."
                        )

                    elif node_name == "extraction":
                        cf = node_output.get("cluster_findings") or {}
                        total = sum(len(v) for v in cf.values())
                        analysis_status.write(
                            f"🔎 Extracted **{total}** candidate findings — verifying against source text..."
                        )

                    elif node_name == "verifier":
                        vf = node_output.get("verified_cluster_findings") or {}
                        total = sum(len(v) for v in vf.values())
                        analysis_status.write(
                            f"✅ **{total}** findings passed verification — writing report..."
                        )

                    elif node_name == "report_writer":
                        analysis_status.write("📄 Report complete.")

                analysis_status.update(label="Analysis complete.", state="complete")

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
            st.session_state.messages.append(
                {
                    "role": "docx",
                    "content": report_docx,
                    "query": st.session_state.current_state.get("enriched_query", "report"),
                }
            )

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
