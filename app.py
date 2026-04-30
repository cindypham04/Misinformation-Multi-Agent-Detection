"""Streamlit UI for the Misinformation Multi-Agent Detection System."""

from __future__ import annotations

import streamlit as st

from misinfo_detection.cli import run_claim
from misinfo_detection.schemas import Evidence


def get_verdict_style(verdict: str | None) -> tuple[str, str, str]:
    """Return (background_color, text_color, icon) for verdict."""
    if verdict == "supported":
        return "#d4edda", "#155724", "✓"
    elif verdict == "refuted":
        return "#f8d7da", "#721c24", "✗"
    else:
        return "#fff3cd", "#856404", "?"


def render_evidence_item(ev: Evidence, idx: int) -> None:
    """Render a single evidence item."""
    score_pct = int(ev.get("score", 0) * 100)
    st.markdown(f"**{idx}. [{ev.get('title', 'Untitled')}]({ev.get('url', '#')})**")
    st.caption(f"Source: {ev.get('source', 'Unknown')} | Relevance: {score_pct}%")
    with st.expander("View content"):
        st.write(ev.get("content", "No content available."))


def render_evidence_pool(evidence_pool: dict[str, list[Evidence]], title: str = "Evidence Pool") -> None:
    """Render grouped evidence by query."""
    if not evidence_pool:
        st.info("No evidence collected.")
        return

    total_items = sum(len(items) for items in evidence_pool.values())
    st.markdown(f"**{total_items} evidence items** from {len(evidence_pool)} queries")

    for query, items in evidence_pool.items():
        with st.expander(f'Query: "{query}" ({len(items)} results)'):
            for i, ev in enumerate(items, 1):
                render_evidence_item(ev, i)
                if i < len(items):
                    st.divider()


def render_debate_trace(debate_log: list[str]) -> None:
    """Render the debate log with role styling."""
    if not debate_log:
        st.info("No debate turns recorded.")
        return

    st.markdown(f"**{len(debate_log)} debate turns**")

    for i, turn in enumerate(debate_log, 1):
        if turn.startswith("[negative]"):
            role = "Negative"
            content = turn.replace("[negative]", "").strip()
            color = "#ffcccc"
        elif turn.startswith("[affirmative]"):
            role = "Affirmative"
            content = turn.replace("[affirmative]", "").strip()
            color = "#ccffcc"
        else:
            role = "Unknown"
            content = turn
            color = "#f0f0f0"

        st.markdown(
            f"""<div style="background-color: {color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <strong>{role} (Turn {i})</strong><br>{content}
            </div>""",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Misinformation Detector",
        page_icon="🔍",
        layout="wide",
    )

    st.title("Misinformation Multi-Agent Detection")
    st.markdown(
        "Enter a claim below to analyze it using a multi-agent debate system. "
        "The system will gather evidence, conduct a structured debate, and provide a verdict."
    )

    st.divider()

    claim = st.text_area(
        "Enter a claim to analyze:",
        placeholder="e.g., The Earth is flat.",
        height=100,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("Analyze Claim", type="primary", use_container_width=True)

    if analyze_button and claim.strip():
        with st.spinner("Analyzing claim... This may take a minute as agents debate and gather evidence."):
            try:
                result = run_claim(claim.strip())
                st.session_state["result"] = result
                st.session_state["analyzed_claim"] = claim.strip()
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                return

    if "result" in st.session_state:
        result = st.session_state["result"]
        analyzed_claim = st.session_state.get("analyzed_claim", "")

        st.divider()

        verdict = result.get("final_verdict")
        bg_color, text_color, icon = get_verdict_style(verdict)

        st.markdown(
            f"""<div style="background-color: {bg_color}; color: {text_color}; 
            padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h2 style="margin: 0; color: {text_color};">{icon} Verdict: {(verdict or 'Unknown').upper()}</h2>
            <p style="margin: 5px 0 0 0; color: {text_color};">Claim: {analyzed_claim}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        with st.expander("Final Report", expanded=True):
            report = result.get("final_report")
            if report:
                st.markdown(report)
            else:
                st.info("No final report available.")

        st.subheader("Advisor")
        advisor_tab1, advisor_tab2 = st.tabs(["Analysis", "Advice"])

        with advisor_tab1:
            analysis = result.get("advisor_analysis")
            if analysis:
                st.markdown(analysis)
            else:
                st.info("No advisor analysis available.")

        with advisor_tab2:
            advice = result.get("advisor_advice")
            if advice:
                st.markdown(advice)
            else:
                st.info("No advisor advice available.")

        st.subheader("Evidence Pool")
        evidence_tabs = st.tabs(["Debate Evidence", "Verifier Evidence"])

        with evidence_tabs[0]:
            render_evidence_pool(result.get("evidence_pool", {}))

        with evidence_tabs[1]:
            render_evidence_pool(result.get("verifier_evidence", {}), "Verifier Evidence")

        st.subheader("Debate Trace")
        render_debate_trace(result.get("debate_log", []))

        st.divider()
        if st.button("Clear Results"):
            del st.session_state["result"]
            if "analyzed_claim" in st.session_state:
                del st.session_state["analyzed_claim"]
            st.rerun()


if __name__ == "__main__":
    main()
