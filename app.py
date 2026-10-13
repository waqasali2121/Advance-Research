
import streamlit as st
import pandas as pd

from crew import run_research

st.set_page_config(
    page_title="Evidence Research Studio",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(140deg,#f7f8ff,#f6fbff);
    color: #172033;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}
.hero {
    padding: 2.4rem;
    border-radius: 26px;
    color: white;
    background: linear-gradient(120deg,#171b35,#393477,#635bff);
    box-shadow: 0 16px 40px #26245b20;
}
.hero h1 {
    font-family: 'Manrope',sans-serif;
    color: white;
    font-size: clamp(2.1rem,5vw,3.5rem);
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1.1;
}
.hero p { color: #e2e5ff; font-size: 1.02rem; }
.section-title {
    font-family: 'Manrope',sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -.5px;
}
.agent-card {
    background: white;
    border: 1px solid #e5e9f3;
    border-radius: 15px;
    padding: 13px;
    margin-bottom: 9px;
}
.agent-active {
    border: 1px solid #746bff;
    background: #efefff;
    box-shadow: 0 0 0 3px #635bff12;
}
.stButton > button {
    border-radius: 12px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

AGENTS = [
    ("🧭", "Research Planner", "Plans the search"),
    ("🔎", "Literature Searcher", "Finds scholarly records"),
    ("🧪", "Evidence Appraiser", "Reviews findings and limitations"),
    ("📚", "Citation Verifier", "Checks DOI metadata"),
    ("✍️", "Literature Synthesizer", "Writes the review"),
    ("🛡️", "Security & Quality Auditor", "Audits the final output"),
]

if "result" not in st.session_state:
    st.session_state.result = None

st.markdown("""
<div class="hero">
    <div style="letter-spacing:2px;font-size:.75rem;font-weight:700">
        MULTI-AGENT RESEARCH WORKSPACE
    </div>
    <h1>Evidence Research<br>Studio</h1>
    <p>Turn a research question into an evidence-grounded
    literature synthesis, with citation checks and security auditing.</p>
    <span>CREWAI AGENTS &nbsp; · &nbsp; GROQ GPT-OSS 120B</span>
</div>
""", unsafe_allow_html=True)

st.write("")

with st.sidebar:
    st.markdown("## Research settings")
    years = st.selectbox(
        "Publication period",
        ["2018–2026", "2020–2026", "2015–2026",
         "Any available year"],
    )
    max_papers = st.slider(
        "Maximum candidate papers", 3, 10, 5
    )

    st.divider()
    st.markdown("### Security safeguards")
    st.caption(
        "Retrieved content is untrusted data. "
        "Citation metadata is checked separately. "
        "No prompt-injection defense is perfect."
    )

left, right = st.columns([1.4, 1], gap="large")

with left:
    st.markdown(
        '<div class="section-title">What are you researching?</div>',
        unsafe_allow_html=True,
    )
    question = st.text_area(
        "Research question",
        placeholder=(
            "Example: How effective are AI-assisted "
            "interventions in rural healthcare?"
        ),
        height=150,
    )
    launch = st.button(
        "🚀 Launch research team",
        type="primary",
        use_container_width=True,
    )

with right:
    st.markdown(
        '<div class="section-title">Your agent team</div>',
        unsafe_allow_html=True,
    )
    status_panel = st.empty()

    def show_agents(active=None, completed=None):
        completed = completed or set()
        html = ""

        for emoji, name, detail in AGENTS:
            if name == active:
                status = "● WORKING NOW"
                cls = "agent-card agent-active"
                color = "#635bff"
            elif name in completed:
                status = "✓ COMPLETE"
                cls = "agent-card"
                color = "#16835d"
            else:
                status = "○ WAITING"
                cls = "agent-card"
                color = "#94a3b8"

            html += f"""
            <div class="{cls}">
              <div style="display:flex;gap:10px;align-items:center">
                <div style="font-size:1.4rem">{emoji}</div>
                <div style="flex:1">
                  <div style="font-weight:750">{name}</div>
                  <div style="font-size:.8rem;color:#64748b">
                    {detail}
                  </div>
                </div>
                <div style="font-size:.65rem;font-weight:800;
                            color:{color};white-space:nowrap">
                  {status}
                </div>
              </div>
            </div>
            """

        status_panel.markdown(html, unsafe_allow_html=True)

    show_agents()

if launch:
    if not question.strip():
        st.warning("Enter a research question first.")
    else:
        completed = set()
        progress = st.progress(0)
        live_status = st.empty()

        def on_stage(index, total, name, state):
            if state == "working":
                live_status.info(
                    f"🔄 Working now: **{name}** "
                    f"— stage {index} of {total}"
                )
                show_agents(name, completed)
                progress.progress(int((index - 1) / total * 100))
            else:
                completed.add(name)
                show_agents(None, completed)
                progress.progress(int(index / total * 100))

        try:
            with st.spinner(
                "Research is running. Please keep this page open."
            ):
                st.session_state.result = run_research(
                    question.strip(),
                    years,
                    max_papers,
                    on_stage,
                )

            progress.progress(100)
            live_status.success("Research workflow completed.")
            show_agents(None, completed)

        except Exception as exc:
            st.error(
                f"Research failed: {type(exc).__name__}: {exc}"
            )
            st.info(
                "Check the Groq API key and the Streamlit Cloud logs."
            )

result = st.session_state.result

if result:
    st.divider()
    st.markdown(
        '<div class="section-title">Research output</div>',
        unsafe_allow_html=True,
    )

    final_tab, log_tab, export_tab = st.tabs([
        "✨ Final synthesis",
        "🧾 Agent work log",
        "⬇️ Export",
    ])

    with final_tab:
        st.markdown(result["final"])

    with log_tab:
        for i, stage in enumerate(result["stages"], start=1):
            with st.expander(
                f"{i}. {stage['agent']}",
                expanded=(i == len(result["stages"])),
            ):
                st.markdown(stage["output"])

    with export_tab:
        st.download_button(
            "Download literature review (.md)",
            data=result["final"],
            file_name="literature_review.md",
            mime="text/markdown",
        )

        csv_data = pd.DataFrame(result["stages"]).to_csv(
            index=False
        )

        st.download_button(
            "Download agent work log (.csv)",
            data=csv_data,
            file_name="agent_worklog.csv",
            mime="text/csv",
        )

st.write("")
st.caption(
    "Evidence Research Studio · Always review sources and citations "
    "before using research in academic or professional work."
)

