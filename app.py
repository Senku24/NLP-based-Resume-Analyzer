"""
app.py — AI Resume Analyzer & Job Match System
Main Streamlit application entry point.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from utils.parser import extract_resume_text
from utils.preprocess import preprocess_text, get_top_keywords
from utils.skill_extractor import load_skills, extract_skills, get_skill_gap
from utils.similarity import compute_tfidf_similarity, get_match_percentage
from utils.ats_score import compute_ats_score, get_ats_band
from utils.suggestions import generate_suggestions

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 8px 0;
}
.metric-card h1 { font-size: 3rem; font-weight: 700; margin: 0; }
.metric-card p  { color: #8b949e; font-size: 0.85rem; margin: 4px 0 0; }

/* Skill badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 3px;
}
.badge-matched { background: #1a3a2a; color: #3fb950; border: 1px solid #3fb950; }
.badge-missing  { background: #3a1a1a; color: #f85149; border: 1px solid #f85149; }
.badge-extra    { background: #1a2a3a; color: #58a6ff; border: 1px solid #58a6ff; }

/* Suggestion cards */
.tip-card {
    background: #161b22;
    border-left: 4px solid #58a6ff;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
}
.tip-card.skill { border-left-color: #f85149; }
.tip-card.general { border-left-color: #3fb950; }

/* Section headings */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #58a6ff;
    margin: 20px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
}

/* Tab styling override */
.stTabs [data-baseweb="tab"] {
    color: #8b949e;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
}

/* Header gradient text */
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #3fb950, #a371f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    color: #8b949e;
    font-size: 1rem;
    margin-top: 4px;
}
hr.divider { border-color: #21262d; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_load_skills():
    return load_skills()


def render_badges(skills: list, badge_class: str) -> str:
    return " ".join(f'<span class="badge {badge_class}">{s}</span>' for s in skills)


def make_gauge(value: int, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#e6edf3", "size": 14}},
        number={"suffix": "%", "font": {"color": color, "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e",
                     "tickfont": {"color": "#8b949e"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#21262d",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, 40],  "color": "#1a1a2a"},
                {"range": [40, 70], "color": "#1a2a1a"},
                {"range": [70, 100],"color": "#1a3a2a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font={"color": "#e6edf3"},
        height=260,
        margin=dict(t=40, b=10, l=20, r=20),
    )
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="hero-title" style="font-size:1.4rem;">🎯 Resume Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">NLP-Powered Job Match System</p>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("### 📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "Supported: PDF, DOCX, TXT",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )

    st.markdown("### 📋 Job Description")
    jd_input = st.text_area(
        "Paste the job description here",
        height=260,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze Resume", use_container_width=True, type="primary")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#484f58;font-size:0.75rem;text-align:center;">'
        'AI Resume Analyzer v1.0<br>NLP-based Prototype MVP</p>',
        unsafe_allow_html=True,
    )


# ─── Main Area ────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">AI Resume Analyzer & Job Match System</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload your resume, paste a job description, and get instant NLP-powered insights.</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Instructions when nothing uploaded yet
if not uploaded_file or not jd_input.strip():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h1>📄</h1>
            <h3 style="color:#58a6ff;margin:8px 0;">Step 1</h3>
            <p>Upload your resume<br>(PDF, DOCX, or TXT)</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h1>📋</h1>
            <h3 style="color:#3fb950;margin:8px 0;">Step 2</h3>
            <p>Paste the job description<br>in the sidebar</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h1>🔍</h1>
            <h3 style="color:#a371f7;margin:8px 0;">Step 3</h3>
            <p>Click Analyze Resume<br>to see your results</p>
        </div>""", unsafe_allow_html=True)
    st.stop()


# ─── Analysis Pipeline ────────────────────────────────────────────────────────
if analyze_btn or ("results" in st.session_state):

    if analyze_btn:
        if not uploaded_file:
            st.error("⚠️ Please upload a resume file.")
            st.stop()
        if not jd_input.strip():
            st.error("⚠️ Please paste a job description.")
            st.stop()

        with st.spinner("⚙️ Running NLP pipeline…"):
            try:
                # 1. Parse resume
                resume_text = extract_resume_text(uploaded_file)
                if not resume_text.strip():
                    st.error("Could not extract text from the uploaded file. Try a different format.")
                    st.stop()

                # 2. Preprocess
                resume_nlp = preprocess_text(resume_text)
                jd_nlp     = preprocess_text(jd_input)

                # 3. Skill extraction
                skills_df      = cached_load_skills()
                resume_skills  = extract_skills(resume_text, skills_df)
                jd_skills      = extract_skills(jd_input,   skills_df)
                skill_gap      = get_skill_gap(resume_skills, jd_skills)

                # 4. TF-IDF similarity
                tfidf_score  = compute_tfidf_similarity(resume_nlp["processed"], jd_nlp["processed"])
                match_pct    = get_match_percentage(tfidf_score)

                # 5. ATS score
                ats_score = compute_ats_score(tfidf_score, skill_gap["matched"], skill_gap["missing"], resume_text)
                ats_band  = get_ats_band(ats_score)

                # 6. Suggestions
                suggestions = generate_suggestions(skill_gap["missing"], match_pct, resume_text)

                # 7. Top keywords
                resume_keywords = get_top_keywords(resume_nlp["lemmas"], 15)
                jd_keywords     = get_top_keywords(jd_nlp["lemmas"], 15)

                # Cache results
                st.session_state["results"] = {
                    "resume_text": resume_text,
                    "resume_nlp": resume_nlp,
                    "jd_nlp": jd_nlp,
                    "resume_skills": resume_skills,
                    "jd_skills": jd_skills,
                    "skill_gap": skill_gap,
                    "tfidf_score": tfidf_score,
                    "match_pct": match_pct,
                    "ats_score": ats_score,
                    "ats_band": ats_band,
                    "suggestions": suggestions,
                    "resume_keywords": resume_keywords,
                    "jd_keywords": jd_keywords,
                    "filename": uploaded_file.name,
                }
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.stop()

    # ── Load cached results ───────────────────────────────────────────────────
    R = st.session_state.get("results")
    if not R:
        st.info("Upload a resume and click **Analyze Resume** to begin.")
        st.stop()

    skill_gap   = R["skill_gap"]
    ats_band    = R["ats_band"]
    suggestions = R["suggestions"]

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🛠️ Skills Analysis", "🔬 NLP Details", "💡 Suggestions"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown(f"**Analyzed:** `{R['filename']}`")

        # Score gauges
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_match = make_gauge(R["match_pct"], "Resume Match Score", "#58a6ff")
            st.plotly_chart(fig_match, use_container_width=True)
        with col_g2:
            fig_ats = make_gauge(R["ats_score"], "ATS Compatibility Score", ats_band["color"])
            st.plotly_chart(fig_ats, use_container_width=True)

        # ATS band summary
        st.markdown(
            f'<div class="metric-card" style="text-align:left;">'
            f'<span style="font-size:1.8rem;">{ats_band["emoji"]}</span>&nbsp;'
            f'<strong style="color:{ats_band["color"]};font-size:1.1rem;">{ats_band["band"]}</strong>'
            f'<p style="margin-top:8px;">{ats_band["advice"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Quick stats row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><h1 style="color:#58a6ff;">{R["match_pct"]}%</h1><p>Match Score</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><h1 style="color:{ats_band["color"]};">{R["ats_score"]}</h1><p>ATS Score /100</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h1 style="color:#3fb950;">{len(skill_gap["matched"])}</h1><p>Matched Skills</p></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><h1 style="color:#f85149;">{len(skill_gap["missing"])}</h1><p>Missing Skills</p></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — SKILLS ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        # Donut chart
        matched_n = len(skill_gap["matched"])
        missing_n = len(skill_gap["missing"])
        extra_n   = len(skill_gap["extra"])

        if matched_n + missing_n + extra_n > 0:
            fig_donut = go.Figure(go.Pie(
                labels=["Matched", "Missing", "Additional"],
                values=[matched_n, missing_n, extra_n],
                hole=0.6,
                marker_colors=["#3fb950", "#f85149", "#58a6ff"],
                textinfo="label+percent",
                textfont={"color": "#e6edf3"},
            ))
            fig_donut.update_layout(
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                font={"color": "#e6edf3"},
                showlegend=False,
                height=300,
                margin=dict(t=20, b=20, l=20, r=20),
                annotations=[{"text": f"{matched_n}/{matched_n+missing_n}", "font_size": 22,
                               "showarrow": False, "font": {"color": "#e6edf3"}}],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Bar chart — top 10 JD skills
        if skill_gap["matched"] or skill_gap["missing"]:
            all_skills = (
                [(s, "Matched") for s in skill_gap["matched"][:10]] +
                [(s, "Missing") for s in skill_gap["missing"][:10]]
            )
            df_bar = pd.DataFrame(all_skills, columns=["Skill", "Status"])
            color_map = {"Matched": "#3fb950", "Missing": "#f85149"}
            fig_bar = px.bar(
                df_bar, x="Skill", color="Status",
                color_discrete_map=color_map,
                title="Top JD Skills — Match Status",
            )
            fig_bar.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font={"color": "#e6edf3"}, height=350,
                title_font={"color": "#8b949e"},
                xaxis={"tickangle": -30, "tickfont": {"color": "#8b949e"}},
                yaxis={"tickfont": {"color": "#8b949e"}, "gridcolor": "#21262d"},
                legend={"bgcolor": "#0d1117"},
                margin=dict(t=50, b=80),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        col_m, col_x = st.columns(2)

        with col_m:
            st.markdown('<p class="section-title">✅ Matched Skills</p>', unsafe_allow_html=True)
            if skill_gap["matched"]:
                st.markdown(render_badges(skill_gap["matched"], "badge-matched"), unsafe_allow_html=True)
            else:
                st.info("No matching skills detected.")

        with col_x:
            st.markdown('<p class="section-title">❌ Missing Skills (from JD)</p>', unsafe_allow_html=True)
            if skill_gap["missing"]:
                st.markdown(render_badges(skill_gap["missing"], "badge-missing"), unsafe_allow_html=True)
            else:
                st.success("No missing skills — great coverage!")

        if skill_gap["extra"]:
            st.markdown('<p class="section-title">➕ Additional Resume Skills</p>', unsafe_allow_html=True)
            st.markdown(render_badges(skill_gap["extra"], "badge-extra"), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — NLP DETAILS
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### NLP Pipeline Steps")

        with st.expander("🔡 Step 1 — Lowercased & Cleaned Text", expanded=False):
            st.code(R["resume_nlp"]["cleaned"][:1500] + ("…" if len(R["resume_nlp"]["cleaned"]) > 1500 else ""), language=None)

        with st.expander("🔤 Step 2 — Tokens (after punctuation removal)", expanded=False):
            tokens_preview = R["resume_nlp"]["tokens"][:60]
            st.write(tokens_preview)

        with st.expander("🚫 Step 3 — After Stopword Removal", expanded=False):
            st.write(R["resume_nlp"]["filtered"][:60])

        with st.expander("📝 Step 4 — Lemmatized Tokens", expanded=False):
            st.write(R["resume_nlp"]["lemmas"][:60])

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### 🔑 Top Keywords Comparison")

        col_rk, col_jk = st.columns(2)
        with col_rk:
            st.markdown('<p class="section-title">Resume — Top Keywords</p>', unsafe_allow_html=True)
            if R["resume_keywords"]:
                df_rk = pd.DataFrame(R["resume_keywords"], columns=["Keyword", "Frequency"])
                fig_rk = px.bar(df_rk, x="Frequency", y="Keyword", orientation="h",
                                color="Frequency", color_continuous_scale="Blues")
                fig_rk.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                     font={"color": "#e6edf3"}, height=380,
                                     coloraxis_showscale=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig_rk, use_container_width=True)

        with col_jk:
            st.markdown('<p class="section-title">Job Description — Top Keywords</p>', unsafe_allow_html=True)
            if R["jd_keywords"]:
                df_jk = pd.DataFrame(R["jd_keywords"], columns=["Keyword", "Frequency"])
                fig_jk = px.bar(df_jk, x="Frequency", y="Keyword", orientation="h",
                                color="Frequency", color_continuous_scale="Greens")
                fig_jk.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                     font={"color": "#e6edf3"}, height=380,
                                     coloraxis_showscale=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig_jk, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### 📄 Extracted Resume Text (Raw)")
        with st.expander("Show extracted text", expanded=False):
            st.text(R["resume_text"][:3000] + ("…" if len(R["resume_text"]) > 3000 else ""))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — SUGGESTIONS
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        priority_color = {"High": "#f85149", "Medium": "#ffe66d", "Low": "#3fb950"}
        p = suggestions["priority"]
        st.markdown(
            f'<div class="metric-card" style="text-align:left;">'
            f'<span style="color:{priority_color.get(p,"#58a6ff")};font-weight:700;">'
            f'Priority: {p}</span>'
            f'<p style="margin-top:8px;">{suggestions["score_message"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if suggestions["skill_tips"]:
            st.markdown('<p class="section-title">🛠️ Skill Gap Recommendations</p>', unsafe_allow_html=True)
            for tip in suggestions["skill_tips"]:
                st.markdown(f'<div class="tip-card skill">{tip}</div>', unsafe_allow_html=True)

        if suggestions["general_tips"]:
            st.markdown('<p class="section-title">📌 General Resume Tips</p>', unsafe_allow_html=True)
            for tip in suggestions["general_tips"]:
                st.markdown(f'<div class="tip-card general">{tip}</div>', unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#484f58;font-size:0.8rem;">'
            '💡 These suggestions are generated automatically based on NLP analysis. '
            'Always review with your own judgement.</p>',
            unsafe_allow_html=True,
        )
