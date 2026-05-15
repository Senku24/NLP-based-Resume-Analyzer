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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Keyframe Animations ─────────────────────────────── */
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseBorder {
  0%, 100% { box-shadow: 0 0 0 0 rgba(88,166,255,0.0); }
  50%       { box-shadow: 0 0 0 6px rgba(88,166,255,0.18); }
}
@keyframes badgePop {
  0%   { transform: scale(0.85); opacity: 0; }
  70%  { transform: scale(1.05); }
  100% { transform: scale(1);    opacity: 1; }
}

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13192a 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}

/* ── Analyze Button ──────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 12px 0 !important;
    transition: filter 0.2s, transform 0.15s !important;
    animation: pulseBorder 2.8s ease-in-out infinite;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
    filter: brightness(1.18) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:active {
    transform: translateY(0px) scale(0.97) !important;
    filter: brightness(0.92) !important;
}

/* ── Tab Navigation ──────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #161b22;
    border-radius: 12px;
    padding: 6px;
    border: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    color: #8b949e;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    border-radius: 8px;
    padding: 8px 20px;
    transition: color 0.2s, background 0.2s;
    border: none;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #c9d1d9;
    background: #21262d;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(135deg, #1f6feb 0%, #a371f7 100%) !important;
    box-shadow: 0 2px 12px rgba(88,166,255,0.35) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* Cards */
.metric-card {
    background: linear-gradient(145deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 22px 24px;
    text-align: center;
    margin: 8px 0;
    animation: fadeSlideUp 0.45s ease both;
    transition: border-color 0.25s, box-shadow 0.25s;
}
.metric-card:hover {
    border-color: #388bfd44;
    box-shadow: 0 4px 20px rgba(56,139,253,0.12);
}
.metric-card h1 { font-size: 2.8rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
.metric-card p  { color: #8b949e; font-size: 0.82rem; margin: 6px 0 0; text-transform: uppercase; letter-spacing: 0.06em; }

/* ATS Alert Strip */
.alert-strip {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 10px 18px;
    margin: 6px 0 16px;
    animation: fadeSlideUp 0.4s ease both;
}
.alert-strip .alert-emoji { font-size: 1.25rem; flex-shrink: 0; }
.alert-strip .alert-band  { font-weight: 700; font-size: 0.95rem; flex-shrink: 0; }
.alert-strip .alert-text  { color: #8b949e; font-size: 0.85rem; line-height: 1.4; }

/* Skill badges */
.badge {
    display: inline-block;
    padding: 5px 13px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px;
    cursor: default;
    transition: transform 0.18s, box-shadow 0.18s;
    animation: badgePop 0.35s ease both;
}
.badge:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
.badge-matched { background: #122d1e; color: #3fb950; border: 1px solid #2ea043; }
.badge-missing  { background: #2d1216; color: #f85149; border: 1px solid #da3633; }
.badge-extra    { background: #0d1f33; color: #58a6ff; border: 1px solid #1f6feb; }

/* Suggestion cards */
.tip-card {
    background: linear-gradient(135deg, #161b22, #1a2030);
    border-left: 4px solid #58a6ff;
    border-radius: 0 10px 10px 0;
    padding: 11px 16px;
    margin: 7px 0;
    font-size: 0.88rem;
    line-height: 1.6;
    animation: fadeSlideUp 0.4s ease both;
    transition: border-left-width 0.18s;
}
.tip-card:hover { border-left-width: 6px; }
.tip-card.skill { border-left-color: #f85149; }
.tip-card.general { border-left-color: #3fb950; }

/* Section headings */
.section-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 22px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
}

/* Header gradient text */
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #58a6ff 0%, #3fb950 50%, #a371f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.02em;
    line-height: 1.15;
}
.hero-sub {
    color: #8b949e;
    font-size: 0.95rem;
    margin-top: 6px;
    letter-spacing: 0.01em;
}
hr.divider { border-color: #21262d; margin: 16px 0; }

/* Spinner override */
[data-testid="stSpinner"] > div {
    border-color: #388bfd transparent transparent transparent !important;
}

/* File analyzed banner */
.file-banner {
    display:flex; align-items:center; gap:10px;
    background:linear-gradient(90deg,#0d1f33,#13192a);
    border:1px solid #1f6feb44; border-radius:10px;
    padding:10px 18px; margin-bottom:20px;
    animation: fadeSlideUp 0.3s ease both;
}
.file-banner .fb-icon { font-size:1.3rem; }
.file-banner .fb-name { font-weight:700; font-size:1rem; color:#e6edf3; }
.file-banner .fb-label { font-size:0.78rem; color:#58a6ff; text-transform:uppercase; letter-spacing:0.08em; margin-left:auto; }

/* Stat cards */
.stat-card {
    background:linear-gradient(145deg,#161b22,#1c2333);
    border:1px solid #30363d; border-radius:14px;
    padding:20px 16px 16px; text-align:center;
    animation: fadeSlideUp 0.5s ease both;
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    position:relative; overflow:hidden;
}
.stat-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:14px 14px 0 0; }
.stat-card:hover { transform:translateY(-4px); box-shadow:0 8px 28px rgba(0,0,0,0.4); }
.stat-card .sc-num { font-size:2.8rem; font-weight:800; letter-spacing:-0.02em; line-height:1; }
.stat-card .sc-label { font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.09em; color:#8b949e; margin-top:8px; }
.stat-card .sc-sub { font-size:0.72rem; color:#484f58; margin-top:3px; }
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
        title={"text": title, "font": {"color": "#8b949e", "size": 13, "family": "Inter"}},
        number={"suffix": "%", "font": {"color": color, "size": 42, "family": "Inter"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "#30363d",
                "tickfont": {"color": "#484f58", "size": 10},
                "tickwidth": 1,
                "nticks": 6,
            },
            # Thicker bar so score arc pops — no threshold line (removes T-artifact)
            "bar": {"color": color, "thickness": 0.38},
            "bgcolor": "#0a0d12",      # Very dark track so arc pops
            "borderwidth": 0,
            "steps": [
                {"range": [0,   100], "color": "#0e1118"},  # Single dark track
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font={"color": "#e6edf3", "family": "Inter"},
        height=270,
        margin=dict(t=30, b=0, l=30, r=30),
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

        with st.spinner("⚙️  Parsing resume · extracting skills · computing similarity…"):
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "  📊  Overview  ",
        "  🛠️  Skills Analysis  ",
        "  🔬  NLP Details  ",
        "  💡  Suggestions  ",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        # File analyzed banner
        st.markdown(
            f'<div class="file-banner">'
            f'<span class="fb-icon">📄</span>'
            f'<span class="fb-name">{R["filename"]}</span>'
            f'<span class="fb-label">✔ Analyzed</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Score gauges
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_match = make_gauge(R["match_pct"], "Resume Match Score", "#58a6ff")
            st.plotly_chart(fig_match, use_container_width=True)
        with col_g2:
            fig_ats = make_gauge(R["ats_score"], "ATS Compatibility Score", ats_band["color"])
            st.plotly_chart(fig_ats, use_container_width=True)

        # ATS alert strip — slim, not a big block
        st.markdown(
            f'<div class="alert-strip" style="border-left:4px solid {ats_band["color"]}">'
            f'<span class="alert-emoji">{ats_band["emoji"]}</span>'
            f'<span class="alert-band" style="color:{ats_band["color"]}">{ats_band["band"]}</span>'
            f'<span class="alert-text">{ats_band["advice"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-title">📈 At a Glance</p>',
            unsafe_allow_html=True,
        )

        # Quick stats row
        c1, c2, c3, c4 = st.columns(4)
        stats = [
            (c1, R["match_pct"], "%", "Match Score", "TF-IDF similarity", "#58a6ff"),
            (c2, R["ats_score"], "/100", "ATS Score", "Weighted compatibility", ats_band["color"]),
            (c3, len(skill_gap["matched"]), "", "Skills Matched", "Found in both resume & JD", "#3fb950"),
            (c4, len(skill_gap["missing"]), "", "Skills Missing", "Present in JD, absent in resume", "#f85149"),
        ]
        for col, num, suffix, label, sub, color in stats:
            with col:
                st.markdown(
                    f'<div class="stat-card" style="border-color:{color}33">'
                    f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
                    f'background:{color};border-radius:14px 14px 0 0;"></div>'
                    f'<div class="sc-num" style="color:{color}">{num}{suffix}</div>'
                    f'<div class="sc-label">{label}</div>'
                    f'<div class="sc-sub">{sub}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

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
        pc = priority_color.get(p, "#58a6ff")
        st.markdown(
            f'<div class="alert-strip" style="border-left:4px solid {pc}">'
            f'<span class="alert-band" style="color:{pc}">⚡ {p} Priority</span>'
            f'<span class="alert-text">{suggestions["score_message"]}</span>'
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
