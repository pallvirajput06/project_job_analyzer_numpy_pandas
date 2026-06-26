"""
Skill Signal — Data Analyst Job Market Explorer
A Streamlit app that scans job posting descriptions for in-demand skills
and visualizes the results.
"""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Skill Signal · Job Market Explorer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom CSS — dark "signal" theme with a teal/amber accent pair
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-deep:      #0b0f14;
    --bg-panel:     #11161d;
    --bg-panel-2:   #161c25;
    --line:         #232b36;
    --teal:         #2dd4bf;
    --teal-dim:     #134e4a;
    --amber:        #f5a524;
    --amber-dim:    #4a3414;
    --text-hi:      #eef2f6;
    --text-mid:     #9aa7b5;
    --text-low:     #5b6776;
}

html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(45,212,191,0.07), transparent 40%),
        radial-gradient(circle at 85% 15%, rgba(245,165,36,0.05), transparent 35%),
        var(--bg-deep);
    color: var(--text-hi);
}

section[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
    color: var(--text-mid);
}

/* Headline block */
.app-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.6rem 1.8rem;
    border: 1px solid var(--line);
    border-radius: 14px;
    background: linear-gradient(135deg, var(--bg-panel) 0%, var(--bg-panel-2) 100%);
    margin-bottom: 1.4rem;
}
.app-hero h1 {
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0;
    color: var(--text-hi);
    letter-spacing: -0.02em;
}
.app-hero .sub {
    color: var(--text-mid);
    font-size: 0.92rem;
    margin-top: 0.25rem;
}
.app-hero .pulse-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 0 0 rgba(45,212,191,0.7);
    animation: pulse 2s infinite;
    display: inline-block;
    margin-right: 8px;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(45,212,191,0.55); }
    70%  { box-shadow: 0 0 0 8px rgba(45,212,191,0); }
    100% { box-shadow: 0 0 0 0 rgba(45,212,191,0); }
}
.live-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--teal);
    border: 1px solid var(--teal-dim);
    background: rgba(45,212,191,0.07);
    padding: 5px 12px;
    border-radius: 100px;
    white-space: nowrap;
}

/* KPI cards */
.kpi-card {
    border: 1px solid var(--line);
    background: var(--bg-panel);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    height: 100%;
}
.kpi-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-low);
    font-weight: 600;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-hi);
    margin-top: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-foot {
    font-size: 0.78rem;
    color: var(--teal);
    margin-top: 0.3rem;
}
.kpi-foot.warn { color: var(--amber); }

/* Section labels */
.section-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--amber);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1.6rem 0 0.4rem 0;
    border-bottom: 1px dashed var(--line);
    padding-bottom: 0.5rem;
}

/* Skill chip leaderboard rows */
.rank-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.55rem 0.2rem;
    border-bottom: 1px solid var(--line);
}
.rank-num {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-low);
    width: 22px;
    font-size: 0.85rem;
}
.rank-name {
    width: 130px;
    font-weight: 600;
    color: var(--text-hi);
    font-size: 0.92rem;
}
.rank-bar-bg {
    flex: 1;
    background: var(--bg-panel-2);
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
}
.rank-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, var(--teal-dim), var(--teal));
}
.rank-pct {
    width: 56px;
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-mid);
}

/* Dataframe + misc */
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 10px;
}
hr { border-color: var(--line); }

.footnote {
    color: var(--text-low);
    font-size: 0.78rem;
    margin-top: 2rem;
    text-align: center;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Default skill taxonomy (editable in the sidebar)
# --------------------------------------------------------------------------
DEFAULT_SKILLS = [
    "Python", "SQL", "Excel", "Power BI", "Tableau",
    "Pandas", "NumPy", "R", "Statistics", "Machine Learning",
]


@st.cache_data
def load_csv(file_or_path):
    return pd.read_csv(file_or_path)


@st.cache_data
def compute_skill_counts(df: pd.DataFrame, desc_col: str, skills: list[str]) -> pd.DataFrame:
    """Count how many job descriptions mention each skill (case-insensitive, whole-word)."""
    desc = df[desc_col].astype(str)
    counts = {}
    for skill in skills:
        # Word-boundary match so short tokens like "R" don't match inside
        # words like "are" / "required", but multi-word skills (e.g. "Power BI")
        # still match correctly.
        pattern = r"(?<![A-Za-z0-9])" + re.escape(skill) + r"(?![A-Za-z0-9])"
        counts[skill] = int(desc.str.contains(pattern, case=False, na=False, regex=True).sum())
    out = pd.DataFrame(list(counts.items()), columns=["Skill", "Count"])
    total = len(df)
    out["Share"] = (out["Count"] / total * 100).round(1) if total else 0
    return out.sort_values("Count", ascending=False).reset_index(drop=True)


def kpi_card(label, value, foot=None, warn=False):
    foot_html = f'<div class="kpi-foot {"warn" if warn else ""}">{foot}</div>' if foot else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {foot_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Sidebar — data source & controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📡 Skill Signal")
    st.caption("Scan job postings for the skills that actually get hired for.")

    st.markdown("#### Data source")
    use_sample = st.toggle("Use bundled sample dataset", value=True)

    uploaded_file = None
    if not use_sample:
        uploaded_file = st.file_uploader("Upload a job postings CSV", type=["csv"])

    st.markdown("#### Skills to track")
    skills_text = st.text_area(
        "One skill per line (edit freely)",
        value="\n".join(DEFAULT_SKILLS),
        height=200,
    )
    custom_skills = [s.strip() for s in skills_text.splitlines() if s.strip()]

    st.markdown("---")
    st.caption("Tip: add niche tools (e.g. `Looker`, `dbt`, `Snowflake`) to see how they stack up.")

# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
df = None
load_error = None

if use_sample:
    try:
        df = load_csv("data.csv")
    except Exception as e:
        load_error = str(e)
elif uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
    except Exception as e:
        load_error = str(e)

# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-hero">
        <div>
            <h1><span class="pulse-dot"></span>Skill Signal</h1>
            <div class="sub">Turning a pile of job descriptions into a ranked list of what to actually learn next.</div>
        </div>
        <div class="live-tag">DATA · ANALYST · MARKET</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.error(f"Couldn't load that file: {load_error}")
    st.stop()

if df is None:
    st.info("⬅️ Upload a CSV in the sidebar, or switch on the bundled sample dataset, to get started.")
    st.stop()

# --------------------------------------------------------------------------
# Column mapping (in case the user's CSV uses different headers)
# --------------------------------------------------------------------------
cols = list(df.columns)


def guess_col(candidates, columns):
    for c in candidates:
        for col in columns:
            if c.lower() == col.lower():
                return col
    return columns[0]


desc_guess = guess_col(["Description", "Job Description", "description"], cols)
title_guess = guess_col(["Job Title", "Title", "Position"], cols)

with st.expander("⚙️ Column mapping (auto-detected — adjust if needed)", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        desc_col = st.selectbox("Job description column", cols, index=cols.index(desc_guess))
    with c2:
        title_col = st.selectbox("Job title column", cols, index=cols.index(title_guess))

if not custom_skills:
    st.warning("Add at least one skill in the sidebar to analyze.")
    st.stop()

skill_df = compute_skill_counts(df, desc_col, custom_skills)

# --------------------------------------------------------------------------
# KPI row
# --------------------------------------------------------------------------
n_jobs = len(df)
top_skill_row = skill_df.iloc[0]
n_missing_desc = df[desc_col].isna().sum()
unique_titles = df[title_col].nunique() if title_col in df.columns else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("Postings analyzed", f"{n_jobs:,}", "across all sources")
with k2:
    kpi_card("Top skill", top_skill_row["Skill"], f"{top_skill_row['Count']:,} mentions · {top_skill_row['Share']}%")
with k3:
    kpi_card("Unique job titles", f"{unique_titles:,}")
with k4:
    kpi_card(
        "Missing descriptions",
        f"{n_missing_desc:,}",
        "rows excluded from scan" if n_missing_desc else "clean dataset",
        warn=bool(n_missing_desc),
    )

# --------------------------------------------------------------------------
# Leaderboard + bar chart
# --------------------------------------------------------------------------
st.markdown('<div class="section-tag">Skill Leaderboard</div>', unsafe_allow_html=True)

left, right = st.columns([1, 1.3], gap="large")

with left:
    max_count = skill_df["Count"].max() or 1
    rows_html = ""
    for i, row in skill_df.iterrows():
        pct_of_max = max(row["Count"] / max_count * 100, 2)
        rows_html += f"""
        <div class="rank-row">
            <div class="rank-num">{i+1:02d}</div>
            <div class="rank-name">{row['Skill']}</div>
            <div class="rank-bar-bg"><div class="rank-bar-fill" style="width:{pct_of_max}%;"></div></div>
            <div class="rank-pct">{row['Count']:,}</div>
        </div>
        """
    st.markdown(rows_html, unsafe_allow_html=True)

with right:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=skill_df["Skill"],
            y=skill_df["Count"],
            marker=dict(
                color=skill_df["Count"],
                colorscale=[[0, "#134e4a"], [1, "#2dd4bf"]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{x}</b><br>%{y} postings<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, sans-serif", color="#eef2f6"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#232b36"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Share-of-postings donut + top job titles
# --------------------------------------------------------------------------
st.markdown('<div class="section-tag">Market Composition</div>', unsafe_allow_html=True)

d1, d2 = st.columns(2, gap="large")

with d1:
    top_n = skill_df.head(6).copy()
    others_count = skill_df["Count"].iloc[6:].sum()
    if others_count > 0:
        top_n = pd.concat(
            [top_n, pd.DataFrame([{"Skill": "Other", "Count": others_count, "Share": np.nan}])],
            ignore_index=True,
        )
    donut = go.Figure(
        data=[
            go.Pie(
                labels=top_n["Skill"],
                values=top_n["Count"],
                hole=0.62,
                marker=dict(colors=px.colors.sequential.Teal[::-1] + ["#3a4250"]),
                textfont=dict(color="#eef2f6"),
            )
        ]
    )
    donut.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, sans-serif", color="#eef2f6"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=380,
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        annotations=[dict(text=f"{n_jobs}<br>jobs", x=0.5, y=0.5, showarrow=False, font_size=16)],
    )
    st.plotly_chart(donut, use_container_width=True)

with d2:
    if title_col in df.columns:
        top_titles = df[title_col].value_counts().head(8).reset_index()
        top_titles.columns = ["Job Title", "Postings"]
        title_fig = go.Figure(
            go.Bar(
                x=top_titles["Postings"],
                y=top_titles["Job Title"],
                orientation="h",
                marker=dict(color="#f5a524"),
                hovertemplate="<b>%{y}</b><br>%{x} postings<extra></extra>",
            )
        )
        title_fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk, sans-serif", color="#eef2f6"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=380,
            yaxis=dict(autorange="reversed", showgrid=False),
            xaxis=dict(showgrid=True, gridcolor="#232b36"),
        )
        st.plotly_chart(title_fig, use_container_width=True)
    else:
        st.info("No job title column detected for this view.")

# --------------------------------------------------------------------------
# Explorer table
# --------------------------------------------------------------------------
st.markdown('<div class="section-tag">Posting Explorer</div>', unsafe_allow_html=True)

filter_skill = st.selectbox(
    "Filter postings that mention…",
    options=["(no filter)"] + list(skill_df["Skill"]),
)

view_df = df.copy()
if filter_skill != "(no filter)":
    pattern = r"(?<![A-Za-z0-9])" + re.escape(filter_skill) + r"(?![A-Za-z0-9])"
    mask = view_df[desc_col].astype(str).str.contains(pattern, case=False, na=False, regex=True)
    view_df = view_df[mask]

st.dataframe(view_df, use_container_width=True, height=320)

st.download_button(
    "⬇️ Download skill counts as CSV",
    data=skill_df.to_csv(index=False).encode("utf-8"),
    file_name="skill_counts.csv",
    mime="text/csv",
)

st.markdown(
    '<div class="footnote">Built with Streamlit · Skill mentions are detected via case-insensitive, whole-word matching on the description column.</div>',
    unsafe_allow_html=True,
)