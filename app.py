"""Streamlit dashboard for dataset-backed resume screening."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.data import load_jobs, load_matches, load_resumes
from src.job_config import job_from_existing, parse_skill_text, role_suggestions
from src.ranker import _skills, rank_resumes
from src.uploads import process_uploaded_resumes


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(page_title="TalentLens | Resume Screening", page_icon="TL", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink:#17212b; --muted:#65727e; --teal:#087f8c; --mint:#dff4ed; --line:#dfe7e8; --paper:#f5f8f7; }
    .stApp { background:var(--paper); color:var(--ink); }
    [data-testid="stSidebar"] { background:#122c35; }
    [data-testid="stSidebar"] * { color:#edf8f5 !important; }
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] label { color:var(--ink) !important; }
    [data-testid="stMain"] input,
    [data-testid="stMain"] textarea,
    [data-testid="stMain"] [data-baseweb="select"] > div {
        background:#ffffff !important;
        color:var(--ink) !important;
        border-color:#b9c8cc !important;
    }
    [data-testid="stMain"] input::placeholder,
    [data-testid="stMain"] textarea::placeholder { color:#71808a !important; opacity:1; }
    [data-testid="stMain"] input:focus,
    [data-testid="stMain"] textarea:focus { color:var(--ink) !important; }
    [data-testid="stMain"] [data-baseweb="select"] * { color:var(--ink) !important; }
    h1,h2,h3 { letter-spacing:0; }
    .eyebrow { color:var(--teal); font-size:.76rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .subtitle { color:var(--muted); font-size:1.05rem; margin-bottom:1.4rem; }
    .metric-card { background:white; border:1px solid var(--line); border-radius:8px; padding:1rem; min-height:100px; }
    .metric-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; }
    .metric-value { color:var(--ink); font-size:1.8rem; font-weight:700; margin-top:.3rem; }
    .tag { background:var(--mint); border-radius:999px; color:#176252; display:inline-block; font-size:.75rem; margin:.2rem .25rem .1rem 0; padding:.25rem .5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_resumes(), load_jobs(), load_matches()


@st.cache_data(show_spinner=False)
def get_metrics() -> dict[str, dict[str, float]]:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        raise FileNotFoundError("models/metrics.json is missing. Run python -m src.evaluate first.")
    return pd.read_json(path).to_dict(orient="index")


def format_skills(value: Any) -> str:
    values = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(values, str):
        values = [values]
    return ", ".join(str(item) for item in values) if values else "None"


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def dashboard_page(resumes: pd.DataFrame, jobs: pd.DataFrame, matches: pd.DataFrame) -> None:
    st.markdown('<div class="eyebrow">TalentLens overview</div>', unsafe_allow_html=True)
    st.title("Intelligent Resume Screening")
    st.markdown(
        '<div class="subtitle">A dataset-backed demonstration of resume preprocessing, candidate ranking, and model evaluation.</div>',
        unsafe_allow_html=True,
    )
    one, two, three, four = st.columns(4)
    with one:
        metric_card("Jobs in dataset", f"{len(jobs):,}")
    with two:
        metric_card("Resumes in dataset", f"{len(resumes):,}")
    with three:
        metric_card("Known job matches", f"{len(matches):,}")
    with four:
        metric_card("Job industries", str(jobs["industry"].nunique()))
    st.markdown("### How to use the demonstration")
    st.write(
        "Use **Screen Candidates** to select a real dataset job or create a custom role, upload PDF/DOCX resumes, "
        "review extraction results, rank candidates, and export a shortlist. Use **Model Evaluation** to inspect "
        "saved held-out metrics for the synthetic dataset."
    )
    st.info("Uploaded resumes receive genuine model scores, but they do not have dataset ground-truth labels.")
    st.markdown("### Dataset snapshot")
    left, right = st.columns(2)
    with left:
        st.dataframe(jobs["job_title"].value_counts().head(10).rename("Jobs"), use_container_width=True)
    with right:
        st.dataframe(resumes["role"].value_counts().head(10).rename("Resumes"), use_container_width=True)


def _job_configuration(jobs: pd.DataFrame) -> dict[str, Any] | None:
    st.markdown("### 1. Configure the job")
    source = st.radio("Job source", ["Existing dataset job", "Custom job"], horizontal=True)
    if source == "Existing dataset job":
        labels = [f"{row.job_id} | {row.job_title}" for row in jobs.itertuples()]
        selected = st.selectbox("Select an existing job", labels)
        row = jobs.set_index("job_id").loc[selected.split(" | ", 1)[0]]
        base = job_from_existing(row)
        if st.session_state.get("_active_existing_job_id") != base["job_id"]:
            st.session_state[f"existing_title_{base['job_id']}"] = str(base["job_title"])
            st.session_state[f"existing_description_{base['job_id']}"] = str(base["description"])
            st.session_state[f"required_existing_{base['job_id']}"] = ", ".join(base["must_have_skills"])
            st.session_state[f"preferred_existing_{base['job_id']}"] = ", ".join(base["nice_to_have_skills"])
            st.session_state["_active_existing_job_id"] = base["job_id"]
        st.caption(f"Dataset job: {base['job_title']} · {base['seniority']} · {base['industry']}")
        title = st.text_input("Job title", value=str(base["job_title"]), key=f"existing_title_{base['job_id']}")
        description = st.text_area("Job description", value=str(base["description"]), height=120, key=f"existing_description_{base['job_id']}")
        default_required = ", ".join(base["must_have_skills"])
        default_preferred = ", ".join(base["nice_to_have_skills"])
        seniority = base["seniority"]
        industry = base["industry"]
        min_experience = 0.0
        role = title
    else:
        title = st.text_input("Job title", placeholder="e.g. Senior Data Engineer", key="custom_title")
        role = title
        description = st.text_area("Job description", height=120, key="custom_description")
        suggested_required, suggested_preferred = role_suggestions(role, jobs)
        if suggested_required or suggested_preferred:
            st.caption(
                f"Suggestions for {role}: required: {format_skills(suggested_required)}; "
                f"preferred: {format_skills(suggested_preferred)}"
            )
        if st.button("Apply role suggestions", key="apply_role_suggestions"):
            st.session_state["custom_required"] = ", ".join(suggested_required)
            st.session_state["custom_preferred"] = ", ".join(suggested_preferred)
        default_required = st.session_state.get("custom_required", "")
        default_preferred = st.session_state.get("custom_preferred", "")
        seniority = st.selectbox("Seniority", ["Not specified", "Junior", "Mid", "Senior", "Lead", "Principal"])
        industry = st.text_input("Industry (optional)", key="custom_industry")
        min_experience = st.number_input("Minimum relevant experience (years)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
    if source == "Existing dataset job":
        required_key = f"required_existing_{base['job_id']}"
        preferred_key = f"preferred_existing_{base['job_id']}"
    else:
        required_key = "required_custom"
        preferred_key = "preferred_custom"
    required_text = st.text_area("Required skills", value=default_required, key=required_key)
    preferred_text = st.text_area("Preferred skills", value=default_preferred, key=preferred_key)
    required = parse_skill_text(required_text)
    preferred = parse_skill_text(preferred_text)
    if not title.strip():
        st.warning("Enter a job title before continuing.")
        return None
    return {
        "job_title": title.strip(),
        "description": description.strip(),
        "must_have_skills": required,
        "nice_to_have_skills": preferred,
        "seniority": "" if seniority == "Not specified" else seniority,
        "industry": industry.strip(),
        "minimum_experience": min_experience,
    }


def _prepare_upload_results(
    uploaded_files: list[Any],
    job: dict[str, Any],
    method: str,
    top_k: int,
    resumes: pd.DataFrame | None = None,
    errors: list[dict[str, str]] | None = None,
) -> None:
    request = {
        "job": job,
        "method": method,
        "top_k": top_k,
        "files": [{"name": item.name, "hash": hashlib.sha256(item.getvalue()).hexdigest()} for item in uploaded_files],
    }
    signature = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
    with st.spinner("Extracting resumes and calculating candidate scores..."):
        if resumes is None:
            resumes, errors = process_uploaded_resumes(uploaded_files, job)
        errors = list(errors or [])
        ranked = pd.DataFrame()
        if not resumes.empty:
            try:
                ranked = rank_resumes(job, top_k=min(top_k, len(resumes)), method=method, resumes=resumes)
            except (FileNotFoundError, OSError, ValueError) as error:
                errors.append({"filename": "Ranking pipeline", "error": str(error)})
    st.session_state["screening_result"] = {"signature": signature, "ranked": ranked, "errors": errors}


def _show_screening_results(result: dict[str, Any], job: dict[str, Any], method: str) -> None:
    for error in result["errors"]:
        st.error(f"{error['filename']}: {error['error']}")
    ranked = result["ranked"].copy()
    if ranked.empty:
        st.error("No valid resumes remain. Upload a readable PDF or DOCX and try again.")
        return
    required = _skills(job["must_have_skills"])
    preferred = _skills(job["nice_to_have_skills"])
    candidate_skills = ranked["skills"].map(lambda value: set(_skills(value)))
    ranked["matched_required_skills"] = candidate_skills.map(lambda values: [skill for skill in required if skill in values])
    ranked["missing_required_skills"] = candidate_skills.map(lambda values: [skill for skill in required if skill not in values])
    ranked["matched_preferred_skills"] = candidate_skills.map(lambda values: [skill for skill in preferred if skill in values])
    ranked["experience_match"] = ranked["years_experience"] >= float(job.get("minimum_experience", 0))
    st.session_state["screening_ranked_display"] = ranked
    st.markdown("### 3. Ranked candidates")
    st.caption("Scores are ranking scores, not hiring probabilities. Uploaded resumes have no known ground-truth labels.")
    shortlist = st.slider("Shortlist size", 1, len(ranked), min(10, len(ranked)), key="shortlist_size")
    filtered = ranked.head(shortlist).copy()
    min_score = st.slider("Minimum score", 0.0, max(1.0, float(ranked["score"].max())), 0.0, 0.01, key="minimum_score")
    filtered = filtered[filtered["score"] >= min_score]
    table = filtered[
        [
            "filename", "score", "text_similarity", "required_skill_coverage",
            "preferred_skill_coverage", "matched_required_skills",
            "missing_required_skills", "matched_preferred_skills", "years_experience",
        ]
    ].copy()
    table.insert(0, "rank", range(1, len(table) + 1))
    for column in ("matched_required_skills", "missing_required_skills", "matched_preferred_skills"):
        table[column] = table[column].map(format_skills)
    table["score_percent"] = (table["score"] * 100).round(2)
    table = table.rename(
        columns={
            "rank": "Rank", "filename": "Filename", "score": "Raw score", "score_percent": "Match score (%)",
            "text_similarity": "Text similarity",
            "required_skill_coverage": "Required skill coverage",
            "preferred_skill_coverage": "Preferred skill coverage",
            "matched_required_skills": "Matched required skills",
            "missing_required_skills": "Missing required skills", "matched_preferred_skills": "Matched preferred skills",
            "years_experience": "Experience (years)",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True, column_config={"Match score (%)": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("Download displayed shortlist (CSV)", table.to_csv(index=False).encode("utf-8"), "ranked_shortlist.csv", "text/csv")
    st.markdown("### 4. Candidate details")
    for _, candidate in filtered.iterrows():
        with st.expander(f"{candidate['filename']} · score {candidate['score']:.4f}"):
            st.write(f"**Extracted experience:** {candidate['years_experience']} years")
            st.write(f"**Seniority:** {candidate.get('seniority', '') or 'Not detected'}")
            st.write(f"**Extracted skills:** {format_skills(candidate.get('skills', []))}")
            st.write(f"**Matched required skills:** {format_skills(candidate['matched_required_skills'])}")
            st.write(f"**Missing required skills:** {format_skills(candidate['missing_required_skills'])}")
            st.write(f"**Matched preferred skills:** {format_skills(candidate['matched_preferred_skills'])}")
            st.write(
                f"**Text similarity:** {candidate['text_similarity']:.4f} · "
                f"**Required coverage:** {candidate['required_skill_coverage']:.2%} · "
                f"**Preferred coverage:** {candidate['preferred_skill_coverage']:.2%}"
            )
            st.write(candidate.get("summary", "No extracted summary."))


def screening_page(jobs: pd.DataFrame) -> None:
    st.markdown('<div class="eyebrow">Screen candidates</div>', unsafe_allow_html=True)
    st.title("Configure a role, upload resumes, and build a shortlist")
    job = _job_configuration(jobs)
    if job is None:
        return
    st.markdown("### 2. Upload and review files")
    method = st.selectbox("Ranking method", ["combined", "skills", "tfidf", "learned"], format_func=str.title)
    st.caption(
        "Combined uses 70% TF-IDF job-text similarity, 22.5% required-skill coverage, and 7.5% preferred-skill coverage. "
        "Choose Skills when explicit skill coverage should dominate."
    )
    top_k = st.slider("Maximum candidates to rank", 1, 100, 10)
    uploaded_files = st.file_uploader("Upload PDF or DOCX resumes", type=["pdf", "docx"], accept_multiple_files=True)
    if not uploaded_files:
        st.info("Upload one or more resumes to continue.")
        return
    with st.spinner("Reading uploaded files..."):
        preview_resumes, preview_errors = process_uploaded_resumes(uploaded_files, job)
    st.write(f"{len(uploaded_files)} file(s) selected; {len(preview_resumes)} valid candidate(s) ready for ranking.")
    if preview_errors:
        st.warning(f"{len(preview_errors)} file(s) could not be read and will be excluded.")
        for error in preview_errors:
            st.error(f"{error['filename']}: {error['error']}")
    if not preview_resumes.empty:
        preview = preview_resumes[["filename", "role", "seniority", "years_experience", "skills"]].copy()
        preview["skills"] = preview["skills"].map(format_skills)
        st.dataframe(preview, use_container_width=True, hide_index=True)
    if st.button("Rank candidates", type="primary"):
        _prepare_upload_results(uploaded_files, job, method, top_k, preview_resumes, preview_errors)
    request = {
        "job": job, "method": method, "top_k": top_k,
        "files": [{"name": item.name, "hash": hashlib.sha256(item.getvalue()).hexdigest()} for item in uploaded_files],
    }
    signature = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
    result = st.session_state.get("screening_result")
    if not result or result["signature"] != signature:
        st.info("Click **Rank candidates** to process this upload batch.")
        return
    valid_count = len(result["ranked"])
    st.success(f"{valid_count} valid candidate(s) processed.")
    if result["errors"]:
        st.warning(f"{len(result['errors'])} file(s) failed and were excluded.")
    with st.expander("Developer diagnostics"):
        st.write({
            "job_title": job["job_title"],
            "description_characters": len(job["description"]),
            "required_skills": job["must_have_skills"],
            "preferred_skills": job["nice_to_have_skills"],
            "ranking_method": method,
            "candidates_ranked": len(result["ranked"]),
            "file_errors": len(result["errors"]),
        })
    _show_screening_results(result, job, method)


def evaluation_page() -> None:
    st.markdown('<div class="eyebrow">Model evaluation</div>', unsafe_allow_html=True)
    st.title("Evaluation on the synthetic dataset")
    st.markdown("These metrics use saved held-out job matches and are separate from uploaded-resume screening.")
    try:
        metrics = get_metrics()
    except FileNotFoundError as error:
        st.error(str(error))
        return
    table = pd.DataFrame(metrics).T
    table.index.name = "Method"
    st.dataframe(table.style.format("{:.4f}"), use_container_width=True)
    metric = st.selectbox("Metric to compare", list(table.columns), format_func=lambda value: value.replace("_", " ").title())
    st.bar_chart(table[[metric]].sort_values(metric, ascending=False))
    best = table[metric].idxmax()
    st.success(f"Best saved method for {metric.replace('_', ' ')}: {best} ({table.loc[best, metric]:.4f})")
    st.caption("Precision@10 is the fraction of the first ten results that are relevant; Recall@30 measures relevant resumes found in the first thirty; NDCG rewards relevant results near the top; MRR measures the first relevant result position.")


def explorer_page(resumes: pd.DataFrame, jobs: pd.DataFrame) -> None:
    st.markdown('<div class="eyebrow">Optional dataset explorer</div>', unsafe_allow_html=True)
    st.title("Dataset explorer")
    tab_jobs, tab_resumes = st.tabs(["Sample jobs", "Sample resumes"])
    with tab_jobs:
        st.dataframe(jobs.head(50), use_container_width=True, hide_index=True)
    with tab_resumes:
        st.dataframe(resumes.head(50), use_container_width=True, hide_index=True)


def main() -> None:
    try:
        resumes, jobs, matches = get_data()
    except FileNotFoundError as error:
        st.error(f"Dataset unavailable: {error}")
        return
    st.sidebar.markdown("## TalentLens")
    st.sidebar.caption("Intelligent resume screening")
    page = st.sidebar.radio("Workspace", ["Dashboard", "Screen Candidates", "Model Evaluation", "Dataset Explorer"])
    if page == "Dashboard":
        dashboard_page(resumes, jobs, matches)
    elif page == "Screen Candidates":
        screening_page(jobs)
    elif page == "Model Evaluation":
        evaluation_page()
    else:
        explorer_page(resumes, jobs)


if __name__ == "__main__":
    main()
