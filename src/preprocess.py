"""Text cleaning and construction of resume and job text."""

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import load_jobs, load_resumes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def _as_text(value: Any) -> str:
    """Turn scalar and list-like values into plain text."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, tuple, set)) or hasattr(value, "tolist"):
        values = value.tolist() if hasattr(value, "tolist") else value
        return " ".join(str(item) for item in values)
    return str(value)


def clean_text(value: Any) -> str:
    """Lowercase text, remove URLs/emails/symbols, and collapse whitespace."""
    text = _as_text(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_resume_text(resume: pd.Series | dict[str, Any]) -> str:
    """Build searchable text from the requested resume fields."""
    values = [
        resume.get("skills", ""),
        resume.get("role", ""),
        resume.get("seniority", ""),
        resume.get("industry", ""),
        resume.get("summary", ""),
    ]
    return clean_text(" ".join(_as_text(value) for value in values))


def build_job_text(job: pd.Series | dict[str, Any]) -> str:
    """Build searchable text from the job title, skills, and description."""
    values = [
        job.get("job_title", job.get("title", "")),
        job.get("must_have_skills", ""),
        job.get("nice_to_have_skills", ""),
        job.get("description", ""),
    ]
    return clean_text(" ".join(_as_text(value) for value in values))


def build_leakage_safe_resume_text(resume: pd.Series | dict[str, Any]) -> str:
    """Build resume text without the role field or its repeated summary phrase."""
    role = clean_text(resume.get("role", ""))
    summary = clean_text(resume.get("summary", ""))
    if role:
        summary = re.sub(rf"\b{re.escape(role)}\b", " ", summary)
    values = [resume.get("skills", ""), resume.get("seniority", ""), resume.get("industry", ""), summary]
    return clean_text(" ".join(_as_text(value) for value in values))


def main() -> None:
    """Create cleaned Parquet files for resumes and jobs."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    resumes = load_resumes().copy()
    jobs = load_jobs().copy()
    resumes["resume_text"] = resumes.apply(build_resume_text, axis=1)
    jobs["job_text"] = jobs.apply(build_job_text, axis=1)
    resumes.to_parquet(PROCESSED_DATA_DIR / "resumes.parquet", index=False)
    jobs.to_parquet(PROCESSED_DATA_DIR / "jobs.parquet", index=False)
    print(f"Saved processed data to {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()