"""Reusable scoring functions for ranking resumes against a job."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data import load_resumes
from src.preprocess import build_job_text, build_leakage_safe_resume_text, clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def _skills(value: Any) -> list[str]:
    """Return normalized unique skill names."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    values = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(values, str):
        values = [values]
    return list(dict.fromkeys(clean_text(item) for item in values if clean_text(item)))


def skill_scores(job: pd.Series | dict[str, Any], resume: pd.Series | dict[str, Any]) -> tuple[float, list[str]]:
    """Return skill overlap score and the matched skills."""
    resume_skills = set(_skills(resume.get("skills", [])))
    must_have = _skills(job.get("must_have_skills", []))
    nice_to_have = _skills(job.get("nice_to_have_skills", []))
    matched_must = [skill for skill in must_have if skill in resume_skills]
    matched_nice = [skill for skill in nice_to_have if skill in resume_skills]
    must_score = len(matched_must) / len(must_have) if must_have else 0.0
    nice_score = len(matched_nice) / len(nice_to_have) if nice_to_have else 0.0
    return must_score + 0.25 * nice_score, matched_must + matched_nice


def tfidf_scores(job: pd.Series | dict[str, Any], resumes: pd.DataFrame, vectorizer: Any, resume_vectors: Any = None) -> np.ndarray:
    """Return TF-IDF cosine similarities for all resumes."""
    resume_texts = [build_leakage_safe_resume_text(resume) for _, resume in resumes.iterrows()]
    job_vector = vectorizer.transform([build_job_text(job)])
    candidate_vectors = resume_vectors if resume_vectors is not None else vectorizer.transform(resume_texts)
    return cosine_similarity(job_vector, candidate_vectors)[0]


def _pair_features(job: pd.Series | dict[str, Any], resume: pd.Series | dict[str, Any], similarity: float) -> list[float]:
    """Build the six features used by the learned ranker."""
    must = set(_skills(job.get("must_have_skills", [])))
    nice = set(_skills(job.get("nice_to_have_skills", [])))
    candidate = set(_skills(resume.get("skills", [])))
    must_overlap = len(must & candidate) / len(must) if must else 0.0
    nice_overlap = len(nice & candidate) / len(nice) if nice else 0.0
    seniority_match = int(clean_text(job.get("seniority", "")) == clean_text(resume.get("seniority", "")))
    industry_match = int(clean_text(job.get("industry", "")) == clean_text(resume.get("industry", "")))
    years = float(resume.get("years_experience", 0) or 0)
    return [similarity, must_overlap, nice_overlap, seniority_match, industry_match, years]


def rank_resumes(job: pd.Series | dict[str, Any], top_k: int = 10, method: str = "combined", combined_weight: float = 0.7, resumes: pd.DataFrame | None = None, models_dir: str | Path | None = None) -> pd.DataFrame:
    """Rank resumes and return their IDs, scores, and matched skills."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if method not in {"tfidf", "skills", "combined", "learned"}:
        raise ValueError("method must be tfidf, skills, combined, or learned")
    resumes = load_resumes() if resumes is None else resumes.copy()
    artifact_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    vectorizer = joblib.load(artifact_dir / "tfidf_vectorizer.joblib")
    similarities = tfidf_scores(job, resumes, vectorizer)
    skill_values = np.array([skill_scores(job, row)[0] for _, row in resumes.iterrows()])
    if method == "tfidf":
        scores = similarities
    elif method == "skills":
        scores = skill_values
    elif method == "combined":
        scores = combined_weight * similarities + (1 - combined_weight) * skill_values
    else:
        model = joblib.load(artifact_dir / "ranker_model.joblib")
        features = np.array([_pair_features(job, row, similarity) for (_, row), similarity in zip(resumes.iterrows(), similarities)])
        scores = model.predict_proba(features)[:, 1]
    result = resumes.copy()
    result["score"] = scores
    result["matched_skills"] = [skill_scores(job, row)[1] for _, row in resumes.iterrows()]
    return result.sort_values(["score", "resume_id"], ascending=[False, True]).head(top_k).reset_index(drop=True)


def pair_features(job: pd.Series | dict[str, Any], resume: pd.Series | dict[str, Any], similarity: float) -> list[float]:
    """Return the learned ranker's six pair features."""
    return _pair_features(job, resume, similarity)