"""Evaluate ranking methods on held-out jobs."""

import json
import math
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data import load_jobs, load_matches, load_resumes
from src.preprocess import build_job_text, build_leakage_safe_resume_text
from src.ranker import _skills, skill_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def precision_at_k(ranked_ids: Iterable[str], relevant_ids: set[str], k: int = 10) -> float:
    """Return the fraction of the first k results that are relevant."""
    ranked = list(ranked_ids)[:k]
    return sum(item in relevant_ids for item in ranked) / k


def recall_at_k(ranked_ids: Iterable[str], relevant_ids: set[str], k: int = 30) -> float:
    """Return the fraction of relevant resumes found in the first k results."""
    ranked = list(ranked_ids)[:k]
    return sum(item in relevant_ids for item in ranked) / len(relevant_ids) if relevant_ids else 0.0


def ndcg_at_k(ranked_ids: Iterable[str], relevant_ids: set[str], k: int = 10) -> float:
    """Return normalized discounted cumulative gain for binary relevance."""
    ranked = list(ranked_ids)[:k]
    dcg = sum(1 / math.log2(index + 2) for index, item in enumerate(ranked) if item in relevant_ids)
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def mean_reciprocal_rank(ranked_ids: Iterable[str], relevant_ids: set[str]) -> float:
    """Return the reciprocal rank of the first relevant result."""
    for index, item in enumerate(ranked_ids, start=1):
        if item in relevant_ids:
            return 1 / index
    return 0.0


def evaluate_method(jobs: pd.DataFrame, matches: pd.DataFrame, test_job_ids: list[str], method: str, combined_weight: float = 0.7) -> dict[str, float]:
    """Calculate all requested metrics for one method."""
    values = {"precision_at_10": [], "recall_at_30": [], "ndcg_at_10": [], "mrr": []}
    truth = matches.set_index("job_id")["relevant_resume_ids"]
    jobs_by_id = jobs.set_index("job_id")
    for job_id in test_job_ids:
        ranked = rank_resumes(jobs_by_id.loc[job_id], top_k=30, method=method, combined_weight=combined_weight)
        ranked_ids = ranked["resume_id"].tolist()
        relevant = set(truth.loc[job_id])
        values["precision_at_10"].append(precision_at_k(ranked_ids, relevant))
        values["recall_at_30"].append(recall_at_k(ranked_ids, relevant))
        values["ndcg_at_10"].append(ndcg_at_k(ranked_ids, relevant))
        values["mrr"].append(mean_reciprocal_rank(ranked_ids, relevant))
    return {metric: sum(scores) / len(scores) for metric, scores in values.items()}


def evaluate_all(jobs: pd.DataFrame, matches: pd.DataFrame, test_job_ids: list[str]) -> dict[str, dict[str, float]]:
    """Evaluate every method while computing each job's base scores once."""
    resumes = load_resumes()
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    resume_vectors = vectorizer.transform([build_leakage_safe_resume_text(row) for _, row in resumes.iterrows()])
    resume_skills = [set(_skills(value)) for value in resumes["skills"]]
    resume_seniority = resumes["seniority"].map(lambda value: str(value).lower()).to_numpy()
    resume_industry = resumes["industry"].map(lambda value: str(value).lower()).to_numpy()
    resume_years = resumes["years_experience"].fillna(0).astype(float).to_numpy()
    job_vectors = vectorizer.transform([build_job_text(row) for _, row in jobs.iterrows()])
    all_similarities = cosine_similarity(job_vectors, resume_vectors)
    job_positions = {job_id: index for index, job_id in enumerate(jobs["job_id"])}
    model = joblib.load(MODELS_DIR / "ranker_model.joblib")
    jobs_by_id = jobs.set_index("job_id")
    truth = matches.set_index("job_id")["relevant_resume_ids"]
    method_scores = {name: {metric: [] for metric in ("precision_at_10", "recall_at_30", "ndcg_at_10", "mrr")} for name in ("tfidf", "skills", "learned", "combined_0.3", "combined_0.5", "combined_0.7", "combined_0.9")}
    for job_id in test_job_ids:
        job = jobs_by_id.loc[job_id]
        similarities = all_similarities[job_positions[job_id]]
        must_have = set(_skills(job.get("must_have_skills", [])))
        nice_to_have = set(_skills(job.get("nice_to_have_skills", [])))
        must_denominator = len(must_have) or 1
        nice_denominator = len(nice_to_have) or 1
        required_coverage = np.array([len(must_have & candidate) / must_denominator for candidate in resume_skills])
        preferred_coverage = np.array([len(nice_to_have & candidate) / nice_denominator for candidate in resume_skills])
        skills = 0.75 * required_coverage + 0.25 * preferred_coverage
        seniority = str(job.get("seniority", "")).lower()
        industry = str(job.get("industry", "")).lower()
        learned_features = np.column_stack((
            similarities,
            required_coverage,
            preferred_coverage,
            resume_seniority == seniority,
            resume_industry == industry,
            resume_years,
        ))
        scores_by_method = {"tfidf": similarities, "skills": skills, "learned": model.predict_proba(learned_features)[:, 1]}
        for weight in (0.3, 0.5, 0.7, 0.9):
            scores_by_method[f"combined_{weight:.1f}"] = weight * similarities + (1 - weight) * skills
        relevant = set(truth.loc[job_id])
        for method, scores in scores_by_method.items():
            order = np.lexsort((resumes["resume_id"].to_numpy(), -scores))[:30]
            ranked_ids = resumes.iloc[order]["resume_id"].tolist()
            method_scores[method]["precision_at_10"].append(precision_at_k(ranked_ids, relevant))
            method_scores[method]["recall_at_30"].append(recall_at_k(ranked_ids, relevant))
            method_scores[method]["ndcg_at_10"].append(ndcg_at_k(ranked_ids, relevant))
            method_scores[method]["mrr"].append(mean_reciprocal_rank(ranked_ids, relevant))
    return {method: {metric: sum(values) / len(values) for metric, values in metrics.items()} for method, metrics in method_scores.items()}


def main() -> None:
    """Evaluate all ranking methods and save a JSON comparison."""
    jobs = load_jobs()
    matches = load_matches()
    test_ids = json.loads((MODELS_DIR / "test_job_ids.json").read_text(encoding="utf-8"))
    results = evaluate_all(jobs, matches, test_ids)
    print(pd.DataFrame(results).T.to_string(float_format=lambda value: f"{value:.4f}"))
    (MODELS_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()