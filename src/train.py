"""Train and persist the TF-IDF vectorizer and learned ranker."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from src.data import load_jobs, load_matches, load_resumes
from src.preprocess import build_job_text, build_leakage_safe_resume_text
from src.ranker import pair_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    """Fit artifacts and save them in the models directory."""
    resumes = load_resumes()
    jobs = load_jobs()
    matches = load_matches()
    resume_by_id = resumes.set_index("resume_id")
    job_by_id = jobs.set_index("job_id")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    resume_texts = [build_leakage_safe_resume_text(row) for _, row in resumes.iterrows()]
    vectorizer.fit(resume_texts)
    resume_vectors = vectorizer.transform(resume_texts)
    job_texts = [build_job_text(row) for _, row in jobs.iterrows()]
    job_vectors = vectorizer.transform(job_texts)
    resume_positions = {resume_id: index for index, resume_id in enumerate(resumes["resume_id"])}
    job_positions = {job_id: index for index, job_id in enumerate(jobs["job_id"])}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")

    job_ids = matches["job_id"].tolist()
    train_ids, test_ids = train_test_split(job_ids, test_size=0.2, random_state=42)
    rows: list[list[float]] = []
    labels: list[int] = []
    rng = np.random.default_rng(42)
    all_resume_ids = set(resumes["resume_id"])
    for job_id in train_ids:
        job = job_by_id.loc[job_id]
        positive_ids = [resume_id for resume_id in matches.loc[matches["job_id"] == job_id, "relevant_resume_ids"].iloc[0] if resume_id in resume_by_id.index]
        negative_pool = sorted(all_resume_ids - set(positive_ids))
        negative_ids = rng.choice(negative_pool, size=min(len(positive_ids), len(negative_pool)), replace=False)
        for resume_id, label in [(item, 1) for item in positive_ids] + [(item, 0) for item in negative_ids]:
            resume = resume_by_id.loc[resume_id]
            similarity = float(cosine_similarity(job_vectors[job_positions[job_id]], resume_vectors[resume_positions[resume_id]])[0, 0])
            rows.append(pair_features(job, resume, similarity))
            labels.append(label)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(rows, labels)
    joblib.dump(model, MODELS_DIR / "ranker_model.joblib")
    (MODELS_DIR / "test_job_ids.json").write_text(json.dumps(test_ids, indent=2), encoding="utf-8")
    print(f"Trained on {len(train_ids)} jobs; held out {len(test_ids)} jobs.")


if __name__ == "__main__":
    main()