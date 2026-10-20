import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.preprocess import build_leakage_safe_resume_text
from src.ranker import pair_features, rank_resumes, skill_scores


def sample_resumes():
    return pd.DataFrame(
        [
            {"resume_id": "R1", "role": "Engineer", "seniority": "Senior", "industry": "Tech", "years_experience": 8, "skills": ["Python", "SQL"], "summary": "Engineer"},
            {"resume_id": "R2", "role": "Designer", "seniority": "Junior", "industry": "Retail", "years_experience": 2, "skills": ["Figma"], "summary": "Designer"},
        ]
    )


def sample_job():
    return {"job_title": "Python Engineer", "seniority": "Senior", "industry": "Tech", "must_have_skills": ["Python"], "nice_to_have_skills": ["SQL"], "description": "Build software"}


def test_skill_scores_and_pair_features():
    score, matched = skill_scores(sample_job(), sample_resumes().iloc[0])
    assert score == 1.25
    assert matched == ["python", "sql"]
    assert pair_features(sample_job(), sample_resumes().iloc[0], 0.5) == [0.5, 1.0, 1.0, 1, 1, 8.0]


def test_rank_resumes_returns_top_k_and_expected_order(tmp_path):
    resumes = sample_resumes()
    vectorizer = TfidfVectorizer()
    vectorizer.fit([build_leakage_safe_resume_text(row) for _, row in resumes.iterrows()])
    joblib.dump(vectorizer, tmp_path / "tfidf_vectorizer.joblib")
    result = rank_resumes(sample_job(), top_k=1, method="skills", resumes=resumes, models_dir=tmp_path)
    assert list(result["resume_id"]) == ["R1"]
    assert list(result["matched_skills"]) == [["python", "sql"]]


def test_learned_method_loads_saved_model(tmp_path):
    resumes = sample_resumes()
    vectorizer = TfidfVectorizer()
    vectorizer.fit([build_leakage_safe_resume_text(row) for _, row in resumes.iterrows()])
    joblib.dump(vectorizer, tmp_path / "tfidf_vectorizer.joblib")
    model = LogisticRegression().fit([[1, 1, 1, 1, 1, 8], [0, 0, 0, 0, 0, 2]], [1, 0])
    joblib.dump(model, tmp_path / "ranker_model.joblib")
    result = rank_resumes(sample_job(), top_k=2, method="learned", resumes=resumes, models_dir=tmp_path)
    assert len(result) == 2
    assert "score" in result


def test_custom_required_skills_change_skill_ranking(tmp_path):
    resumes = sample_resumes()
    vectorizer = TfidfVectorizer()
    vectorizer.fit([build_leakage_safe_resume_text(row) for _, row in resumes.iterrows()])
    joblib.dump(vectorizer, tmp_path / "tfidf_vectorizer.joblib")
    python_job = {"job_title": "Engineer", "must_have_skills": ["Python"], "nice_to_have_skills": []}
    figma_job = {"job_title": "Designer", "must_have_skills": ["Figma"], "nice_to_have_skills": []}
    assert rank_resumes(python_job, top_k=1, method="skills", resumes=resumes, models_dir=tmp_path).iloc[0]["resume_id"] == "R1"
    assert rank_resumes(figma_job, top_k=1, method="skills", resumes=resumes, models_dir=tmp_path).iloc[0]["resume_id"] == "R2"


def test_combined_score_exposes_bounded_explainable_components(tmp_path):
    resumes = sample_resumes()
    vectorizer = TfidfVectorizer()
    vectorizer.fit([build_leakage_safe_resume_text(row) for _, row in resumes.iterrows()])
    joblib.dump(vectorizer, tmp_path / "tfidf_vectorizer.joblib")
    result = rank_resumes(sample_job(), top_k=2, method="combined", resumes=resumes, models_dir=tmp_path)
    assert result["score"].between(0, 1).all()
    assert result["required_skill_coverage"].between(0, 1).all()
    assert result["preferred_skill_coverage"].between(0, 1).all()
    assert result.iloc[0]["required_skill_coverage"] == 1.0