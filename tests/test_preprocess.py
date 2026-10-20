import pandas as pd

from src.preprocess import build_job_text, build_resume_text, clean_text


def test_clean_text_removes_urls_emails_and_symbols():
    assert clean_text("Hello@example.com Visit https://example.com!  Data-science") == "visit data science"


def test_build_resume_text_uses_requested_fields_and_ignores_extra_fields():
    resume = {
        "skills": ["Python", "SQL"],
        "role": "Data Engineer",
        "seniority": "Senior",
        "industry": "Finance",
        "summary": "Builds reliable pipelines.",
        "experience_bullets": ["This must not be used"],
    }
    text = build_resume_text(pd.Series(resume))
    assert "python" in text
    assert "data engineer" in text
    assert "this must not be used" not in text


def test_build_job_text_supports_title_alias():
    job = {"title": "Backend Engineer", "must_have_skills": ["Python"], "nice_to_have_skills": ["Docker"], "description": "Build APIs"}
    assert build_job_text(job) == "backend engineer python docker build apis"