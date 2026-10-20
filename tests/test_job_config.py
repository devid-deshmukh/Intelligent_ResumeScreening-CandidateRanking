import pandas as pd

from src.job_config import job_from_existing, parse_skill_text, role_suggestions


def test_parse_skill_text_deduplicates_case_and_whitespace():
    assert parse_skill_text("Python, python; SQL\n SQL") == ["Python", "SQL"]


def test_role_suggestions_use_dataset_skills():
    jobs = pd.DataFrame(
        [
            {
                "job_id": "J1",
                "job_title": "Data Engineer",
                "must_have_skills": ["Python", "SQL"],
                "nice_to_have_skills": ["Airflow"],
            },
            {
                "job_id": "J2",
                "job_title": "Data Engineer",
                "must_have_skills": ["Python"],
                "nice_to_have_skills": ["Cloud"],
            },
        ]
    )
    required, preferred = role_suggestions("Data Engineer", jobs)
    assert required == ["python", "sql"]
    assert preferred == ["airflow", "cloud"]


def test_unknown_role_uses_maintainable_catalogue():
    required, preferred = role_suggestions("Software Engineer")
    assert "Python" in required
    assert preferred


def test_existing_job_conversion_preserves_supported_fields():
    row = pd.Series(
        {
            "job_id": "J1",
            "job_title": "Analyst",
            "description": "Analyze data",
            "must_have_skills": ["SQL"],
            "nice_to_have_skills": ["Excel"],
            "seniority": "Senior",
            "industry": "Finance",
        }
    )
    job = job_from_existing(row)
    assert job["job_id"] == "J1"
    assert job["must_have_skills"] == ["sql"]
    assert job["industry"] == "Finance"


def test_three_selected_roles_produce_distinct_job_inputs():
    jobs = pd.DataFrame(
        [
            {"job_id": "J1", "job_title": "Python Developer", "description": "Build Python services", "must_have_skills": ["Python"], "nice_to_have_skills": [], "seniority": "Mid", "industry": "Tech"},
            {"job_id": "J2", "job_title": "Java Developer", "description": "Build Java services", "must_have_skills": ["Java"], "nice_to_have_skills": [], "seniority": "Senior", "industry": "Finance"},
            {"job_id": "J3", "job_title": "Data Analyst", "description": "Analyze data with SQL", "must_have_skills": ["SQL"], "nice_to_have_skills": ["Excel"], "seniority": "Junior", "industry": "Retail"},
        ]
    )
    profiles = [job_from_existing(row) for _, row in jobs.iterrows()]
    assert [profile["job_title"] for profile in profiles] == ["Python Developer", "Java Developer", "Data Analyst"]
    assert [profile["must_have_skills"] for profile in profiles] == [["python"], ["java"], ["sql"]]
    assert len({profile["description"] for profile in profiles}) == 3


def test_existing_array_fields_are_safe_to_combine():
    row = pd.Series(
        {
            "job_id": "J1",
            "job_title": "Data Analyst",
            "description": "Analyze data",
            "responsibilities": ["Build reports", "Present findings"],
            "requirements": ["SQL", "Excel"],
            "must_have_skills": ["SQL"],
            "nice_to_have_skills": ["Excel"],
            "seniority": "Junior",
            "industry": "Retail",
        }
    )
    profile = job_from_existing(row)
    assert "Build reports" in profile["description"]
    assert "SQL" in profile["description"]
