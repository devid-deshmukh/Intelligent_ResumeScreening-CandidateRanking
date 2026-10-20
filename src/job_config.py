"""Job configuration and role-based skill suggestions."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from src.ranker import _skills


CATALOGUE: dict[str, tuple[list[str], list[str]]] = {
    "software engineer": (["Python", "Software Development"], ["Git", "Testing", "Cloud"]),
    "data engineer": (["Python", "SQL", "Data Engineering"], ["Airflow", "Cloud", "Spark"]),
    "data scientist": (["Python", "Statistics", "Machine Learning"], ["SQL", "TensorFlow", "Experimentation"]),
    "product manager": (["Product Strategy", "Stakeholder Communication"], ["Agile", "Analytics", "Roadmapping"]),
    "project manager": (["Project Management", "Stakeholder Communication"], ["Agile", "Asana", "Risk Management"]),
    "business analyst": (["Requirements Gathering", "SQL", "Data Analysis"], ["Excel", "Process Improvement", "Power BI"]),
    "marketing manager": (["Marketing Strategy", "Content Marketing"], ["SEO", "Analytics", "Campaign Management"]),
}


def _field_text(value: Any) -> str:
    """Convert scalar and array-valued dataset fields into searchable text."""
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value)
    return str(value)


def parse_skill_text(value: str) -> list[str]:
    """Parse comma, semicolon, or newline-separated skills into unique names."""
    values = value.replace(";", ",").replace("\n", ",").split(",")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        skill = value.strip()
        key = " ".join(skill.casefold().split())
        if skill and key not in seen:
            result.append(skill)
            seen.add(key)
    return result


def role_suggestions(role: str, jobs: pd.DataFrame | None = None) -> tuple[list[str], list[str]]:
    """Return editable required and preferred skills for a role."""
    normalized = " ".join(role.casefold().split())
    if jobs is not None and normalized:
        matching = jobs[jobs["job_title"].astype(str).str.casefold().str.strip() == normalized]
        if not matching.empty:
            required = Counter(skill for values in matching["must_have_skills"] for skill in _skills(values))
            preferred = Counter(skill for values in matching["nice_to_have_skills"] for skill in _skills(values))
            return [skill for skill, _ in required.most_common()], [skill for skill, _ in preferred.most_common()]
    for catalogue_role, suggestions in CATALOGUE.items():
        if catalogue_role in normalized or normalized in catalogue_role:
            return suggestions[0].copy(), suggestions[1].copy()
    return [], []


def job_from_existing(row: pd.Series) -> dict[str, Any]:
    """Convert a dataset job row to the schema consumed by the ranker."""
    description_parts = [
        _field_text(row.get("description", "")),
        _field_text(row.get("responsibilities", "")),
        _field_text(row.get("requirements", "")),
    ]
    return {
        "job_id": row.get("job_id", ""),
        "job_title": row.get("job_title", ""),
        "description": " ".join(part for part in description_parts if part).strip(),
        "responsibilities": row.get("responsibilities", ""),
        "requirements": row.get("requirements", ""),
        "must_have_skills": _skills(row.get("must_have_skills", [])),
        "nice_to_have_skills": _skills(row.get("nice_to_have_skills", [])),
        "seniority": row.get("seniority", ""),
        "industry": row.get("industry", ""),
    }


def job_profile_signature(job: dict[str, Any]) -> str:
    """Return a stable signature for invalidating rankings after job edits."""
    fields = (
        job.get("job_title", ""),
        job.get("description", ""),
        tuple(job.get("must_have_skills", [])),
        tuple(job.get("nice_to_have_skills", [])),
        job.get("seniority", ""),
        job.get("industry", ""),
        job.get("minimum_experience", 0),
    )
    return repr(fields)
