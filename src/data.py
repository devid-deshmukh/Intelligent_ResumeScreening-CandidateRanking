"""Load the raw resume screening data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _load_parquet(filename: str, path: str | Path | None = None) -> pd.DataFrame:
    """Load one Parquet file from the raw data directory."""
    file_path = Path(path) if path is not None else RAW_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_parquet(file_path)


def load_resumes(path: str | Path | None = None) -> pd.DataFrame:
    """Return the resume records."""
    return _load_parquet("resume.parquet", path)


def load_jobs(path: str | Path | None = None) -> pd.DataFrame:
    """Return the job records."""
    return _load_parquet("jobs.parquet", path)


def load_matches(path: str | Path | None = None) -> pd.DataFrame:
    """Return the ground-truth relevant resume IDs for each job."""
    return _load_parquet("matches.parquet", path)