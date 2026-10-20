# Intelligent Resume Screening and Candidate Ranking

This project ranks synthetic resumes for a job using text similarity, skill overlap, a combined score, and a learned logistic-regression ranker. The code is organized as reusable Python functions so a future Streamlit interface can import it directly.

## Setup

From the project root on Windows:

```text
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

Place these files in `data/raw/`:

- `resume.parquet`
- `jobs.parquet`
- `matches.parquet`

The raw dataset is intentionally ignored by Git.

TODO: add the link to the dataset source.

## Workflow

Run these commands from the project root in order:

```text
venv\Scripts\python.exe -m src.preprocess
venv\Scripts\python.exe -m src.train
venv\Scripts\python.exe -m src.evaluate
venv\Scripts\python.exe -m pytest -q
```

Preprocessing writes cleaned resume and job files to `data/processed/`. Training writes `tfidf_vectorizer.joblib`, `ranker_model.joblib`, and `test_job_ids.json` to `models/`. Evaluation prints a comparison table and writes `models/metrics.json`.

## Command-line demo

Use an existing job ID:

```text
venv\Scripts\python.exe -m src.rank_cli --job-id J_000000 --method combined --top-k 10
```

Use a custom job instead:

```text
venv\Scripts\python.exe -m src.rank_cli --title "Python Engineer" --must-have-skills Python SQL --nice-to-have-skills Docker --method skills --top-k 10
```

Available methods are `tfidf`, `skills`, `combined`, and `learned`.

## Limitations

- The supplied data is synthetic, so scores may not represent real hiring performance.
- Automated resume screening can reproduce or amplify bias in data and job criteria.
- Ranking is a decision-support tool and should not be the sole basis for hiring decisions.
- The learned model uses job-level train/test splitting, but it is still trained from synthetic relevance labels.