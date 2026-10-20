"""Command-line demo for ranking resumes."""

import argparse

from src.data import load_jobs
from src.ranker import rank_resumes


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank resumes for a job")
    parser.add_argument("--job-id")
    parser.add_argument("--title")
    parser.add_argument("--must-have-skills", nargs="*", default=[])
    parser.add_argument("--nice-to-have-skills", nargs="*", default=[])
    parser.add_argument("--method", choices=("tfidf", "skills", "combined", "learned"), default="combined")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if bool(args.job_id) == bool(args.title):
        parser.error("provide exactly one of --job-id or --title")
    if args.job_id:
        jobs = load_jobs().set_index("job_id")
        if args.job_id not in jobs.index:
            parser.error(f"unknown job ID: {args.job_id}")
        job = jobs.loc[args.job_id]
    else:
        job = {"job_title": args.title, "must_have_skills": args.must_have_skills, "nice_to_have_skills": args.nice_to_have_skills}
    results = rank_resumes(job, top_k=args.top_k, method=args.method)
    for index, row in results.iterrows():
        print(f"{index + 1:>2}. {row['resume_id']}  score={row['score']:.4f}  matched skills={', '.join(row['matched_skills']) or 'none'}")


if __name__ == "__main__":
    main()