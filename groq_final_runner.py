
"""
Counterfactual Resume Auditor — Groq Final Experiment
30 profiles × 2 conditions = 60 evaluations.
Supports checkpointing, resume, retries, and paired output.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from openai import RateLimitError
from screening import screen_resume




def call_with_retry(resume, job, max_retries=5):
    for attempt in range(max_retries + 1):
        try:
            return screen_resume(resume, job)

        except Exception as exc:
            error = str(exc).lower()

            transient = any(term in error for term in [
                "429",
                "rate limit",
                "empty response",
                "did not return valid json",
                "invalid json",
                "503",
                "502",
                "500",
            ])

            if not transient or attempt == max_retries:
                raise

            wait = min(30, 2 ** (attempt + 1))

            print(
                f"Retryable error: {exc}\n"
                f"Waiting {wait}s "
                f"({attempt + 1}/{max_retries})...",
                flush=True,
            )

            time.sleep(wait)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default="groq_profiles.csv")
    parser.add_argument("--job", default="job_description.txt")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--output",
        default="results/groq_final_experiment.csv",
    )

    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")

    profiles = pd.read_csv(args.input, dtype=str).fillna("")

    required = {
        "profile_id",
        "resume",
        "original_name",
        "counterfactual_name",
    }

    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"Input missing columns: {sorted(missing)}")

    if profiles["profile_id"].duplicated().any():
        raise ValueError("Duplicate profile_id values in input.")

    job_path = Path(args.job)
    if not job_path.exists():
        raise FileNotFoundError(f"Job description not found: {job_path}")

    job = job_path.read_text(encoding="utf-8")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load previous checkpoint.
    if output.exists() and output.stat().st_size > 0:
        existing = pd.read_csv(output)

        required_out = {
            "profile_id",
            "trial",
            "condition",
            "score",
        }

        if not required_out.issubset(existing.columns):
            raise ValueError(
                f"Checkpoint missing columns: "
                f"{required_out - set(existing.columns)}"
            )

        existing["trial"] = existing["trial"].astype(int)
        rows = existing.to_dict("records")

    else:
        rows = []

    # Keep latest record for each completed evaluation.
    completed = {}

    for row in rows:
        key = (
            str(row["profile_id"]),
            int(row["trial"]),
            str(row["condition"]),
        )
        completed[key] = row

    total = len(profiles) * args.trials * 2

    already_done = sum(
        1 for key in completed
        if key[1] <= args.trials
    )

    print(
        f"Target: {total} evaluations | "
        f"Already checkpointed: {already_done} | "
        f"Remaining: {max(0, total - already_done)}",
        flush=True,
    )

    # Run experiment.
    for _, profile in profiles.iterrows():

        pid = str(profile["profile_id"])
        original = str(profile["resume"])
        original_name = str(profile["original_name"])
        cf_name = str(profile["counterfactual_name"])

        if not original_name:
            raise ValueError(f"{pid}: original_name is empty.")

        if original.count(original_name) != 1:
            raise ValueError(
                f"{pid}: original_name must occur exactly once."
            )

        counterfactual = original.replace(
            original_name,
            cf_name,
            1,
        )

        if counterfactual == original:
            raise ValueError(
                f"{pid}: counterfactual resume was not generated."
            )

        for trial in range(1, args.trials + 1):

            for condition, resume_text in [
                ("original", original),
                ("counterfactual", counterfactual),
            ]:

                key = (pid, trial, condition)

                if key in completed:
                    continue

                print(
                    f"[{len(completed) + 1}/{total}] "
                    f"{pid} trial {trial} {condition}",
                    flush=True,
                )

                result = call_with_retry(resume_text, job)

                row = {
                    "profile_id": pid,
                    "trial": trial,
                    "condition": condition,
                    "score": result["score"],
                    "skills_score": result["skills_score"],
                    "experience_score": result["experience_score"],
                    "education_score": result["education_score"],
                    "relevance_score": result["relevance_score"],
                    "reasoning": result.get("reasoning", ""),
                }

                completed[key] = row

                # Save checkpoint after every API call.
                pd.DataFrame(
                    list(completed.values())
                ).to_csv(output, index=False)

                time.sleep(1.0)

    # Generate paired data using trial 1 only.
    raw = pd.DataFrame(list(completed.values()))

    raw = raw[
        raw["trial"].astype(int) == 1
    ]

    wide = raw.pivot(
        index="profile_id",
        columns="condition",
        values="score",
    ).reset_index()

    if {"original", "counterfactual"}.issubset(wide.columns):

        wide["signed_difference_B_minus_A"] = (
            wide["counterfactual"] - wide["original"]
        )

        wide["absolute_gap"] = (
            wide["signed_difference_B_minus_A"].abs()
        )

    paired_path = output.with_name(
        "groq_final_paired.csv"
    )

    wide.to_csv(paired_path, index=False)

    completed_profiles = len(
        wide.dropna(
            subset=["original", "counterfactual"]
        )
    )

    print("\nExperiment finished.", flush=True)
    print(
        f"Completed paired profiles: "
        f"{completed_profiles}/{len(profiles)}",
        flush=True,
    )
    print(f"Raw checkpoint: {output}", flush=True)
    print(f"Paired output: {paired_path}", flush=True)


if __name__ == "__main__":
    main()
