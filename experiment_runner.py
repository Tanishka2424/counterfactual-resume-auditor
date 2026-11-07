
from pathlib import Path
import random
import time
import pandas as pd

from screening import screen_resume

DATA_PATH = Path("data/gender_name_resumes.csv")
OUTPUT_DIR = Path("reports/generated")

CHECKPOINT_PATH = OUTPUT_DIR / "batch_checkpoint.csv"
RAW_PATH = OUTPUT_DIR / "batch_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "batch_summary.csv"

JOB_DESCRIPTION = """
Python Developer / AI Engineer.

Requirements:
Python programming, software development,
data structures, APIs, and relevant technical education.
"""


def run_batch(trials=1):
    if trials < 1:
        raise ValueError("Trials must be at least 1.")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(DATA_PATH)

    required = {"resume_id", "resume_a", "resume_b"}
    if not required.issubset(dataset.columns):
        raise ValueError("Dataset has missing required columns.")

    # Load completed resume pairs if restarting.
    if CHECKPOINT_PATH.exists():
        existing = pd.read_csv(CHECKPOINT_PATH)
        completed_ids = set(existing["resume_id"].unique())
        results = existing.to_dict("records")
        print(f"Resuming: {len(completed_ids)} pairs already completed.")
    else:
        completed_ids = set()
        results = []

    pending = dataset[~dataset["resume_id"].isin(completed_ids)]

    total_evaluations = len(pending) * trials * 2
    completed = 0
    start_time = time.perf_counter()

    for _, row in pending.iterrows():
        pair_results = []

        for trial in range(1, trials + 1):
            versions = ["A", "B"]
            random.shuffle(versions)
            evaluations = {}

            for version in versions:
                resume_text = (
                    row["resume_a"] if version == "A"
                    else row["resume_b"]
                )

                evaluations[version] = screen_resume(
                    resume_text,
                    JOB_DESCRIPTION,
                )

                completed += 1
                print(
                    f"[{completed}/{total_evaluations}] "
                    f"Resume {row['resume_id']} | "
                    f"Trial {trial} | Version {version}",
                    flush=True,
                )

            score_a = evaluations["A"]
            score_b = evaluations["B"]
            gap = score_b["score"] - score_a["score"]

            pair_results.append({
                "resume_id": row["resume_id"],
                "trial": trial,
                "score_a": score_a["score"],
                "score_b": score_b["score"],
                "signed_gap": gap,
                "absolute_gap": abs(gap),
                "reasoning_a": score_a["reasoning"],
                "reasoning_b": score_b["reasoning"],
            })

        # Save after each fully completed resume pair.
        results.extend(pair_results)
        checkpoint = pd.DataFrame(results)
        checkpoint.to_csv(CHECKPOINT_PATH, index=False)

        print(
            f"Checkpoint saved: Resume {row['resume_id']}",
            flush=True,
        )

    df = pd.DataFrame(results)
    df.to_csv(RAW_PATH, index=False)

    summary = (
        df.groupby("resume_id")
        .agg(
            mean_score_a=("score_a", "mean"),
            mean_score_b=("score_b", "mean"),
            mean_signed_gap=("signed_gap", "mean"),
            mean_absolute_gap=("absolute_gap", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY_PATH, index=False)

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 45)
    print("BATCH EXPERIMENT COMPLETED")
    print("=" * 45)
    print(f"Resume pairs: {dataset['resume_id'].nunique()}")
    print(f"Trials per pair: {trials}")
    print(f"Completed evaluations: {len(df) * 2}")
    print(f"Elapsed this run: {elapsed:.2f}s")
    print(
        "Overall mean absolute gap:",
        round(df["absolute_gap"].mean(), 3),
    )
    print(f"\nRaw results: {RAW_PATH}")
    print(f"Summary: {SUMMARY_PATH}")

    return df, summary


if __name__ == "__main__":
    run_batch(trials=1)
