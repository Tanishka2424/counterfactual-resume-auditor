
from pathlib import Path
import random
import time
import pandas as pd

from screening import screen_resume

DATA_PATH = Path("data/synthetic_resumes.csv")
OUTPUT_PATH = Path("reports/generated/resume6_diagnostic.csv")

JOB_DESCRIPTION = """
Python Developer / AI Engineer.

Requirements:
Python programming, software development,
data structures, APIs, and relevant technical education.
"""


def run_diagnostic(trials=3, control_trials=2):
    dataset = pd.read_csv(DATA_PATH)
    row = dataset.loc[dataset["resume_id"] == 6].iloc[0]

    resume_a = row["resume_a"]
    resume_b = row["resume_b"]

    results = []
    total = trials * 2 + control_trials * 2
    completed = 0
    start = time.perf_counter()

    # Repeat the actual name comparison.
    for trial in range(1, trials + 1):
        versions = ["A", "B"]
        random.shuffle(versions)
        scores = {}

        for version in versions:
            resume = resume_a if version == "A" else resume_b
            result = screen_resume(resume, JOB_DESCRIPTION)
            scores[version] = result
            completed += 1

            print(
                f"[{completed}/{total}] "
                f"Name comparison | Trial {trial} | {version}",
                flush=True,
            )

        results.append({
            "experiment": "name_comparison",
            "trial": trial,
            "score_a": scores["A"]["score"],
            "score_b": scores["B"]["score"],
            "signed_gap": scores["B"]["score"] - scores["A"]["score"],
            "reasoning_a": scores["A"]["reasoning"],
            "reasoning_b": scores["B"]["reasoning"],
        })

    # Control: identical resume screened twice.
    for trial in range(1, control_trials + 1):
        scores = []

        for _ in range(2):
            result = screen_resume(resume_a, JOB_DESCRIPTION)
            scores.append(result)
            completed += 1

            print(
                f"[{completed}/{total}] "
                f"Identical-resume control | Trial {trial}",
                flush=True,
            )

        results.append({
            "experiment": "identical_resume_control",
            "trial": trial,
            "score_a": scores[0]["score"],
            "score_b": scores[1]["score"],
            "signed_gap": scores[1]["score"] - scores[0]["score"],
            "reasoning_a": scores[0]["reasoning"],
            "reasoning_b": scores[1]["reasoning"],
        })

    df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 45)
    print("RESUME 6 DIAGNOSTIC")
    print("=" * 45)

    print("\nName comparison:")
    print(df[df["experiment"] == "name_comparison"][
        ["trial", "score_a", "score_b", "signed_gap"]
    ].to_string(index=False))

    print("\nIdentical-resume control:")
    print(df[df["experiment"] == "identical_resume_control"][
        ["trial", "score_a", "score_b", "signed_gap"]
    ].to_string(index=False))

    print(f"\nElapsed: {time.perf_counter() - start:.1f}s")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_diagnostic()
