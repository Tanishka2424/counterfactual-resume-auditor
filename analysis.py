
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

RESULTS_PATH = Path("reports/generated/batch_results.csv")
OUTPUT_DIR = Path("reports/generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Results not found: {RESULTS_PATH}")

    df = pd.read_csv(RESULTS_PATH)

    required = {
        "resume_id",
        "score_a",
        "score_b",
        "signed_gap",
        "absolute_gap",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Aggregate repeated trials if present.
    paired = (
        df.groupby("resume_id", as_index=False)
        .agg(
            mean_score_a=("score_a", "mean"),
            mean_score_b=("score_b", "mean"),
            mean_signed_gap=("signed_gap", "mean"),
            mean_absolute_gap=("absolute_gap", "mean"),
        )
    )

    gaps = paired["mean_signed_gap"].to_numpy(dtype=float)
    absolute_gaps = paired["mean_absolute_gap"].to_numpy(dtype=float)

    n = len(gaps)

    mean_gap = float(np.mean(gaps))
    mean_absolute_gap = float(np.mean(absolute_gaps))
    std_gap = float(np.std(gaps, ddof=1)) if n > 1 else 0.0

    # Confidence interval for mean paired difference.
    if n > 1:
        standard_error = stats.sem(gaps)
        ci_low, ci_high = stats.t.interval(
            confidence=0.95,
            df=n - 1,
            loc=mean_gap,
            scale=standard_error,
        )
    else:
        ci_low = ci_high = np.nan

    # Directional counts.
    positive = int((gaps > 0).sum())
    negative = int((gaps < 0).sum())
    zero = int((gaps == 0).sum())

    affected = int((absolute_gaps > 0).sum())
    affected_percentage = affected / n * 100 if n else 0.0

    # Paired Wilcoxon signed-rank test.
    nonzero_gaps = gaps[gaps != 0]

    if len(nonzero_gaps) > 0:
        wilcoxon_result = stats.wilcoxon(
            gaps,
            zero_method="wilcox",
            alternative="two-sided",
        )
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_pvalue = float(wilcoxon_result.pvalue)
    else:
        wilcoxon_statistic = 0.0
        wilcoxon_pvalue = 1.0

    # Cohen's dz for paired differences.
    if std_gap > 0:
        cohens_dz = mean_gap / std_gap
    else:
        cohens_dz = 0.0

    # Identify largest score gap.
    largest_index = paired["mean_absolute_gap"].idxmax()
    largest_row = paired.loc[largest_index]

    summary = {
        "resume_pairs": int(n),
        "mean_score_a": float(paired["mean_score_a"].mean()),
        "mean_score_b": float(paired["mean_score_b"].mean()),
        "mean_signed_gap": mean_gap,
        "mean_absolute_gap": mean_absolute_gap,
        "std_signed_gap": std_gap,
        "ci_95_lower": float(ci_low),
        "ci_95_upper": float(ci_high),
        "affected_pairs": affected,
        "affected_percentage": affected_percentage,
        "positive_gap_pairs": positive,
        "negative_gap_pairs": negative,
        "zero_gap_pairs": zero,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "cohens_dz": float(cohens_dz),
        "largest_gap": float(largest_row["mean_absolute_gap"]),
        "largest_gap_resume_id": int(largest_row["resume_id"]),
    }

    # Save per-profile analysis.
    paired.to_csv(OUTPUT_DIR / "pair_analysis.csv", index=False)

    # Save statistical summary.
    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "statistical_summary.csv",
        index=False,
    )

    with open(OUTPUT_DIR / "statistical_summary.json", "w") as file:
        json.dump(summary, file, indent=4)

    # Chart 1: Score comparison.
    plt.figure(figsize=(12, 6))
    plt.plot(
        paired["resume_id"],
        paired["mean_score_a"],
        marker="o",
        label="Masculine-associated names",
    )
    plt.plot(
        paired["resume_id"],
        paired["mean_score_b"],
        marker="o",
        label="Feminine-associated names",
    )
    plt.xlabel("Resume ID")
    plt.ylabel("Mean Screening Score")
    plt.title("Counterfactual Resume Score Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "score_comparison.png",
        dpi=200,
    )
    plt.close()

    # Chart 2: Signed score differences.
    plt.figure(figsize=(12, 6))
    plt.bar(
        paired["resume_id"],
        paired["mean_signed_gap"],
    )
    plt.axhline(0, linewidth=1)
    plt.xlabel("Resume ID")
    plt.ylabel("Score Difference (B - A)")
    plt.title("Counterfactual Score Differences")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "score_differences.png",
        dpi=200,
    )
    plt.close()

    # Chart 3: Distribution of signed gaps.
    plt.figure(figsize=(10, 6))
    plt.hist(gaps, bins="auto", edgecolor="black")
    plt.axvline(0, linewidth=1)
    plt.xlabel("Score Difference (B - A)")
    plt.ylabel("Number of Resume Pairs")
    plt.title("Distribution of Counterfactual Score Differences")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "gap_distribution.png",
        dpi=200,
    )
    plt.close()

    print("\n" + "=" * 50)
    print("COUNTERFACTUAL FAIRNESS SENSITIVITY ANALYSIS")
    print("=" * 50)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nGenerated files:")
    print("- pair_analysis.csv")
    print("- statistical_summary.csv")
    print("- statistical_summary.json")
    print("- score_comparison.png")
    print("- score_differences.png")
    print("- gap_distribution.png")

    return paired, summary


if __name__ == "__main__":
    analyze_results()
