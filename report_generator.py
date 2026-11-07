
from pathlib import Path
import json
from datetime import datetime

import pandas as pd

OUTPUT_DIR = Path("reports/generated")
SUMMARY_PATH = OUTPUT_DIR / "statistical_summary.json"
PAIR_PATH = OUTPUT_DIR / "pair_analysis.csv"
REPORT_PATH = OUTPUT_DIR / "fairness_sensitivity_report.md"

AFFECTED_COLUMNS = [
    "resume_id",
    "mean_score_a",
    "mean_score_b",
    "mean_signed_gap",
    "mean_absolute_gap",
]


def generate_report():
    # Validate required input files
    missing_files = [
        path for path in (SUMMARY_PATH, PAIR_PATH)
        if not path.exists()
    ]

    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            f"Analysis files missing: {missing}. Run analysis.py first."
        )

    # Load analysis outputs
    with SUMMARY_PATH.open(encoding="utf-8") as file:
        summary = json.load(file)

    pairs = pd.read_csv(PAIR_PATH)

    if pairs.empty:
        raise ValueError("pair_analysis.csv contains no resume profiles.")

    required_columns = set(AFFECTED_COLUMNS)
    missing_columns = required_columns - set(pairs.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in pair_analysis.csv: "
            f"{sorted(missing_columns)}"
        )

    # Identify profiles with nonzero score gaps
    affected = pairs.loc[
        pairs["mean_absolute_gap"] > 0,
        AFFECTED_COLUMNS,
    ]

    # Native Pandas formatting; no tabulate dependency
    affected_str = (
        affected.to_string(index=False)
        if not affected.empty
        else "No profiles had nonzero score gaps."
    )

    largest = pairs.loc[pairs["mean_absolute_gap"].idxmax()]

    # Build research report
    report = f"""# Counterfactual Resume Auditor
## Fairness Sensitivity Experiment Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## 1. Executive Summary

This experiment evaluated whether an LLM-based resume screening system
produces different professional qualification scores when only a
gender-associated candidate name is changed.

The experiment used {summary['resume_pairs']} synthetic matched resume
pairs and the Qwen 2.5 1.5B model.

The mean absolute score gap was
**{summary['mean_absolute_gap']:.2f} points**.

{summary['affected_percentage']:.1f}% of resume pairs had a nonzero
score difference. The mean signed score difference was
{summary['mean_signed_gap']:.2f} points, with a 95% confidence interval
of [{summary['ci_95_lower']:.2f}, {summary['ci_95_upper']:.2f}].

The Wilcoxon signed-rank test returned p =
{summary['wilcoxon_pvalue']:.4f}. This result did not meet the
conventional 0.05 significance threshold.

These findings indicate observed name-associated score sensitivity
within this experimental setup. They do not establish real-world
hiring discrimination.

## 2. Research Objective

Assess whether changing a gender-associated candidate name while
holding professional qualifications constant changes an LLM's
resume screening score.

## 3. Experimental Methodology

- **Model:** Qwen 2.5 1.5B
- **Inference platform:** Ollama
- **Profiles:** {summary['resume_pairs']} synthetic resumes
- **Conditions:** Masculine-associated name vs feminine-associated name
- **Trials:** One evaluation per resume version
- **Evaluations:** {summary['resume_pairs'] * 2}
- **Scoring:** Fixed 100-point rubric
- **Ordering:** Randomized within each matched pair
- **Validation:** Resume equivalence checks

The resumes were synthetically generated. Name substitutions were
intended to preserve professional qualifications.

## 4. Quantitative Results

| Metric | Result |
|---|---:|
| Resume pairs | {summary['resume_pairs']} |
| Mean score A | {summary['mean_score_a']:.2f} |
| Mean score B | {summary['mean_score_b']:.2f} |
| Mean signed difference (B-A) | {summary['mean_signed_gap']:.2f} |
| Mean absolute gap | {summary['mean_absolute_gap']:.2f} |
| Standard deviation of differences | {summary['std_signed_gap']:.2f} |
| 95% CI lower | {summary['ci_95_lower']:.2f} |
| 95% CI upper | {summary['ci_95_upper']:.2f} |
| Affected profiles | {summary['affected_pairs']} |
| Affected percentage | {summary['affected_percentage']:.1f}% |
| Positive differences | {summary['positive_gap_pairs']} |
| Negative differences | {summary['negative_gap_pairs']} |
| Zero differences | {summary['zero_gap_pairs']} |
| Wilcoxon statistic | {summary['wilcoxon_statistic']:.2f} |
| Wilcoxon p-value | {summary['wilcoxon_pvalue']:.4f} |
| Cohen's dz | {summary['cohens_dz']:.3f} |

## 5. Score Discrepancies

The largest observed score gap occurred in resume profile
{summary['largest_gap_resume_id']}, with a difference of
{summary['largest_gap']:.2f} points.

### Profiles with Nonzero Score Gaps

{affected_str}

## 6. Interpretation

The experiment observed score changes for a subset of matched
synthetic resumes after candidate-name substitution.

The average signed difference was negative, indicating that condition
B received a lower average score than condition A in this dataset.

However, the confidence interval includes zero and the Wilcoxon
test was not statistically significant at the 0.05 threshold.

These results should be interpreted as preliminary evidence of
model sensitivity rather than definitive evidence of demographic bias.

## 7. Limitations

1. Only {summary['resume_pairs']} synthetic resume profiles were evaluated.
2. Each resume condition received one evaluation.
3. Name associations are imperfect proxies for demographic identity.
4. Results concern one model and one prompt configuration.
5. Synthetic resumes may not represent real applicant populations.
6. A score difference does not establish discriminatory intent.
7. Statistical inference is limited to variation across the sampled
   synthetic profiles.
8. Model behavior may change across versions and configurations.

## 8. Responsible Use

This prototype is intended for auditing and research.

It should not be used to automatically rank, reject, or select
real job applicants.

Human oversight, transparent criteria, representative datasets,
and independent validation are necessary before any employment-related
application.

## 9. Generated Artifacts

- `batch_results.csv`: raw screening results
- `batch_summary.csv`: per-profile screening summary
- `pair_analysis.csv`: paired score analysis
- `statistical_summary.csv`: statistical metrics
- `statistical_summary.json`: machine-readable metrics
- `score_comparison.png`: paired score visualization
- `score_differences.png`: signed difference visualization
- `gap_distribution.png`: score difference distribution

---

*Counterfactual Resume Auditor — Controlled LLM Fairness Sensitivity Experiment*
"""

    # Write report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("=" * 45)
    print("RESEARCH REPORT GENERATED")
    print("=" * 45)
    print(f"Report: {REPORT_PATH.resolve()}")
    print(f"Affected profiles: {len(affected)}")
    print(f"Total profiles: {len(pairs)}")


if __name__ == "__main__":
    generate_report()
