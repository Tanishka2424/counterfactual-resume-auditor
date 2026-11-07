
# Counterfactual Resume Auditor

### Controlled LLM Fairness Sensitivity Experiment

Counterfactual Resume Auditor is a research-oriented project that investigates whether an LLM-based resume screening system produces different professional qualification scores when only a candidate's gender-associated name is changed.

The project uses controlled synthetic resume pairs, automated equivalence validation, LLM-based scoring, paired statistical analysis, and report generation to examine potential name-associated score sensitivity.

> **Research disclaimer:** Observed score differences indicate model sensitivity under the experimental conditions. They do not independently establish real-world hiring discrimination.

---

## Research Objective

Evaluate whether changing a gender-associated candidate name while preserving professional qualifications influences an LLM's resume screening score.

### Research Question

Does an LLM-based resume screening system assign different qualification scores to otherwise identical synthetic resumes when the candidate name changes between masculine-associated and feminine-associated names?

---

## Features

- **Counterfactual Resume Generation:** Creates matched resume variants by replacing candidate names.
- **Equivalence Validation:** Verifies that professional resume content remains unchanged.
- **LLM-Based Screening:** Evaluates resumes using a fixed 100-point qualification rubric.
- **Controlled Experiments:** Screens matched resume pairs under consistent evaluation conditions.
- **Batch Processing:** Runs experiments across multiple synthetic profiles.
- **Checkpointing:** Preserves completed experiments for recovery after interruptions.
- **Statistical Analysis:** Calculates paired score differences, confidence intervals, Wilcoxon signed-rank tests, and effect sizes.
- **Visual Analytics:** Generates score comparisons and score-gap distributions.
- **Research Report:** Automatically generates a Markdown report containing results, interpretation, and limitations.
- **Interactive Interface:** Streamlit-based application for conducting individual counterfactual experiments.

---

## System Architecture

```text
Synthetic Resume Dataset
          |
          v
Counterfactual Resume Generator
          |
          v
Resume Equivalence Validator
          |
          v
LLM Screening Engine
          |
          v
Batch Experiment Runner
          |
          v
Experiment Results (CSV)
          |
          v
Statistical Analysis
          |
          +------> Statistical Summary
          |
          +------> Visualizations
          |
          v
Research Report Generator
          |
          v
Fairness Sensitivity Report
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM | Qwen 2.5 1.5B |
| Inference Runtime | Ollama |
| Data Processing | Pandas, NumPy |
| Statistical Analysis | SciPy |
| Visualization | Matplotlib |
| User Interface | Streamlit |
| Experiment Storage | CSV, JSON |
| Version Control | Git, GitHub |

---

## Experimental Methodology

### 1. Synthetic Resume Dataset

The experiment uses 30 synthetic resume profiles covering different technical and professional skill combinations.

Each profile has two versions:

- Condition A: Masculine-associated candidate name
- Condition B: Feminine-associated candidate name

The professional qualifications, skills, experience, and education are held constant.

### 2. Counterfactual Generation

Candidate names are substituted while preserving the remaining resume content.

### 3. Equivalence Validation

Each resume pair is validated to ensure that the intended name substitution is the only content change.

### 4. LLM Screening

Both resume versions are evaluated using the same model, job description, scoring criteria, and prompt configuration.

### 5. Scoring Rubric

| Category | Maximum Score |
|---|---:|
| Skills | 40 |
| Experience | 30 |
| Education | 20 |
| Job Relevance | 10 |
| **Total** | **100** |

The model is instructed to evaluate professional qualifications and avoid demographic characteristics.

### 6. Paired Statistical Analysis

The experiment calculates:

- Mean score for each condition
- Signed score difference (B − A)
- Absolute score difference
- Standard deviation of paired differences
- 95% confidence interval
- Wilcoxon signed-rank test
- Cohen's dz effect size
- Number and percentage of affected profiles

---

## Experimental Results

### Baseline Experiment

| Metric | Result |
|---|---:|
| Synthetic resume pairs | 30 |
| Total screening evaluations | 60 |
| Mean score A | 58.33 |
| Mean score B | 52.33 |
| Mean signed difference | -6.00 |
| Mean absolute score gap | 9.67 |
| Profiles with nonzero gaps | 9 / 30 |
| Affected profiles | 30% |
| Positive differences | 3 |
| Negative differences | 6 |
| Zero differences | 21 |
| 95% confidence interval | [-13.54, 1.54] |
| Wilcoxon p-value | 0.1379 |
| Cohen's dz | -0.297 |

### Interpretation

The experiment observed nonzero score differences in 9 of 30 matched synthetic resume pairs.

Condition B received a lower average score than condition A in this experimental sample.

However:

- The confidence interval includes zero.
- The Wilcoxon test did not meet the conventional 0.05 significance threshold.
- The experiment uses a relatively small synthetic dataset.
- Each resume condition received one evaluation.

These findings represent preliminary evidence of name-associated model sensitivity, not conclusive evidence of demographic discrimination.

---

## Project Structure

```text
counterfactual-resume-auditor/
│
├── app.py
├── counterfactual.py
├── validator.py
├── screening.py
├── dataset_generator.py
├── experiment_runner.py
├── analysis.py
├── report_generator.py
│
├── data/
│   └── gender_name_resumes.csv
│
├── reports/
│   └── generated/
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.12 recommended
- Git
- Ollama

### 1. Clone the repository

```bash
git clone https://github.com/Tanishka2424/counterfactual-resume-auditor.git
cd counterfactual-resume-auditor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install Ollama

Download Ollama:

https://ollama.com/download

Pull the required model:

```bash
ollama pull qwen2.5:1.5b
```

Ensure Ollama is running before executing experiments.

---

## Usage

### Generate Synthetic Resumes

```bash
python dataset_generator.py
```

### Run Batch Experiment

```bash
python experiment_runner.py
```

The experiment runner generates paired screening results and checkpoint files.

### Perform Statistical Analysis

```bash
python analysis.py
```

This generates statistical summaries and visualizations.

### Generate Research Report

```bash
python report_generator.py
```

Output:

```text
reports/generated/fairness_sensitivity_report.md
```

### Launch Interactive Application

```bash
streamlit run app.py
```

---

## Generated Artifacts

| Artifact | Description |
|---|---|
| `batch_results.csv` | Individual screening evaluations |
| `batch_summary.csv` | Per-profile score summaries |
| `batch_checkpoint.csv` | Experiment recovery checkpoint |
| `pair_analysis.csv` | Paired score-gap analysis |
| `statistical_summary.csv` | Statistical metrics |
| `statistical_summary.json` | Machine-readable statistical results |
| `score_comparison.png` | Score comparison visualization |
| `score_differences.png` | Signed score differences |
| `gap_distribution.png` | Distribution of score gaps |
| `fairness_sensitivity_report.md` | Automated research report |

Generated experiment outputs are excluded from version control by default.

---

## Limitations

1. The dataset contains only 30 synthetic resume profiles.
2. Each resume condition was evaluated once in the reported experiment.
3. Gender-associated names are imperfect demographic proxies.
4. The findings concern one model and one prompt configuration.
5. Synthetic profiles may not represent real applicant populations.
6. Model outputs may contain stochastic or formatting variation.
7. Statistical inference is limited to the sampled synthetic profiles.
8. Score sensitivity does not establish discriminatory intent or real-world hiring outcomes.

---

## Responsible Use

This project is intended for research, experimentation, and auditing.

It is not designed for automated hiring decisions, candidate rejection, or applicant ranking.

Employment-related AI systems require appropriate human oversight, transparent evaluation criteria, representative validation data, and independent assessment.

---

## Future Improvements

- Increase the synthetic resume sample size.
- Introduce repeated trials per resume condition.
- Evaluate additional open-source LLMs.
- Compare different prompt configurations.
- Investigate name associations across additional demographic dimensions.
- Add automated experiment reproducibility checks.
- Evaluate scoring consistency and inter-run variability.
- Improve confidence interval estimation and robustness analysis.
- Add automated testing and continuous integration.

---

## Author

**Tanishka Uttarkar**

B.Tech Computer Science — Artificial Intelligence & Analytics

GitHub: [Tanishka2424](https://github.com/Tanishka2424)

---

*Counterfactual Resume Auditor — Controlled LLM Fairness Sensitivity Experiment*
