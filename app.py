
import random
import time

import pandas as pd
import streamlit as st

from counterfactual import create_counterfactual_resume
from validator import validate_resume_pair
from screening import screen_resume, MODEL_NAME

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Counterfactual Resume Auditor",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Counterfactual Resume Auditor")
st.caption("Controlled LLM fairness sensitivity experiments")

st.info(
    "This research prototype measures score sensitivity "
    "to demographic-associated attributes. Score differences "
    "alone do not establish discrimination."
)

# ---------------- INPUTS ----------------

st.header("1. Experiment Configuration")

with st.form("experiment_form"):

    resume = st.text_area(
        "Original Synthetic Resume",
        height=220,
        placeholder="Enter a synthetic resume..."
    )

    job_description = st.text_area(
        "Job Description",
        height=150,
        placeholder="Enter job requirements..."
    )

    col1, col2 = st.columns(2)

    with col1:
        original_name = st.text_input(
            "Original Name",
            placeholder="Rahul Sharma"
        )

    with col2:
        counterfactual_name = st.text_input(
            "Counterfactual Name",
            placeholder="Riya Sharma"
        )

    repetitions = st.number_input(
        "Trials per resume pair",
        min_value=1,
        max_value=5,
        value=3
    )

    submitted = st.form_submit_button(
        "Run Fairness Experiment",
        type="primary"
    )

# ---------------- EXPERIMENT ----------------

if submitted:

    if not all([
        resume.strip(),
        job_description.strip(),
        original_name.strip(),
        counterfactual_name.strip()
    ]):
        st.error("Please complete all fields.")

    elif original_name == counterfactual_name:
        st.error("Candidate names must be different.")

    else:

        try:
            # 1. Generate counterfactual.
            resume_b = create_counterfactual_resume(
                resume,
                original_name,
                counterfactual_name
            )

            # 2. Validate exact equivalence.
            validation = validate_resume_pair(
                resume,
                resume_b,
                original_name,
                counterfactual_name
            )

            if not validation["is_equivalent"]:
                st.error("Resume equivalence validation failed.")
                st.stop()

            st.success("Resume equivalence validation passed.")

            # 3. Show paired resumes.
            with st.expander("View Resume A and Resume B"):
                left, right = st.columns(2)

                left.text_area(
                    "Resume A",
                    resume,
                    height=250,
                    disabled=True,
                    key="resume_original"
                )

                right.text_area(
                    "Resume B",
                    resume_b,
                    height=250,
                    disabled=True,
                    key="resume_counterfactual"
                )

            # 4. Run randomized paired trials.
            st.header("2. Screening Experiment")

            progress = st.progress(0)
            status = st.empty()

            results = []
            total_evaluations = int(repetitions) * 2
            completed = 0

            start_time = time.perf_counter()

            for trial in range(1, int(repetitions) + 1):

                order = ["A", "B"]
                random.shuffle(order)

                evaluations = {}

                for version in order:

                    status.write(
                        f"Trial {trial}/{repetitions}: "
                        f"Evaluating Resume {version}..."
                    )

                    candidate_resume = (
                        resume if version == "A" else resume_b
                    )

                    evaluations[version] = screen_resume(
                        candidate_resume,
                        job_description
                    )

                    completed += 1
                    progress.progress(
                        completed / total_evaluations
                    )

                score_a = evaluations["A"]["score"]
                score_b = evaluations["B"]["score"]

                results.append({
                    "Trial": trial,
                    "Original Score": score_a,
                    "Counterfactual Score": score_b,
                    "Signed Difference (B-A)": score_b - score_a,
                    "Absolute Gap": abs(score_b - score_a),
                    "Original Reasoning": evaluations["A"]["reasoning"],
                    "Counterfactual Reasoning": evaluations["B"]["reasoning"],
                })

            elapsed = time.perf_counter() - start_time

            status.success(
                f"Experiment completed in {elapsed:.1f} seconds."
            )

            # 5. Analyze results.
            df = pd.DataFrame(results)

            mean_a = df["Original Score"].mean()
            mean_b = df["Counterfactual Score"].mean()
            mean_gap = df["Absolute Gap"].mean()
            signed_gap = df["Signed Difference (B-A)"].mean()

            st.header("3. Results")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Original Mean", f"{mean_a:.2f}")
            c2.metric("Counterfactual Mean", f"{mean_b:.2f}")
            c3.metric("Mean Absolute Gap", f"{mean_gap:.2f}")
            c4.metric("Mean Signed Difference", f"{signed_gap:+.2f}")

            st.subheader("Score Comparison")

            st.line_chart(
                df.set_index("Trial")[
                    ["Original Score", "Counterfactual Score"]
                ]
            )

            st.subheader("Signed Score Differences")

            st.bar_chart(
                df.set_index("Trial")[
                    "Signed Difference (B-A)"
                ]
            )

            st.subheader("Trial Results")

            st.dataframe(
                df[
                    [
                        "Trial",
                        "Original Score",
                        "Counterfactual Score",
                        "Signed Difference (B-A)",
                        "Absolute Gap"
                    ]
                ],
                use_container_width=True
            )

            with st.expander("Evaluation Reasoning"):
                for row in results:
                    st.write(f"Trial {row['Trial']}")
                    st.write("Resume A:", row["Original Reasoning"])
                    st.write("Resume B:", row["Counterfactual Reasoning"])
                    st.divider()

            # 6. Download.
            st.header("4. Export Results")

            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="counterfactual_experiment.csv",
                mime="text/csv"
            )

            st.caption(
                f"Model: {MODEL_NAME} | "
                f"Trials: {repetitions} | "
                f"Evaluations: {total_evaluations} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        except Exception as error:
            st.error(f"Experiment failed: {error}")
            st.exception(error)
