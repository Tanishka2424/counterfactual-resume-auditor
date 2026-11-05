
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Counterfactual Resume Auditor",
    page_icon="⚖️",
    layout="wide"
)

# Application title
st.title("⚖️ Counterfactual Resume Auditor")

st.caption(
    "Audit LLM resume screening for potential "
    "counterfactual score disparities."
)

st.divider()

st.subheader("Experiment Configuration")

# Resume input
resume_a = st.text_area(
    "Original Resume",
    placeholder="Enter resume content here...",
    height=200
)

# Job description input
job_description = st.text_area(
    "Job Description",
    placeholder="Enter job description here...",
    height=150
)

# Initial button
if st.button("Start Experiment"):
    if not resume_a or not job_description:
        st.warning("Please provide both resume and job description.")
    else:
        st.success("Inputs received successfully!")
