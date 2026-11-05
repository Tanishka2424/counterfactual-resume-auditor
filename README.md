
# Counterfactual Resume Auditor

An AI-powered tool to evaluate whether an LLM-based resume screener produces different scores when irrelevant demographic attributes change.

## Problem Statement

LLM-based resume screening systems may exhibit sensitivity to demographic attributes such as names, gender-associated signals, or hometowns.

This project uses controlled counterfactual experiments to identify potential score disparities.

## Core Features

- Generate counterfactual resume pairs
- Screen resumes using a hosted Hugging Face model
- Calculate score differences
- Validate professional content equivalence
- Flag potential fairness concerns
- Visualize screening results

## Tech Stack

- Python
- Streamlit
- Hugging Face Inference Providers
- Pandas
- Sentence Transformers

## Project Status

Under development.

## Disclaimer

This tool identifies potential score disparities. A score difference alone does not establish discrimination or prove that a model is biased.
