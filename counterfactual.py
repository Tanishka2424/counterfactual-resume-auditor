
def create_counterfactual_resume(
    resume: str,
    original_name: str,
    counterfactual_name: str
) -> str:
    """
    Create a counterfactual resume by replacing
    only the candidate's name.
    """

    if not resume.strip():
        raise ValueError("Resume cannot be empty.")

    if not original_name.strip() or not counterfactual_name.strip():
        raise ValueError("Names cannot be empty.")

    if original_name == counterfactual_name:
        raise ValueError("Names must be different.")

    if original_name not in resume:
        raise ValueError("Original name not found in resume.")

    counterfactual_resume = resume.replace(
        original_name,
        counterfactual_name
    )

    return counterfactual_resume
