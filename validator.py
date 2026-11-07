
def validate_resume_pair(
    original_resume: str,
    counterfactual_resume: str,
    original_name: str,
    counterfactual_name: str
) -> dict:
    """
    Validate whether two resumes differ only by the
    intended candidate-name replacement.
    """

    expected_counterfactual = original_resume.replace(
        original_name,
        counterfactual_name
    )

    is_equivalent = (
        expected_counterfactual == counterfactual_resume
    )

    return {
        "is_equivalent": is_equivalent,
        "professional_content_unchanged": is_equivalent,
        "message": (
            "Resume pair is equivalent."
            if is_equivalent
            else "Unexpected content differences detected."
        )
    }
