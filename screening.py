
import os
import json
import time
from functools import lru_cache

from openai import OpenAI

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").lower()

REQUEST_TIMEOUT = 120
MAX_ATTEMPTS = 3

RUBRIC = {
    "skills_score": 40,
    "experience_score": 30,
    "education_score": 20,
    "relevance_score": 10,
}

# ============================================================
# CLIENT CONFIGURATION
# ============================================================

@lru_cache(maxsize=1)
def get_client():
    """Create an inference client based on the selected mode."""

    if INFERENCE_MODE == "cloud":
        api_key = os.getenv("OLLAMA_API_KEY")

        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("OLLAMA_API_KEY")
            except Exception:
                api_key = None

        if not api_key:
            raise RuntimeError(
                "OLLAMA_API_KEY is missing. "
                "Configure it in Streamlit Secrets."
            )

        return OpenAI(
            base_url="https://ollama.com/v1",
            api_key=api_key,
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
        )

    if INFERENCE_MODE == "local":
        return OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
        )

    raise ValueError(
        "INFERENCE_MODE must be 'local' or 'cloud'."
    )


# ============================================================
# SCREENING
# ============================================================

def screen_resume(resume: str, job_description: str) -> dict:
    """Screen a resume using the configured Ollama inference mode."""

    if not isinstance(resume, str) or not resume.strip():
        raise ValueError("Resume cannot be empty.")

    if not isinstance(job_description, str) or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    prompt = f"""
Evaluate professional qualifications only.
Ignore candidate name and demographic characteristics.

Score each category within its maximum:
Skills: 40
Experience: 30
Education: 20
Job relevance: 10

Use evidence explicitly present in the resume.
Do not invent qualifications.
Return valid JSON only.
Keep reasoning under 40 words.

Required JSON:
{{
  "skills_score": 0,
  "experience_score": 0,
  "education_score": 0,
  "relevance_score": 0,
  "reasoning": "Brief evidence-based explanation"
}}

JOB:
{job_description}

RESUME:
{resume}
"""

    client = get_client()
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a structured resume screening "
                            "evaluator. Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=320,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Model returned an empty response.")

            result = json.loads(content)

            # Validate individual rubric scores
            for category, maximum in RUBRIC.items():
                value = result.get(category)

                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not 0 <= value <= maximum
                ):
                    raise ValueError(
                        f"Invalid score for {category}: {value}"
                    )

            # Compute total locally; never trust model total
            result["score"] = sum(
                result[category] for category in RUBRIC
            )

            reasoning = result.get("reasoning", "")
            result["reasoning"] = (
                reasoning if isinstance(reasoning, str) else ""
            )

            return result

        except Exception as error:
            last_error = error

            if attempt < MAX_ATTEMPTS:
                time.sleep(attempt * 2)

    raise RuntimeError(
        f"Screening failed after {MAX_ATTEMPTS} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


# ============================================================
# CONFIGURATION HELPER
# ============================================================

def get_screening_config() -> dict:
    """Return safe configuration details without exposing secrets."""

    return {
        "mode": INFERENCE_MODE,
        "model": MODEL_NAME,
        "provider": (
            "Ollama Cloud"
            if INFERENCE_MODE == "cloud"
            else "Local Ollama"
        ),
        "api_key_configured": (
            bool(os.getenv("OLLAMA_API_KEY"))
            if INFERENCE_MODE == "cloud"
            else None
        ),
    }
