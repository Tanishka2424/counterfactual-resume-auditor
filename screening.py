
import os
import json
import time
from functools import lru_cache

import streamlit as st
from openai import OpenAI

# ============================================================
# CONFIGURATION
# ============================================================

REQUEST_TIMEOUT = 90
MAX_ATTEMPTS = 3

RUBRIC = {
    "skills_score": 40,
    "experience_score": 30,
    "education_score": 20,
    "relevance_score": 10,
}


def get_setting(key, default=None):
    """Read environment variables first, then Streamlit Secrets."""
    value = os.getenv(key)

    if value:
        return value

    try:
        value = st.secrets.get(key)
        if value:
            return value
    except Exception:
        pass

    return default


MODEL_NAME = get_setting(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

# ============================================================
# GROQ CLIENT
# ============================================================

@lru_cache(maxsize=1)
def get_client():
    api_key = get_setting("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Secrets."
        )

    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


# ============================================================
# SCREENING
# ============================================================

def screen_resume(resume: str, job_description: str) -> dict:
    """Evaluate resume qualifications using Groq."""

    if not isinstance(resume, str) or not resume.strip():
        raise ValueError("Resume cannot be empty.")

    if (
        not isinstance(job_description, str)
        or not job_description.strip()
    ):
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
                            "You are a structured resume screening evaluator. "
                            "Return ONLY one valid JSON object. "
                            "Do not include markdown, code fences, or text "
                            "outside JSON. All scores must be numeric."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=320,
            )

            content = response.choices[0].message.content
  
            if not content:
                raise ValueError("Model returned an empty response.")

            content = response.choices[0].message.content.strip()

            # Remove markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0].strip()

            # Extract JSON object
            start = content.find("{")
            end = content.rfind("}")

            if start == -1 or end == -1:
                raise ValueError(
                    f"Model did not return valid JSON: {content}"
                )

            result = json.loads(content[start:end + 1])

            # Validate individual scores
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

            # Compute total locally
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

            # Avoid retrying authentication, model, or billing errors
            status_code = getattr(error, "status_code", None)

            if status_code in (400, 401, 402, 403, 404):
                break

            if attempt < MAX_ATTEMPTS:
                time.sleep(attempt * 2)

    raise RuntimeError(
        f"Groq screening failed: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


# ============================================================
# CONFIGURATION DISPLAY
# ============================================================

def get_screening_config():
    return {
        "provider": "Groq",
        "model": MODEL_NAME,
        "api_key_configured": bool(
            get_setting("GROQ_API_KEY")
        ),
    }
