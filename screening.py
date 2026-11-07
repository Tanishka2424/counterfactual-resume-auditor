
import json
import ollama

MODEL_NAME = "qwen2.5:1.5b"
REQUEST_TIMEOUT = 120
MAX_ATTEMPTS = 3

client = ollama.Client(timeout=REQUEST_TIMEOUT)

RUBRIC = {
    "skills_score": 40,
    "experience_score": 30,
    "education_score": 20,
    "relevance_score": 10,
}


def screen_resume(resume: str, job_description: str) -> dict:
    if not resume.strip() or not job_description.strip():
        raise ValueError("Resume and job description are required.")

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
  "score": 0,
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

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={
                    "temperature": 0,
                    "num_predict": 320,
                },
                keep_alive="5m",
            )

            result = json.loads(response["message"]["content"])

            for category, maximum in RUBRIC.items():
                value = result.get(category)

                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not 0 <= value <= maximum
                ):
                    raise ValueError(f"Invalid score: {category}")

            result["score"] = sum(result[key] for key in RUBRIC)

            if not isinstance(result.get("reasoning"), str):
                result["reasoning"] = ""

            return result

        except (json.JSONDecodeError, ValueError, KeyError) as error:
            last_error = error
            print(
                f"Invalid model response: attempt "
                f"{attempt}/{MAX_ATTEMPTS}: {error}",
                flush=True,
            )

    raise RuntimeError(
        f"Screening failed after {MAX_ATTEMPTS} attempts: {last_error}"
    )
