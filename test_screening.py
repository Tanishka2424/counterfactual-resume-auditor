
from screening import screen_resume
import time

resume = """
Rahul Sharma
Python Developer
Skills: Python, SQL, FastAPI
Experience: 2 years software development
Education: B.Tech Computer Science
"""

job = """
Python Developer required.
Skills: Python, SQL, API development.
Experience: 1-3 years.
Education: Computer Science degree.
"""

print("Testing evaluator...", flush=True)
start = time.perf_counter()

try:
    result = screen_resume(resume, job)

    print("\nScore:", result["score"])
    print("Breakdown:", {
        "skills": result["skills_score"],
        "experience": result["experience_score"],
        "education": result["education_score"],
        "relevance": result["relevance_score"],
    })
    print("Reasoning:", result["reasoning"])

except Exception as error:
    print(f"\nEvaluation failed: {error}")

finally:
    print(f"\nTime: {time.perf_counter() - start:.2f}s")
