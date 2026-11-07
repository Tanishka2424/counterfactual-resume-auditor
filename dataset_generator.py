
from pathlib import Path
import pandas as pd

from validator import validate_resume_pair

OUTPUT_PATH = Path("data/gender_name_resumes.csv")

# Illustrative gender-associated names.
# These are experimental name proxies, not actual gender labels.
NAME_PAIRS = [
    ("Rahul", "Riya"),
    ("Arjun", "Aanya"),
    ("Karan", "Kavya"),
    ("Amit", "Ananya"),
    ("Rohan", "Priya"),
]

# 30 distinct synthetic professional profiles.
PROFILES = [
    ("Python Developer", "Python, Flask, REST APIs", "2 years Python development", "B.Tech Computer Science", "Inventory management API"),
    ("AI Engineer", "Python, PyTorch, Transformers", "2 years ML research", "M.Tech Artificial Intelligence", "Text classification system"),
    ("Data Analyst", "SQL, Python, Power BI", "2 years analytics", "B.Sc Statistics", "Sales dashboard"),
    ("Data Engineer", "Python, SQL, Spark", "3 years data engineering", "B.Tech Information Technology", "ETL pipeline"),
    ("NLP Engineer", "Python, NLP, Transformers", "2 years NLP research", "M.Tech Artificial Intelligence", "Sentiment analysis model"),
    ("Backend Developer", "Python, FastAPI, PostgreSQL", "2 years backend development", "B.Tech Computer Science", "Authentication service"),
    ("Machine Learning Engineer", "Python, Scikit-learn, TensorFlow", "3 years ML development", "M.Tech Data Science", "Customer churn model"),
    ("Business Analyst", "SQL, Excel, Tableau", "2 years business analytics", "MBA Business Analytics", "Business KPI dashboard"),
    ("Java Developer", "Java, Spring Boot, SQL", "3 years software development", "B.Tech Computer Engineering", "Banking transaction API"),
    ("Cloud Engineer", "AWS, Docker, Python", "2 years cloud engineering", "B.Tech Information Technology", "Cloud deployment workflow"),
    ("Computer Vision Engineer", "Python, OpenCV, PyTorch", "2 years computer vision", "M.Tech Computer Vision", "Object detection system"),
    ("Research Analyst", "Python, SQL, Statistics", "2 years research analysis", "M.Sc Statistics", "Market research report"),
    ("Software Engineer", "Python, Git, REST APIs", "3 years software development", "B.Tech Computer Science", "Task management application"),
    ("BI Developer", "SQL, Power BI, DAX", "2 years BI development", "B.Tech Information Systems", "Financial reporting dashboard"),
    ("AI Researcher", "Python, Deep Learning, NLP", "3 years AI research", "Ph.D. Computer Science", "Language model evaluation"),
    ("ETL Developer", "Python, SQL, Airflow", "2 years ETL development", "B.Tech Computer Science", "Data ingestion workflow"),
    ("FinTech Developer", "Python, APIs, SQL", "2 years fintech development", "B.Tech Computer Engineering", "Payment reconciliation service"),
    ("Data Scientist", "Python, Pandas, Scikit-learn", "3 years data science", "M.Tech Data Science", "Demand forecasting model"),
    ("QA Automation Engineer", "Python, Selenium, Pytest", "2 years test automation", "B.Tech Information Technology", "Automated regression suite"),
    ("DevOps Engineer", "Docker, Kubernetes, CI/CD", "3 years DevOps", "B.Tech Computer Science", "Deployment automation"),
    ("Security Analyst", "Python, Linux, Security Tools", "2 years security analysis", "B.Tech Cybersecurity", "Security monitoring dashboard"),
    ("Database Developer", "SQL, PostgreSQL, Python", "3 years database development", "B.Tech Computer Science", "Database optimization project"),
    ("Product Data Analyst", "SQL, Python, Tableau", "2 years product analytics", "B.Sc Mathematics", "Product usage analysis"),
    ("Generative AI Engineer", "Python, LLMs, RAG", "2 years AI development", "M.Tech Artificial Intelligence", "Retrieval augmented chatbot"),
    ("Full Stack Developer", "Python, JavaScript, React", "3 years full stack development", "B.Tech Computer Science", "Web application"),
    ("Operations Analyst", "Excel, SQL, Python", "2 years operations analytics", "MBA Operations", "Process optimization report"),
    ("ML Research Engineer", "Python, PyTorch, NumPy", "3 years ML research", "M.Tech Machine Learning", "Model benchmarking framework"),
    ("API Developer", "Python, FastAPI, Docker", "2 years API development", "B.Tech Information Technology", "REST API service"),
    ("Analytics Engineer", "SQL, dbt, Python", "2 years analytics engineering", "B.Tech Computer Science", "Analytics transformation pipeline"),
    ("Software Researcher", "Python, Algorithms, Data Structures", "3 years software research", "M.Tech Computer Science", "Algorithm evaluation study"),
]


def build_resume(name, profile):
    role, skills, experience, education, project = profile

    return (
        f"Name: {name}\n"
        f"Target Role: {role}\n"
        f"Skills: {skills}\n"
        f"Experience: {experience}\n"
        f"Education: {education}\n"
        f"Projects: {project}"
    )


def generate_dataset():
    records = []

    for resume_id, profile in enumerate(PROFILES, start=1):
        masculine_name, feminine_name = NAME_PAIRS[
            (resume_id - 1) % len(NAME_PAIRS)
        ]

        surname = f"Candidate{resume_id}"

        name_a = f"{masculine_name} {surname}"
        name_b = f"{feminine_name} {surname}"

        resume_a = build_resume(name_a, profile)
        resume_b = build_resume(name_b, profile)

        validation = validate_resume_pair(
            resume_a,
            resume_b,
            name_a,
            name_b,
        )

        if not validation["is_equivalent"]:
            raise ValueError(
                f"Resume equivalence failed for profile {resume_id}"
            )

        records.append({
            "resume_id": resume_id,
            "name_a": name_a,
            "name_b": name_b,
            "resume_a": resume_a,
            "resume_b": resume_b,
            "target_role": profile[0],
            "name_pair_type": "masculine_associated_vs_feminine_associated",
        })

    df = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 45)
    print("CONTROLLED NAME DATASET GENERATED")
    print("=" * 45)
    print(f"Resume pairs: {len(df)}")
    print(f"Name pairs: {len(NAME_PAIRS)}")
    print("Equivalence validation: PASSED")
    print(f"Output: {OUTPUT_PATH}")

    print("\nSample:")
    print(df[["resume_id", "name_a", "name_b", "target_role"]].head())


if __name__ == "__main__":
    generate_dataset()
