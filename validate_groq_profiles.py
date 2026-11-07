import pandas as pd
from pathlib import Path

INPUT = Path("groq_profiles.csv")
REQUIRED = ["profile_id", "resume", "original_name", "counterfactual_name"]

if not INPUT.exists():
    raise SystemExit("ERROR: groq_profiles.csv not found. Put it in the project root.")

df = pd.read_csv(INPUT, dtype=str).fillna("")
errors = []

missing_cols = [c for c in REQUIRED if c not in df.columns]
if missing_cols:
    raise SystemExit(f"ERROR: Missing columns: {missing_cols}. Required: {REQUIRED}")

if len(df) != 30:
    errors.append(f"Expected exactly 30 profiles; found {len(df)}.")

if df["profile_id"].duplicated().any():
    dupes = df.loc[df["profile_id"].duplicated(), "profile_id"].tolist()
    errors.append(f"Duplicate profile_id values: {dupes}")

for i, row in df.iterrows():
    pid = row["profile_id"] or f"row {i+2}"
    resume = row["resume"]
    original = row["original_name"]
    counter = row["counterfactual_name"]

    for col, val in [("resume", resume), ("original_name", original), ("counterfactual_name", counter)]:
        if not val.strip():
            errors.append(f"{pid}: {col} is blank.")

    if original and counter and original == counter:
        errors.append(f"{pid}: original_name and counterfactual_name are identical.")

    if original and resume.count(original) != 1:
        errors.append(f"{pid}: original_name must appear exactly once in resume (found {resume.count(original)}).")

    if counter and counter in resume:
        errors.append(f"{pid}: counterfactual_name already appears in the original resume.")

    if original and counter and original in resume:
        transformed = resume.replace(original, counter)
        if original in transformed:
            errors.append(f"{pid}: replacement check failed.")

if errors:
    print("VALIDATION FAILED:")
    for e in errors:
        print(" -", e)
    raise SystemExit(1)

print("VALIDATION PASSED")
print(f"Profiles: {len(df)}")
print("Unique profile IDs: yes")
print("Original names occur exactly once in each resume: yes")
print("Counterfactual names do not already appear in original resumes: yes")
print("Planned calls: 30 profiles x 3 trials x 2 conditions = 180")
