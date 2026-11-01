"""
build_dataset.py  -  combines EVERYTHING into one file:  data/FutureProof_Dataset.xlsx

Order:  1) python prepare_data.py    (raw files -> clean CSVs)
        2) python build_dataset.py   (clean CSVs + earlier task file -> one workbook)

The app then reads only FutureProof_Dataset.xlsx.
"""

import pandas as pd

D = "data"

# ---- tasks: the 535-job dataset + the team's own earlier task file
big = pd.read_csv(f"{D}/jobs_tasks.csv")
big["source"] = "jobs-535"

old = pd.read_csv(f"{D}/task_automation_data.csv")
team = pd.DataFrame({
    "job_name": old["category"] + " (team-curated)",
    "category": "Team-Curated Profiles",
    "task_name": old["task"],
    "task_ai_probability_pct": (old["automation_risk"] * 100).round(0),
    "source": "team-curated",
    "human_centric_score": old["human_centric_score"],
    "skill_type": old["skill_type"],
})
team["overall_risk_pct"] = team.groupby("job_name")["task_ai_probability_pct"].transform("mean").round(0)

tasks = pd.concat([big, team], ignore_index=True)
tasks = tasks[["job_name", "category", "overall_risk_pct", "task_name",
               "task_ai_probability_pct", "source", "human_centric_score", "skill_type"]]

# ---- other sheets
sheets = {
    "tasks": tasks,
    "occupation_benchmark": pd.read_csv(f"{D}/occupation_benchmark.csv"),
    "emerging_skills": pd.read_csv(f"{D}/emerging_skills_clean.csv"),
    "future_job_demand": pd.read_csv(f"{D}/future_job_demand_clean.csv"),
    "skill_resources": pd.read_csv(f"{D}/skill_resources.csv"),
}

readme = pd.DataFrame({
    "Sheet": ["tasks", "occupation_benchmark", "emerging_skills", "future_job_demand", "skill_resources"],
    "What it holds": [
        "Task-level AI probability per job (535 jobs) plus the team's own 54 curated tasks (12 profiles)",
        "Official occupations (879): WillRobotsTakeMyJob risk merged with OECD AI exposure by SOC code",
        "WEF Future of Jobs 2025: skills expected to rise or fall in importance (2025-2030)",
        "WEF Future of Jobs 2025: fastest-growing and fastest-declining occupations (2025-2030)",
        "Free learning resources mapped to each rising skill",
    ],
    "Source": [
        "jobs-535 dataset (via Kaggle) + team-curated",
        "WillRobotsTakeMyJob; OECD (2026) AI Papers No. 59, CC BY 4.0 (collected via Kaggle)",
        "World Economic Forum, Future of Jobs Report 2025 (collected via Kaggle)",
        "World Economic Forum, Future of Jobs Report 2025 (collected via Kaggle)",
        "Public free-course providers (Google, Coursera, edX, Kaggle, etc.)",
    ],
})

with pd.ExcelWriter(f"{D}/FutureProof_Dataset.xlsx", engine="openpyxl") as xw:
    readme.to_excel(xw, sheet_name="README", index=False)
    for name, df in sheets.items():
        df.to_excel(xw, sheet_name=name, index=False)

print("Built data/FutureProof_Dataset.xlsx")
for name, df in sheets.items():
    print(f"  {name:22s} {len(df):5d} rows")
