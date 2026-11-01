# FutureProof AI - final version (single dataset file)

## On the practical day (2 commands)
1. python -m pip install -r requirements.txt     (only if the libraries aren't installed yet)
2. python -m streamlit run app.py

All data is in ONE file: data/FutureProof_Dataset.xlsx (already built).

## How the dataset was built (to explain to examiners)
1. prepare_data.py  : cleans the raw files in data/raw/ (jobs, WillRobotsTakeMyJob, OECD, WEF)
2. build_dataset.py : merges them with the team's own task file into FutureProof_Dataset.xlsx

## Files
- app.py      the screen          - engine.py   scores and matching
- data/FutureProof_Dataset.xlsx   sheets: README, tasks, occupation_benchmark,
                                  emerging_skills, future_job_demand, skill_resources
