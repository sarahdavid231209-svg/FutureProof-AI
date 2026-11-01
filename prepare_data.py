"""
prepare_data.py  -  run ONCE before starting the app:

    python prepare_data.py

Reads the 5 raw files from data/raw/ (file names may end in _1, _2 etc.)
and writes clean, app-ready CSVs into data/.
"""

import glob
import os
import sys

import pandas as pd

RAW = "data/raw"
OUT = "data"


def find(pattern):
    """Find a raw file by the start of its name, e.g. 'jobs-535*'."""
    hits = sorted(glob.glob(os.path.join(RAW, pattern)))
    if not hits:
        sys.exit(f"ERROR: could not find a file matching '{pattern}' in {RAW}/")
    return hits[0]


# ---------------------------------------------------------------- 1. TASKS
tasks = pd.read_csv(find("jobs-535*.csv"))
tasks.columns = [c.strip() for c in tasks.columns]
for col in ["job_name", "category", "task_name"]:
    tasks[col] = tasks[col].astype(str).str.strip()
tasks = tasks.drop_duplicates()
tasks.to_csv(f"{OUT}/jobs_tasks.csv", index=False)
print(f"jobs_tasks.csv         : {len(tasks)} rows, {tasks.job_name.nunique()} jobs")

# ------------------------------------------- 2. AUTOMATION RISK (willrobots)
wr = pd.read_csv(find("willrobotstakemyjob*.csv"))
wr = wr[wr["Is Group"] == False].copy()  # drop the 4 "group" rows
wr = wr.rename(columns={
    "SOC Code": "soc_code",
    "Name": "occupation",
    "Calculated Risk %": "calc_risk_pct",
    "Voted Risk %": "voted_risk_pct",
    "Votes": "votes",
})
wr["votes"] = pd.to_numeric(wr["votes"].astype(str).str.replace(",", ""), errors="coerce")
wr["soc_code"] = wr["soc_code"].astype(str).str.strip()
wr = wr[["soc_code", "occupation", "calc_risk_pct", "voted_risk_pct", "votes"]]

# --------------------------------------------------- 3. AI EXPOSURE (OECD)
raw = pd.read_excel(find("OECD_AI*.xlsx"), sheet_name="Data", header=None)
codes = raw.iloc[2].tolist()          # row 3 holds the short column names
oecd = raw.iloc[4:].copy()            # real data starts on row 5
oecd.columns = codes
oecd = oecd.rename(columns={
    "OCC_Code": "soc_code",
    "OCC_Title": "oecd_title",
    "AI Capability Gap Index_Rev. norm.": "oecd_exposure",   # 0-1, higher = more exposed
})
oecd["soc_code"] = oecd["soc_code"].astype(str).str.strip()
num_cols = [c for c in oecd.columns if c not in ("soc_code", "oecd_title")]
oecd[num_cols] = oecd[num_cols].apply(pd.to_numeric, errors="coerce")

# Human Advantage Score: how far AI still is from meeting the job's demands in
# the four human-heavy domains (each gap is on a 0-3 scale, bigger = safer).
human_gap_cols = ["Soc_Gap", "Creativity_Gap", "Metacognition_Gap", "ProbSolv_Gap"]
oecd["human_advantage"] = oecd[human_gap_cols].mean(axis=1) / 3 * 100

keep = ["soc_code", "oecd_title", "oecd_exposure", "human_advantage",
        "Soc_Dmd", "Creativity_Dmd", "Metacognition_Dmd", "ProbSolv_Dmd"]
oecd = oecd[keep]

# ---------------------------------------- 4. MERGE (willrobots + OECD by SOC)
bench = wr.merge(oecd, on="soc_code", how="inner")
# Stretch the OECD index to 0-100 across all occupations so it is comparable
# with the other risk numbers (the raw index only ranges from about 0.49 to 1).
lo, hi = bench["oecd_exposure"].min(), bench["oecd_exposure"].max()
bench["oecd_exposure_pct"] = (bench["oecd_exposure"] - lo) / (hi - lo) * 100
bench = bench.drop(columns=["oecd_exposure", "oecd_title"])
bench.to_csv(f"{OUT}/occupation_benchmark.csv", index=False)
print(f"occupation_benchmark.csv: {len(bench)} occupations (merged on SOC code)")

# ------------------------------------------------ 5. SKILLS + FUTURE DEMAND
skills = pd.read_csv(find("emerging_skills*.csv"))
skills = skills[["rank_by_net_increase", "skill", "skill_category",
                 "net_increase_score", "outlook", "source", "source_url"]]
skills.to_csv(f"{OUT}/emerging_skills_clean.csv", index=False)
print(f"emerging_skills_clean.csv: {len(skills)} skills")

demand = pd.read_csv(find("future_job_demand*.csv"))
demand = demand[["occupation", "outlook_group", "net_growth_outlook_percent",
                 "source", "source_url"]]
demand.to_csv(f"{OUT}/future_job_demand_clean.csv", index=False)
print(f"future_job_demand_clean.csv: {len(demand)} occupations")

print("\nDone. Now run:  python -m streamlit run app.py")
