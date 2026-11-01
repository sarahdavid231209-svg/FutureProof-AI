"""
engine.py - all the data loading, matching and score calculations.
app.py only draws the screen; the thinking happens here.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------------ loading
def load_all(path="data/FutureProof_Dataset.xlsx"):
    """Reads the single combined workbook (one sheet per table)."""
    sheets = pd.read_excel(path, sheet_name=None)
    return {
        "tasks": sheets["tasks"],
        "bench": sheets["occupation_benchmark"],
        "skills": sheets["emerging_skills"],
        "demand": sheets["future_job_demand"],
        "resources": sheets["skill_resources"],
    }


# ----------------------------------------------------------- name matching
class NameMatcher:
    """Finds the closest name in a list (character n-gram TF-IDF).
    Works well for 'Data Entry Clerk' vs 'Data Entry Clerks'."""

    def __init__(self, names):
        self.names = list(names)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        self.matrix = self.vec.fit_transform(self.names)

    def best(self, query):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix).flatten()
        i = int(sims.argmax())
        return i, float(sims[i])

    def top(self, query, n=5):
        """Indexes of the n closest names, best first."""
        sims = cosine_similarity(self.vec.transform([query]), self.matrix).flatten()
        return [int(i) for i in sims.argsort()[::-1][:n]]


class TaskMatcher:
    """Matches a sentence typed by the user to the closest known tasks."""

    def __init__(self, tasks_df):
        self.df = tasks_df.drop_duplicates("task_name").reset_index(drop=True)
        self.vec = TfidfVectorizer(stop_words="english")
        self.matrix = self.vec.fit_transform(self.df["task_name"])

    def match(self, text, top_n=3):
        if not text.strip():
            return self.df.iloc[0:0]
        sims = cosine_similarity(self.vec.transform([text]), self.matrix).flatten()
        idx = sims.argsort()[::-1][:top_n]
        out = self.df.iloc[idx].copy()
        out["similarity"] = sims[idx]
        return out[out["similarity"] > 0.25]


# ------------------------------------------------------------------ scores
def compute_scores(selected_tasks, bench_row):
    """
    selected_tasks : rows of jobs_tasks.csv the user does
    bench_row      : one row of occupation_benchmark.csv (official occupation)

    AI Readiness Score (ARS)   = average AI probability of the user's tasks
                                 (how much of the workload AI can already do)
    Human Advantage Score (HAS)= OECD measure of how far AI is from the job's
                                 human-skill demands (social, creative,
                                 critical thinking, problem solving)
    Job Survival Index (JSI)   = 100 - average of three risk signals:
                                 task-level, WillRobotsTakeMyJob, OECD exposure
    """
    task_risk = float(selected_tasks["task_ai_probability_pct"].mean())
    wr_risk = float(bench_row["calc_risk_pct"])
    oecd_risk = float(bench_row["oecd_exposure_pct"])
    has = float(bench_row["human_advantage"])
    jsi = float(100 - np.mean([task_risk, wr_risk, oecd_risk]))
    return {
        "ars": round(task_risk, 1),
        "has": round(has, 1),
        "jsi": round(jsi, 1),
        "wr_risk": round(wr_risk, 1),
        "oecd_risk": round(oecd_risk, 1),
    }


def split_tasks(selected_tasks, threshold=50):
    vulnerable = selected_tasks[selected_tasks["task_ai_probability_pct"] >= threshold]
    safe = selected_tasks[selected_tasks["task_ai_probability_pct"] < threshold]
    return (vulnerable.sort_values("task_ai_probability_pct", ascending=False),
            safe.sort_values("task_ai_probability_pct"))


def project_timeline(ars, years=(3, 5, 10)):
    """Illustrative only: exposure drifts upward if the person does not upskill."""
    out = {}
    for y in years:
        growth = 1 - np.exp(-y / 8)
        out[y] = round(float(min(ars + (100 - ars) * growth * 0.6, 97)), 1)
    return out


# ------------------------------------------------------------ future demand
def find_wef_outlook(job_name, demand_df, threshold=0.55):
    """Return (row, similarity) of the closest WEF occupation, or (None, sim)."""
    m = NameMatcher(demand_df["occupation"])
    i, sim = m.best(job_name)
    if sim >= threshold:
        return demand_df.iloc[i], sim
    return None, sim


# ------------------------------------------------------------------ roadmap
# Which WEF skills reinforce which OECD human-skill domain
DOMAIN_TO_SKILLS = {
    "Soc_Dmd": ["Leadership and social influence", "Empathy and active listening",
                "Service orientation and customer service"],
    "Creativity_Dmd": ["Creative thinking", "Design and user experience"],
    "Metacognition_Dmd": ["Analytical thinking", "Curiosity and lifelong learning"],
    "ProbSolv_Dmd": ["Systems thinking", "Analytical thinking"],
}
DOMAIN_LABEL = {
    "Soc_Dmd": "social interaction", "Creativity_Dmd": "creativity",
    "Metacognition_Dmd": "critical thinking", "ProbSolv_Dmd": "problem solving",
}


def build_roadmap(bench_row, skills_df, resources_df, jsi):
    """
    Phase 1 'Adapt'   : top rising technology skills (work WITH AI)
    Phase 2 'Amplify' : rising skills that match the job's strongest human domains
    """
    rising = skills_df[skills_df["outlook"].str.contains("Increasing")]
    n_tech = 3 if jsi < 60 else 2

    tech = rising[rising["skill_category"].str.startswith("Technology")]
    tech = tech.sort_values("net_increase_score", ascending=False).head(n_tech)

    demands = {d: float(bench_row[d]) for d in DOMAIN_TO_SKILLS}
    top_domains = sorted(demands, key=demands.get, reverse=True)[:2]
    human_names = []
    for d in top_domains:
        for s in DOMAIN_TO_SKILLS[d]:
            if s not in human_names and s not in tech["skill"].tolist():
                human_names.append(s)
    human = rising[rising["skill"].isin(human_names)].sort_values(
        "net_increase_score", ascending=False).head(3)

    def attach(df):
        rows = []
        for _, r in df.iterrows():
            links = resources_df[resources_df["skill"] == r["skill"]]
            rows.append({"skill": r["skill"], "score": int(r["net_increase_score"]),
                         "links": links[["resource_name", "provider", "url"]].to_dict("records")})
        return rows

    return {
        "adapt": attach(tech),
        "amplify": attach(human),
        "strong_domains": [DOMAIN_LABEL[d] for d in top_domains],
    }
