"""
FutureProof AI - Human Skill Navigator   (version 2: real datasets)
Run:  python -m streamlit run app.py
Data lives in one file: data/FutureProof_Dataset.xlsx
"""

import os

import pandas as pd
import streamlit as st

import engine

st.set_page_config(page_title="FutureProof AI - Human Skill Navigator", layout="wide")

# ----------------------------------------------------------------- data
if not os.path.exists("data/FutureProof_Dataset.xlsx"):
    st.error("Dataset file data/FutureProof_Dataset.xlsx not found. Run  `python prepare_data.py`  then  `python build_dataset.py`.")
    st.stop()


@st.cache_data
def get_data():
    return engine.load_all("data/FutureProof_Dataset.xlsx")


@st.cache_resource
def get_matchers(_d):
    return (engine.NameMatcher(_d["bench"]["occupation"]),
            engine.TaskMatcher(_d["tasks"]))


D = get_data()
occ_matcher, task_matcher = get_matchers(D)
tasks_df, bench_df = D["tasks"], D["bench"]

# ------------------------------------------------------------- header
st.title("🧭 FutureProof AI - Human Skill Navigator")
st.caption('"Preparing Humans for an AI-Driven Future, Not Replacing Them"')

with st.sidebar:
    st.header("Your Profile")
    name = st.text_input("Name (optional)")
    years_exp = st.slider("Years of Experience", 0, 40, 3)
    st.markdown("---")
    st.caption(f"Dataset: {tasks_df.job_name.nunique()} jobs, {len(tasks_df)} tasks, "
               f"{len(bench_df)} official occupations")

# --------------------------------------------------- step 1: job + tasks
st.subheader("1. Choose your job and your daily tasks")
c1, c2 = st.columns(2)
category = c1.selectbox("Job category", sorted(tasks_df["category"].unique()))
jobs_in_cat = sorted(tasks_df[tasks_df["category"] == category]["job_name"].unique())
job = c2.selectbox("Your job", jobs_in_cat)

job_tasks = tasks_df[tasks_df["job_name"] == job]
chosen = st.multiselect("Tasks you do (untick any that don't apply)",
                        job_tasks["task_name"].tolist(),
                        default=job_tasks["task_name"].tolist(),
                        key=f"tasks_{job}")
selected = job_tasks[job_tasks["task_name"].isin(chosen)]

with st.expander("Task lookup: how exposed is a task that isn't listed?"):
    text = st.text_input("Describe a task", placeholder="e.g. scheduling appointments for clients")
    matches = task_matcher.match(text) if text else pd.DataFrame()
    if text and matches.empty:
        st.write("No close match found. Try different words.")
    elif not matches.empty:
        show = matches[["task_name", "job_name", "task_ai_probability_pct", "similarity"]].copy()
        show["similarity"] = show["similarity"].round(2)
        st.dataframe(show.rename(columns={"task_name": "Closest task", "job_name": "From job",
                                          "task_ai_probability_pct": "AI probability %",
                                          "similarity": "Match"}), hide_index=True)
        add_labels = st.multiselect("Add any of these to my task list", matches["task_name"].tolist())
        if add_labels:
            extra = matches[matches["task_name"].isin(add_labels)][tasks_df.columns]
            selected = pd.concat([selected, extra]).drop_duplicates("task_name")

# ----------------------------------------- step 2: official occupation
st.subheader("2. Match to an official occupation (for benchmark data)")
top_idx = occ_matcher.top(job, 5)
suggested = [bench_df["occupation"].iloc[i] for i in top_idx]
others = sorted(set(bench_df["occupation"]) - set(suggested))
occ_name = st.selectbox("Closest official occupation (suggestions first, or type to search)",
                        suggested + others, key=f"occ_{job}")
bench_row = bench_df[bench_df["occupation"] == occ_name].iloc[0]

# ------------------------------------------------------ run diagnostic
if "ran" not in st.session_state:
    st.session_state.ran = False
if st.button("🔍 Run Diagnostic", type="primary"):
    st.session_state.ran = True

if st.session_state.ran:
    if selected.empty:
        st.warning("Select at least one task first.")
        st.stop()

    sc = engine.compute_scores(selected, bench_row)
    vulnerable, safe = engine.split_tasks(selected)

    st.divider()
    st.subheader("3. Your Diagnostic Dashboard")
    m1, m2, m3 = st.columns(3)
    m1.metric("AI Readiness Score", f"{sc['ars']}%",
              help="Average AI probability across your tasks. Higher = more of your work AI can already do.")
    m2.metric("Human Advantage Score", f"{sc['has']}%",
              help="OECD: how far AI still is from this occupation's social, creative, critical-thinking and problem-solving demands.")
    m3.metric("Job Survival Index", f"{sc['jsi']}%",
              help="100 minus the average of three risk signals (your tasks, WillRobotsTakeMyJob, OECD exposure).")

    st.caption(f"Benchmarked against: **{occ_name}**  |  "
               f"WillRobotsTakeMyJob risk: {sc['wr_risk']}%  |  OECD exposure (0-100): {sc['oecd_risk']}")

    with st.expander("How are these scores calculated?"):
        st.markdown(
            "- **AI Readiness Score** = average AI probability of the tasks you selected.\n"
            "- **Human Advantage Score** = OECD capability-gap measure for social interaction, "
            "creativity, critical thinking and problem solving (higher = AI is further from matching the job).\n"
            "- **Job Survival Index** = 100 − average of: task risk, WillRobotsTakeMyJob calculated risk, "
            "and OECD AI exposure (rescaled 0-100).\n\n"
            "These are directional indicators for guidance, not predictions of any individual's job loss.")

    # --- task breakdown
    st.subheader("4. Task-Level Automation Breakdown")
    st.bar_chart(selected.set_index("task_name")["task_ai_probability_pct"])
    a, b = st.columns(2)
    with a:
        st.markdown("**⚠️ Vulnerable tasks (50%+)**")
        if vulnerable.empty:
            st.write("None. Your task mix is well protected.")
        for _, r in vulnerable.iterrows():
            st.write(f"- {r['task_name']} ({r['task_ai_probability_pct']:.0f}%)")
    with b:
        st.markdown("**✅ Human-centric strengths (under 50%)**")
        if safe.empty:
            st.write("None flagged as low-risk yet.")
        for _, r in safe.iterrows():
            st.write(f"- {r['task_name']} ({r['task_ai_probability_pct']:.0f}%)")

    # --- future demand
    st.subheader("5. Future Demand Outlook")
    wef_row, sim = engine.find_wef_outlook(job, D["demand"])
    if wef_row is not None:
        pct = int(wef_row["net_growth_outlook_percent"])
        arrow = "📈" if pct > 0 else "📉"
        st.write(f"{arrow} Closest WEF occupation: **{wef_row['occupation']}** "
                 f"({wef_row['outlook_group']}, net outlook {pct:+d}% for 2025-2030).")
    else:
        st.write("No direct match for this job in the WEF fastest-growing / fastest-declining lists. "
                 "That usually means it sits in the middle of the pack.")
    st.caption("World Economic Forum, Future of Jobs Report 2025: global employer expectations, "
               "not a guaranteed job count and not India-specific.")

    # --- roadmap
    st.subheader("6. Personalized Upskilling Roadmap")
    rm = engine.build_roadmap(bench_row, D["skills"], D["resources"], sc["jsi"])

    def show_skills(items):
        for it in items:
            st.markdown(f"**{it['skill']}**  (WEF net-increase score: {it['score']})")
            if it["links"]:
                for l in it["links"]:
                    st.write(f"- [{l['resource_name']}]({l['url']}) ({l['provider']})")
            else:
                st.write("- Search this skill on Coursera, edX or Khan Academy (free audit options).")

    st.markdown("#### Phase 1: Adapt (learn to work with AI)")
    show_skills(rm["adapt"])
    st.markdown("#### Phase 2: Amplify (strengthen your human edge)")
    st.caption(f"Your occupation's strongest human-skill demands: {', '.join(rm['strong_domains'])}.")
    show_skills(rm["amplify"])

    # --- timeline
    st.subheader("7. Career Timeline Projection (directional estimate)")
    proj = engine.project_timeline(sc["ars"])
    st.line_chart(pd.DataFrame({"Years from now": list(proj.keys()),
                                "Projected AI exposure (%)": list(proj.values())}).set_index("Years from now"))
    st.caption("Illustrative curve assuming no upskilling. Not a scientific forecast.")

    # --- export
    st.subheader("8. Export Your Report")
    lines = [
        "FutureProof AI - Diagnostic Report",
        f"Name: {name or 'N/A'}", f"Job: {job} ({category})", f"Years of experience: {years_exp}",
        f"Benchmark occupation: {occ_name}", "",
        f"AI Readiness Score: {sc['ars']}%", f"Human Advantage Score: {sc['has']}%",
        f"Job Survival Index: {sc['jsi']}%", "", "Vulnerable tasks:",
    ] + [f"- {t}" for t in vulnerable["task_name"]] + ["", "Human-centric strengths:"] + \
        [f"- {t}" for t in safe["task_name"]] + ["", "Roadmap - Adapt:"] + \
        [f"- {i['skill']}" for i in rm["adapt"]] + ["Roadmap - Amplify:"] + \
        [f"- {i['skill']}" for i in rm["amplify"]] + ["",
        "Sources: OECD AI Capability Gap Index (OECD 2026, CC BY 4.0); WillRobotsTakeMyJob; "
        "WEF Future of Jobs Report 2025; task-level dataset (jobs-535). All datasets collected via Kaggle."]
    st.download_button("📄 Download Report (.txt)", "\n".join(lines), file_name="futureproof_report.txt")

    with st.expander("Data sources and limitations"):
        st.markdown(
            "- **OECD AI Capability Gap Index**: OECD (2026), OECD AI Papers No. 59, licence CC BY 4.0.\n"
            "- **WillRobotsTakeMyJob**: occupation risk estimates by SOC code.\n"
            "- **World Economic Forum**: Future of Jobs Report 2025 (skills and job outlook).\n"
            "- **Task-level data (jobs-535)**: obtained from Kaggle.\n"
            "- All datasets were collected via **Kaggle** (kaggle.com); credit goes to the original dataset authors.\n"
            "- Data is mostly US-based (O*NET SOC codes); results are guidance, not guarantees.")
