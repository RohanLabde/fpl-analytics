import streamlit as st
import pandas as pd

from fpl_tool.data import load_all, next_deadline_ist
from fpl_tool.features import build_player_master, fixture_softness
from fpl_tool.model import baseline_expected_points, v2_expected_points
from fpl_tool.optimizer import pick_squad_greedy

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="⚽ FPL Analytics – Fast Decisions", layout="wide")

st.title("⚽ FPL Analytics – Fast Decisions")
st.caption("Data: Official Fantasy Premier League API. Toggle V2 for smarter xPts (minutes + Poisson clean sheets + attacking proxy).")

# ----------------------------
# Load data
# ----------------------------
with st.spinner("Loading latest FPL data..."):
    data = load_all()

players = data["elements"]
teams = data["teams"]
events = data["events"]
fixtures = data["fixtures"]

# ✅ FIX: use `element_types` (correct key)
pm = build_player_master(players, teams, data["element_types"])
soft = fixture_softness(fixtures, teams, horizon=3)

deadline = next_deadline_ist(events)
if deadline:
    st.info(f"🕒 Next deadline (IST): **{deadline.strftime('%a, %d %b %Y %H:%M')}**")

# ----------------------------
# Sidebar: Model Settings
# ----------------------------
st.sidebar.header("Model Settings")

use_v2 = st.sidebar.checkbox("Use V2 xPts (Poisson + minutes + roles)", value=False)

fixture_horizon = st.sidebar.slider(
    "Fixture horizon (matches) [V2]",
    min_value=1, max_value=10, value=3
)

report_mode = st.sidebar.radio(
    "How to report over horizon",
    ["Average per match", "Total across horizon"],
    index=0
)

budget = st.sidebar.slider(
    "Budget (15-man optimizer)",
    min_value=70, max_value=120, value=100
)

# ----------------------------
# Calculate expected points
# ----------------------------
if use_v2:
    pred = v2_expected_points(pm, teams, fixtures, horizon=fixture_horizon)
else:
    pred = baseline_expected_points(pm, teams, soft)

# Normalize display depending on report mode
if report_mode == "Average per match":
    pred["xPts"] = pred["xPts"] / fixture_horizon

pred["xPts_per_m"] = pred["xPts"] / pred["now_cost"]

# ----------------------------
# Captaincy Picks (by position)
# ----------------------------
st.subheader("🎯 Captaincy picks (Top by position)")

cols = ["web_name", "pos", "name", "price", "xPts", "xPts_per_m"]

for pos, top_n in [("GKP", 3), ("DEF", 5), ("MID", 5), ("FWD", 5)]:
    st.markdown(f"**Top {top_n} {pos}s**")
    df = pred[pred["pos"] == pos].sort_values("xPts", ascending=False).head(top_n)
    st.dataframe(df[cols].reset_index(drop=True))

# ----------------------------
# Value Picks (by position)
# ----------------------------
st.subheader("💼 Value picks (Top by position)")

for pos, top_n in [("GKP", 3), ("DEF", 5), ("MID", 5), ("FWD", 5)]:
    st.markdown(f"**Top {top_n} {pos}s**")
    df = pred[pred["pos"] == pos].sort_values("xPts_per_m", ascending=False).head(top_n)
    st.dataframe(df[cols].reset_index(drop=True))

# ----------------------------
# Squad Optimizer
# ----------------------------
st.subheader("🧠 Squad Optimizer (15-man, greedy)")

if st.button("Build 15-man Squad"):
    squad, xp, cost = pick_squad_greedy(pred, budget=budget)
    st.success(f"✅ Squad built with total expected points = {xp:.2f}, cost = {cost:.1f}")
    st.dataframe(squad[cols + ["team"]].reset_index(drop=True))

# ----------------------------
# Analyze My Squad
# ----------------------------
st.subheader("🧩 Analyze My 15-man Squad")

if "squad_ids" not in st.session_state:
    st.session_state["squad_ids"] = []

squad_ids = st.multiselect(
    "Select your 15 players",
    options=pred["id"],
    format_func=lambda x: f"{pred.loc[pred['id']==x,'web_name'].values[0]} ({pred.loc[pred['id']==x,'name'].values[0]})",
    default=st.session_state["squad_ids"],
    key="squad_ids"
)

bank = st.number_input("Bank (money in the bank)", min_value=0.0, step=0.1)

if st.button("Analyze my squad"):
    if len(squad_ids) != 15:
        st.error("Please select exactly 15 players.")
    else:
        current = pred.set_index("id").loc[squad_ids]
        st.dataframe(current[cols])
        best_xi = current.sort_values("xPts", ascending=False).head(11)
        st.markdown("**Best XI (sorted by xPts):**")
        st.dataframe(best_xi[cols])
