import streamlit as st
import pandas as pd

from fpl_tool.data import load_all, next_deadline_ist
from fpl_tool.features import build_player_master, fixture_softness
from fpl_tool.model import baseline_expected_points, v2_expected_points
from fpl_tool.optimizer import pick_squad_greedy

st.set_page_config(page_title="⚽ FPL Analytics – Fast Decisions", layout="wide")

st.title("⚽ FPL Analytics – Fast Decisions")
st.caption("Data: Official Fantasy Premier League API. Toggle V2 for smarter xPts (minutes + Poisson clean sheets + attacking proxy).")

with st.spinner("Loading latest FPL data..."):
    data = load_all()

players = data["players"]
teams = data["teams"]
events = data["events"]
fixtures = data["fixtures"]

deadline = next_deadline_ist(events)
if deadline:
    st.info(f"📅 Next deadline (IST): **{deadline.strftime('%a, %d %b %Y %H:%M')}**")

# -----------------------
# Sidebar model settings
# -----------------------
st.sidebar.header("Model Settings")

use_v2 = st.sidebar.checkbox("Use V2 xPts (Poisson + minutes + roles)", value=True)

horizon = st.sidebar.slider("Fixture horizon (matches) [V2]", 1, 6, 3)
report_style = st.sidebar.radio("How to report over horizon", ["Average per match", "Total across horizon"], index=0)
budget = st.sidebar.slider("Budget (15-man optimizer)", 80.0, 110.0, 100.0, 0.5)

# Build player master table
pm = build_player_master(players, teams, data["elements_types"])
soft = fixture_softness(fixtures, teams, horizon=horizon)

# -----------------------
# Prediction model
# -----------------------
if use_v2:
    pred = v2_expected_points(pm, fixtures, teams, horizon=horizon)
    if report_style == "Average per match":
        pred["xPts"] = pred["xPts"] / horizon
else:
    pred = baseline_expected_points(pm, fixtures, horizon=horizon)

pred = pred.sort_values("xPts", ascending=False).reset_index(drop=True)
pred["xPts_per_m"] = pred["xPts"] / pred["price"]

cols = ["web_name", "pos", "name", "price", "sel", "xPts", "xPts_per_m"]

# -----------------------
# Captaincy picks by position
# -----------------------
st.subheader("🎯 Captaincy picks (Top by xPts per position)")

if not pred.empty:
    pos_limits = {"GKP": 3, "DEF": 5, "MID": 5, "FWD": 5}
    for pos, limit in pos_limits.items():
        st.markdown(f"**Top {limit} {pos}s**")
        subset = pred[pred["pos"] == pos].sort_values("xPts", ascending=False)
        st.dataframe(subset[cols].head(limit).reset_index(drop=True))
else:
    st.warning("No prediction data available yet. Please try refreshing.")

# -----------------------
# Value picks by position
# -----------------------
st.subheader("💼 Value picks (Top by xPts per million, per position)")

if not pred.empty:
    pos_limits = {"GKP": 3, "DEF": 5, "MID": 5, "FWD": 5}
    for pos, limit in pos_limits.items():
        st.markdown(f"**Top {limit} {pos}s**")
        subset = pred[pred["pos"] == pos].sort_values("xPts_per_m", ascending=False)
        st.dataframe(subset[cols].head(limit).reset_index(drop=True))
else:
    st.warning("No prediction data available yet. Please try refreshing.")

# -----------------------
# Squad Optimizer (Greedy)
# -----------------------
st.subheader("🧠 Squad Optimizer (15-man, greedy)")

if st.button("Build 15-man Squad"):
    squad, xp, cost = pick_squad_greedy(pred, budget=budget)
    st.success(f"✅ Built squad with total xPts: {xp:.1f}, Cost: {cost:.1f}")
    st.dataframe(squad[cols])

# -----------------------
# Analyze My 15-man Squad
# -----------------------
st.subheader("🧩 Analyze My 15-man Squad")

label_map = {int(r.id): f"{r['web_name']} ({r['name']}, {r['pos']}, {r['price']})" for r in pred.itertuples(index=False)}

# Restore persisted selection
prev = st.session_state.get("squad_ids", [])
prev = [int(x) for x in prev if int(x) in label_map]

squad_ids = st.multiselect(
    "Select your 15 players",
    list(label_map.keys()),
    default=prev,
    format_func=lambda pid: label_map.get(int(pid), str(pid)),
    key="squad_ids",
)

# Save squad selection persistently
if squad_ids:
    st.session_state.squad_ids = squad_ids

bank = st.number_input("Bank (money in the bank)", 0.0, 10.0, 0.0, 0.1)

if st.button("Analyze my squad") and len(squad_ids) == 15:
    squad_df = pred.set_index("id").loc[squad_ids]
    st.write("### Best XI (sorted by xPts):")
    st.dataframe(squad_df[cols].sort_values("xPts", ascending=False).head(11).reset_index(drop=True))

    cap = squad_df.iloc[0]
    vice = squad_df.iloc[1]
    st.success(f"Captain: **{cap.web_name}** ({cap.xPts:.2f} pts), Vice: **{vice.web_name}**")

    current_cost = float(squad_df["price"].sum())
    st.info(f"💰 Current squad cost: {current_cost:.1f}, Bank: {bank:.1f}")
