import streamlit as st
import pandas as pd

from fpl_tool.data import load_all, next_deadline_ist
from fpl_tool.features import build_player_master, fixture_softness
from fpl_tool.model import baseline_expected_points, v2_expected_points
from fpl_tool.optimizer import pick_squad_greedy

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="⚽ FPL Analytics – Fast Decisions", layout="wide")

st.title("⚽ FPL Analytics – Fast Decisions")
st.caption("Data: Official Fantasy Premier League API. Toggle V2 for smarter xPts (minutes + Poisson clean sheets + attacking proxy).")

# ----------------------------
# LOAD DATA
# ----------------------------
with st.spinner("Loading latest FPL data..."):
    data = load_all()

players = data["elements"]
teams = data["teams"]
events = data["events"]
positions = data["element_types"]
fixtures = data["fixtures"]

deadline = next_deadline_ist(events)
if deadline:
    st.info(f"🕒 Next deadline (IST): **{deadline.strftime('%a, %d %b %Y %H:%M')}**")

# ----------------------------
# MODEL SETTINGS
# ----------------------------
st.sidebar.header("⚙️ Model Settings")

use_v2 = st.sidebar.checkbox("Use V2 xPts (Poisson + minutes + roles)", value=True)
horizon = st.sidebar.slider("Fixture horizon (matches) [V2]", 1, 6, 3)
report_mode = st.sidebar.radio("How to report over horizon", ["Average per match", "Total across horizon"])
budget = st.sidebar.slider("Budget (15-man optimizer)", 80.0, 105.0, 100.0, step=0.5)

# ----------------------------
# PLAYER MASTER TABLE
# ----------------------------
pm = build_player_master(players, teams, positions)

# Fixture softness
soft = fixture_softness(fixtures, teams, horizon=horizon)

# ----------------------------
# EXPECTED POINTS
# ----------------------------
if use_v2:
    pred = v2_expected_points(pm, fixtures, teams, horizon=horizon)
else:
    pred = baseline_expected_points(pm, horizon=horizon)

# Adjust reporting mode
if report_mode == "Average per match":
    pred["xPts"] = pred["xPts"] / horizon

# Value metric
pred["xPts_per_m"] = pred["xPts"] / pred["price"]

# ----------------------------
# CAPTAINCY & VALUE PICKS
# ----------------------------
st.subheader("🎯 Captaincy picks (Top players by position)")
cols = ["web_name", "pos", "name", "price", "sel", "xPts", "xPts_per_m"]

# Group by position
for pos in ["GKP", "DEF", "MID", "FWD"]:
    if pos == "GKP":
        top_n = 3
    else:
        top_n = 5
    st.markdown(f"**Top {top_n} {pos}s**")
    df_pos = pred[pred["pos"] == pos].sort_values("xPts", ascending=False).head(top_n)
    st.dataframe(df_pos[cols].reset_index(drop=True))

# ----------------------------
# SQUAD OPTIMIZER
# ----------------------------
st.subheader("🧠 Squad Optimizer (15-man, greedy)")
if st.button("Build 15-man Squad"):
    squad, xp, cost = pick_squad_greedy(pred, budget=budget)
    st.success(f"Built squad with {xp:.1f} expected points and cost {cost:.1f}")
    st.dataframe(squad[cols])

# ----------------------------
# ANALYZE MY SQUAD
# ----------------------------
st.subheader("🧩 Analyze My 15-man Squad")
st.caption("Pick your current 15 players, set your bank, and we’ll pick a best XI, captain/vice, and suggest transfers.")

options = {int(r.id): f"{r['web_name']} ({r['name']}, {r['pos']}, {r['price']})" for r in pred.itertuples(index=False)}

squad_ids = st.multiselect(
    "Select your 15 players",
    options.keys(),
    format_func=lambda x: options[x],
    max_selections=15,
    key="squad_ids"
)

bank = st.number_input("Bank (money in the bank)", 0.0, 10.0, 0.0, step=0.1)

if st.button("Analyze my squad"):
    if len(squad_ids) != 15:
        st.error("Please select exactly 15 players.")
    else:
        # Current squad
        current_squad = pred.set_index("id").loc[squad_ids].reset_index()

        # Best XI
        best_xi = current_squad.sort_values("xPts", ascending=False).head(11)
        st.markdown("**Best XI (sorted by xPts):**")
        st.dataframe(best_xi[cols])

        # Captain & Vice
        captain = best_xi.iloc[0]
        vice = best_xi.iloc[1]
        st.success(f"Captain: **{captain.web_name}** ({captain.xPts:.2f} xPts), Vice: **{vice.web_name}**")

        # Suggest transfers (basic: best players outside current squad)
        available = pred[~pred["id"].isin(squad_ids)]
        suggestions = available.sort_values("xPts", ascending=False).head(5)
        st.markdown("**Suggested transfers (Top 5 by xPts):**")
        st.dataframe(suggestions[cols])
