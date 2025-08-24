
import streamlit as st
import pandas as pd

from fpl_tool.data import load_all, next_deadline_ist
from fpl_tool.features import build_player_master, fixture_softness
from fpl_tool.model import baseline_expected_points
from fpl_tool.optimizer import pick_squad_greedy

st.set_page_config(page_title="FPL Analytics – Render Free", layout="wide")

st.title("⚽ FPL Analytics – Fast Decisions (Free)")
st.caption("Data: Official Fantasy Premier League API. Uses form, minutes, and fixture softness.")

with st.spinner("Loading latest FPL data..."):
    data = load_all()

players = data["players"]; teams=data["teams"]; events=data["events"]; fixtures=data["fixtures"]

deadline = next_deadline_ist(events)
if deadline:
    st.info(f"⏳ Next deadline (IST): **{deadline.strftime('%a, %d %b %Y %H:%M')}**")

pm = build_player_master(players, teams, data["positions"])
soft = fixture_softness(fixtures, teams, horizon=3)

st.sidebar.header("Model Settings")
horizon = st.sidebar.slider("Fixture horizon (matches)", 1, 5, 3)
alpha = st.sidebar.slider("Weight: Form", 0.0, 1.5, 0.7, 0.05)
beta  = st.sidebar.slider("Weight: Minutes", 0.0, 1.0, 0.2, 0.05)
gamma = st.sidebar.slider("Weight: Fixture softness", 0.0, 1.0, 0.1, 0.05)
budget = st.sidebar.slider("Budget", 90.0, 100.0, 100.0, 0.5)

pred = baseline_expected_points(pm, events, soft, horizon=horizon, alpha=alpha, beta=beta, gamma=gamma)

st.subheader("🎯 Captaincy picks (Top 15 by xPts)")
if not pred.empty:
    cols = ["web_name","pos","name","price","sel","xPts","xPts_per_m"]
    st.dataframe(pred[cols].head(15).reset_index(drop=True))
else:
    st.warning("No prediction data available yet. Please try refreshing.")

st.subheader("💼 Value picks (Top 15 by xPts per million)")
if not pred.empty:
    cols = ["web_name","pos","name","price","sel","xPts","xPts_per_m"]
    st.dataframe(pred[cols].sort_values("xPts_per_m", ascending=False).head(15).reset_index(drop=True))

st.subheader("🧠 Squad Optimizer (15-man, greedy)")
if st.button("Build 15-man Squad"):
    if pred.empty:
        st.error("Predictions are empty; cannot build squad.")
    else:
        squad, xp, cost = pick_squad_greedy(pred, budget=budget)
        st.success(f"Selected 15 → Estimated xPts = {xp:.2f}, Cost = {cost:.1f}")
        st.dataframe(squad[["web_name","pos","name","price","xPts","xPts_per_m"]])
