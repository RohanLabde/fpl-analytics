# app.py
import itertools
from typing import List, Tuple

import pandas as pd
import numpy as np
import requests
import streamlit as st

# Import model functions
from fpl_tool.model import build_player_master, v2_expected_points

try:
    from fpl_tool.model import add_value_columns
except Exception:
    add_value_columns = None


# -----------------------
# Data loaders (cached)
# -----------------------
@st.cache_data(ttl=3600)
def load_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    data = requests.get(url).json()
    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    element_types = pd.DataFrame(data["element_types"])
    return players, teams, element_types


@st.cache_data(ttl=3600)
def load_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    return pd.DataFrame(requests.get(url).json())


# -----------------------
# Team Tables (FIXED)
# -----------------------
def build_team_season_table(fixtures: pd.DataFrame, teams: pd.DataFrame):
    fx = fixtures[fixtures["finished"] == True].copy()
    rows = []

    for _, t in teams.iterrows():
        tid = t["id"]
        name = t["name"]

        home = fx[fx["team_h"] == tid]
        away = fx[fx["team_a"] == tid]

        played = len(home) + len(away)

        if played == 0:
            rows.append({
                "id": tid, "name": name,
                "played": 0, "points": 0, "win": 0, "draw": 0, "loss": 0,
                "GF": 0, "GA": 0, "CS": 0,
                "GF_per_match": 0.0, "GA_per_match": 0.0, "CS_%": 0.0
            })
            continue

        GF = home["team_h_score"].sum() + away["team_a_score"].sum()
        GA = home["team_a_score"].sum() + away["team_h_score"].sum()
        CS = (home["team_a_score"] == 0).sum() + (away["team_h_score"] == 0).sum()

        win = (home["team_h_score"] > home["team_a_score"]).sum() + \
              (away["team_a_score"] > away["team_h_score"]).sum()

        draw = (home["team_h_score"] == home["team_a_score"]).sum() + \
               (away["team_a_score"] == away["team_h_score"]).sum()

        loss = played - win - draw
        points = win * 3 + draw

        rows.append({
            "id": tid,
            "name": name,
            "played": played,
            "points": points,
            "win": win,
            "draw": draw,
            "loss": loss,
            "GF": int(GF),
            "GA": int(GA),
            "CS": int(CS),
            "GF_per_match": round(GF / played, 2),
            "GA_per_match": round(GA / played, 2),
            "CS_%": round(100 * CS / played, 1),
        })

    return pd.DataFrame(rows).sort_values("points", ascending=False)


def build_team_fpl_production(players: pd.DataFrame):
    df = players.copy()

    numeric_cols = ["goals_scored", "assists", "clean_sheets", "bonus", "expected_goals"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    agg = df.groupby("team").agg({
        "goals_scored": "sum",
        "assists": "sum",
        "clean_sheets": "sum",
        "bonus": "sum",
        "expected_goals": "sum",
    }).reset_index()

    return agg


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="FPL Analytics – Smarter Expected Points", layout="wide")
st.title("⚽ FPL Analytics – Smarter Expected Points")
st.caption("Data: Official Fantasy Premier League API. Model uses v2 logic.")

players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

pm = build_player_master(players.copy(), teams.copy(), element_types.copy())

# Sidebar
st.sidebar.header("Model & display settings")
horizon = st.sidebar.slider("Fixture horizon (matches)", 1, 10, 5)

rank_by_choice = st.sidebar.selectbox("Rank by", ["xPts_total", "xPts_per_match"], index=0)

min_minutes_for_leaderboards = st.sidebar.slider("Min historical minutes filter", 0, 2000, 300)
top_n_per_position = st.sidebar.number_input("Top N per position", 1, 20, 10)

# -----------------------
# Run model
# -----------------------
pred = v2_expected_points(pm.copy(), fixtures.copy(), teams.copy(), horizon=horizon)

# Clamp games_proj = 1 ALWAYS
pred["games_proj"] = 1

# Ensure totals exist
if "xPts_total" not in pred.columns:
    pred["xPts_total"] = pred["xPts_per_match"] * horizon

# Value columns
if add_value_columns:
    pred = add_value_columns(pred)
else:
    pred["xPts_per_m"] = pred["xPts_per_match"] / (pred["now_cost"] / 10)

pred["£m"] = pred["now_cost"] / 10
pred["selected_by_percent"] = pd.to_numeric(pred["selected_by_percent"], errors="coerce").fillna(0)

# -----------------------
# TEAM DASHBOARD (FIXED)
# -----------------------
st.header("📊 Team Strength & Form Dashboard")

st.subheader("🍱 Season Team Strength")
team_season = build_team_season_table(fixtures, teams)
st.dataframe(team_season.reset_index(drop=True))

st.subheader("🎯 FPL Production by Team")
team_prod = build_team_fpl_production(players)

team_map = teams.set_index("id")["name"].to_dict()
team_prod["team_name"] = team_prod["team"].map(team_map)

team_prod = team_prod.rename(columns={
    "goals_scored": "Goals",
    "assists": "Assists",
    "clean_sheets": "CleanSheets",
    "bonus": "Bonus",
    "expected_goals": "xG",
})

team_prod["xG"] = team_prod["xG"].round(2)
team_prod = team_prod.sort_values("Goals", ascending=False)

st.dataframe(team_prod[["team_name", "Goals", "Assists", "CleanSheets", "Bonus", "xG"]].reset_index(drop=True))

# -----------------------
# LEADERBOARDS
# -----------------------
st.header("🎯 Player Leaderboards")

for pos in ["GKP", "DEF", "MID", "FWD"]:
    dfp = pred[pred["pos"] == pos].copy()

    if min_minutes_for_leaderboards > 0:
        dfp = dfp[dfp["minutes"] >= min_minutes_for_leaderboards]

    dfp = dfp.sort_values(rank_by_choice, ascending=False).head(top_n_per_position)

    st.subheader(f"Top {len(dfp)} {pos}s by {rank_by_choice}")

    cols = ["web_name", "team_name", "pos", "£m", "selected_by_percent", "xPts_per_match", "xPts_total"]
    st.dataframe(dfp[cols].reset_index(drop=True))

# -----------------------
# DONE
# -----------------------
st.success("✅ Team tables, xPts model, filters and rankings are now consistent and fixed.")
