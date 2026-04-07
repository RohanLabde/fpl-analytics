import pandas as pd
import numpy as np
import requests
import streamlit as st

from fpl_tool.model import build_player_master, v3_expected_points, add_value_columns


# -----------------------
# Data loaders
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


@st.cache_data(ttl=3600)
def run_model(pm, fixtures, teams, horizon, form_weight, bonus_weight):
    return v3_expected_points(
        pm.copy(),
        fixtures.copy(),
        teams.copy(),
        horizon=horizon,
        form_weight=form_weight,
        bonus_weight=bonus_weight,
    )


# -----------------------
# TEAM TABLES
# -----------------------
def build_team_table_from_fixtures(fixtures, teams, last_n=None):
    fx = fixtures[fixtures["finished"] == True].copy()

    rows = []

    for _, t in teams.iterrows():
        tid = t["id"]
        name = t["name"]

        team_fx = fx[(fx["team_h"] == tid) | (fx["team_a"] == tid)].copy()

        if "kickoff_time" in team_fx.columns:
            team_fx = team_fx.sort_values("kickoff_time")

        if last_n is not None:
            team_fx = team_fx.tail(last_n)

        played = len(team_fx)

        if played == 0:
            rows.append({"team": name, "played": 0})
            continue

        GF = GA = CS = win = draw = loss = 0

        for _, m in team_fx.iterrows():
            if m["team_h"] == tid:
                gf = m["team_h_score"]
                ga = m["team_a_score"]
            else:
                gf = m["team_a_score"]
                ga = m["team_h_score"]

            GF += gf
            GA += ga
            CS += (ga == 0)
            win += (gf > ga)
            draw += (gf == ga)
            loss += (gf < ga)

        rows.append({
            "team": name,
            "played": played,
            "points": win*3 + draw,
            "GF": GF,
            "GA": GA,
            "CS": CS
        })

    return pd.DataFrame(rows).sort_values("points", ascending=False)


def build_team_fpl_production(players, teams):
    df = players.copy()

    df["goals_scored"] = pd.to_numeric(df["goals_scored"], errors="coerce").fillna(0)
    df["assists"] = pd.to_numeric(df["assists"], errors="coerce").fillna(0)

    agg = df.groupby("team").agg({
        "goals_scored": "sum",
        "assists": "sum"
    }).reset_index()

    team_map = teams.set_index("id")["name"].to_dict()
    agg["team"] = agg["team"].map(team_map)

    return agg.sort_values("goals_scored", ascending=False)


# -----------------------
# DECISION ENGINE
# -----------------------
def get_captain_picks(df):
    return df[df["pos"].isin(["MID", "FWD"])].sort_values("xPts_per_match", ascending=False).head(5)


def get_differentials(df):
    df = df[df["pos"] != "GKP"]
    return df[(df["selected_by_percent"] < 10)].sort_values("xPts_per_match", ascending=False).head(5)


def get_safe_picks(df):
    return df[df["selected_by_percent"] > 20].sort_values("xPts_per_match", ascending=False).head(5)


def get_avoid_players(df):
    df = df[df["pos"] != "GKP"]
    return df[(df["selected_by_percent"] > 15) &
              (df["xPts_per_match"] < df["xPts_per_match"].quantile(0.4))].head(5)


def get_best_goalkeepers(df):
    return df[df["pos"] == "GKP"].sort_values("xPts_per_match", ascending=False).head(5)


def get_fixture_swing_teams(fixtures, teams, horizon=5):
    future_fx = fixtures[fixtures["finished"] == False]

    scores = []
    for _, t in teams.iterrows():
        tid = t["id"]

        team_fx = future_fx[(future_fx["team_h"] == tid) | (future_fx["team_a"] == tid)].head(horizon)
        if len(team_fx) == 0:
            continue

        avg_diff = team_fx[["team_h_difficulty", "team_a_difficulty"]].mean().mean()

        scores.append({"team": t["name"], "difficulty": round(avg_diff, 2)})

    df = pd.DataFrame(scores)
    return df.sort_values("difficulty").head(5), df.sort_values("difficulty", ascending=False).head(5)


# -----------------------
# UI START
# -----------------------
st.set_page_config(layout="wide")
st.title("⚽ FPL Analytics (Product UI)")

players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

pm = build_player_master(players, teams, element_types)

# Sidebar
st.sidebar.header("⚙️ Settings")
horizon = st.sidebar.slider("Fixture Horizon", 1, 10, 5)
form_weight = st.sidebar.slider("Form Weight", 0.0, 1.0, 0.3)
bonus_weight = st.sidebar.slider("Bonus Weight", 0.0, 0.5, 0.2)

pred = run_model(pm, fixtures, teams, horizon, form_weight, bonus_weight)
pred = add_value_columns(pred)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎯 Player Explorer", "🧠 Decision Engine"])

# -----------------------
# TAB 1: DASHBOARD
# -----------------------
with tab1:
    st.header("Team Overview")

    st.subheader("Season Strength")
    st.dataframe(build_team_table_from_fixtures(fixtures, teams))

    st.subheader("FPL Production")
    st.dataframe(build_team_fpl_production(players, teams))


# -----------------------
# TAB 2: PLAYER EXPLORER
# -----------------------
with tab2:
    st.header("Player Rankings")

    for pos in ["GKP", "DEF", "MID", "FWD"]:
        st.subheader(pos)
        dfp = pred[pred["pos"] == pos].sort_values("xPts_total", ascending=False).head(10)
        st.dataframe(dfp[["web_name", "team_name", "xPts_total", "xPts_per_match"]])


# -----------------------
# TAB 3: DECISION ENGINE
# -----------------------
with tab3:
    st.header("Decision Engine")

    st.subheader("👑 Captain Picks")
    st.dataframe(get_captain_picks(pred)[["web_name", "team_name", "xPts_per_match"]])

    st.subheader("💎 Differentials")
    st.dataframe(get_differentials(pred)[["web_name", "team_name", "xPts_per_match"]])

    st.subheader("🛡 Safe Picks")
    st.dataframe(get_safe_picks(pred)[["web_name", "team_name", "xPts_per_match"]])

    st.subheader("🚨 Avoid Players")
    st.dataframe(get_avoid_players(pred)[["web_name", "team_name", "xPts_per_match"]])

    st.subheader("🧤 Best Goalkeepers")
    st.dataframe(get_best_goalkeepers(pred)[["web_name", "team_name", "xPts_per_match"]])

    st.subheader("📅 Fixture Swings")
    good, bad = get_fixture_swing_teams(fixtures, teams, horizon)

    col1, col2 = st.columns(2)
    col1.dataframe(good)
    col2.dataframe(bad)
