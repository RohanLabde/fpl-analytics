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


# -----------------------
# MODEL CACHE
# -----------------------
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
            rows.append({
                "team": name, "played": 0, "points": 0, "win": 0, "draw": 0, "loss": 0,
                "GF": 0, "GA": 0, "CS": 0,
                "GF_per_match": 0.0, "GA_per_match": 0.0, "CS_%": 0.0
            })
            continue

        GF = 0
        GA = 0
        CS = 0
        win = 0
        draw = 0
        loss = 0

        for _, m in team_fx.iterrows():
            if m["team_h"] == tid:
                gf = m["team_h_score"]
                ga = m["team_a_score"]
            else:
                gf = m["team_a_score"]
                ga = m["team_h_score"]

            GF += gf
            GA += ga

            if ga == 0:
                CS += 1
            if gf > ga:
                win += 1
            elif gf == ga:
                draw += 1
            else:
                loss += 1

        points = win * 3 + draw

        rows.append({
            "team": name,
            "played": played,
            "points": points,
            "win": win,
            "draw": draw,
            "loss": loss,
            "GF": GF,
            "GA": GA,
            "CS": CS,
            "GF_per_match": round(GF / played, 2),
            "GA_per_match": round(GA / played, 2),
            "CS_%": round(100 * CS / played, 1),
        })

    df = pd.DataFrame(rows).sort_values("points", ascending=False)
    return df


def build_team_fpl_production(players, teams):
    df = players.copy()

    numeric_cols = ["goals_scored", "assists", "bonus", "expected_goals"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    agg = df.groupby("team").agg({
        "goals_scored": "sum",
        "assists": "sum",
        "bonus": "sum",
        "expected_goals": "sum",
    }).reset_index()

    team_map = teams.set_index("id")["name"].to_dict()
    agg["team"] = agg["team"].map(team_map)

    agg = agg.rename(columns={
        "team": "team_name",
        "goals_scored": "Goals",
        "assists": "Assists",
        "bonus": "Bonus",
        "expected_goals": "xG",
    })

    agg["xG"] = agg["xG"].round(2)

    return agg.sort_values("Goals", ascending=False)


# -----------------------
# DECISION ENGINE
# -----------------------
def get_captain_picks(df):
    return df.sort_values("xPts_per_match", ascending=False).head(5)


def get_differentials(df, max_ownership=10):
    return df[
        (df["selected_by_percent"] < max_ownership) &
        (df["xPts_per_match"] > df["xPts_per_match"].quantile(0.75))
    ].sort_values("xPts_per_match", ascending=False).head(5)


def get_safe_picks(df):
    return df[
        (df["selected_by_percent"] > 20)
    ].sort_values("xPts_per_match", ascending=False).head(5)


def get_avoid_players(df):
    return df[
        (df["selected_by_percent"] > 15) &
        (df["xPts_per_match"] < df["xPts_per_match"].quantile(0.4))
    ].sort_values("selected_by_percent", ascending=False).head(5)


def get_fixture_swing_teams(fixtures, teams, horizon=5):
    future_fx = fixtures[fixtures["finished"] == False].copy()

    swing_scores = []

    for _, t in teams.iterrows():
        tid = t["id"]
        name = t["name"]

        team_fx = future_fx[
            (future_fx["team_h"] == tid) | (future_fx["team_a"] == tid)
        ].head(horizon)

        if len(team_fx) == 0:
            continue

        difficulty = []

        for _, fx in team_fx.iterrows():
            if fx["team_h"] == tid:
                diff = fx.get("team_h_difficulty", 3)
            else:
                diff = fx.get("team_a_difficulty", 3)

            difficulty.append(diff)

        avg_diff = np.mean(difficulty)

        swing_scores.append({
            "team": name,
            "fixture_difficulty": round(avg_diff, 2)
        })

    df = pd.DataFrame(swing_scores)

    good = df.sort_values("fixture_difficulty").head(5)
    bad = df.sort_values("fixture_difficulty", ascending=False).head(5)

    return good, bad


# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="FPL Analytics – Smarter Expected Points", layout="wide")
st.title("⚽ FPL Analytics – Smarter Expected Points (v3 + Decision Engine)")

players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

pm = build_player_master(players.copy(), teams.copy(), element_types.copy())


# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Model Settings")

horizon = st.sidebar.slider("Fixture horizon (matches)", 1, 10, 5)
rank_by_choice = st.sidebar.selectbox("Rank by", ["xPts_total", "xPts_per_match"], index=0)

min_minutes_for_leaderboards = st.sidebar.slider("Min minutes filter", 0, 2000, 300)
top_n_per_position = st.sidebar.number_input("Top N per position", 1, 20, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("Adjustment Controls")

form_weight = st.sidebar.slider("Form weight", 0.0, 1.0, 0.3, 0.05)
bonus_weight = st.sidebar.slider("Bonus weight", 0.0, 0.5, 0.2, 0.05)


# -----------------------
# Run Model
# -----------------------
pred = run_model(pm, fixtures, teams, horizon, form_weight, bonus_weight)

pred["£m"] = pred["price_m"]
pred["selected_by_percent"] = pd.to_numeric(pred["selected_by_percent"], errors="coerce").fillna(0)

pred = add_value_columns(pred)


# -----------------------
# Debug
# -----------------------
with st.expander("🔍 Debug: Raw Model Output"):
    st.dataframe(pred.head(50))


# -----------------------
# TEAM DASHBOARD
# -----------------------
st.header("📊 Team Strength & Form")

st.subheader("🏆 Season Strength")
season_table = build_team_table_from_fixtures(fixtures, teams)
st.dataframe(season_table.reset_index(drop=True))

st.subheader("🔥 Recent Form")
recent_n = st.slider("Recent matches window", 3, 10, 5)
recent_form = build_team_table_from_fixtures(fixtures, teams, last_n=recent_n)
st.dataframe(recent_form.reset_index(drop=True))

st.subheader("🎯 FPL Production")
prod = build_team_fpl_production(players, teams)
st.dataframe(prod.reset_index(drop=True))


# -----------------------
# PLAYER LEADERBOARDS
# -----------------------
st.header("🎯 Player Leaderboards (v3 Model)")

for pos in ["GKP", "DEF", "MID", "FWD"]:
    dfp = pred[pred["pos"] == pos].copy()

    if min_minutes_for_leaderboards > 0:
        dfp = dfp[dfp["minutes"] >= min_minutes_for_leaderboards]

    dfp = dfp.sort_values(rank_by_choice, ascending=False).head(top_n_per_position)

    st.subheader(f"Top {len(dfp)} {pos}s")

    cols = [
        "web_name", "team_name", "pos", "£m",
        "selected_by_percent", "xPts_per_match", "xPts_total",
        "xPts_per_m", "xPts_total_per_m"
    ]

    cols = [c for c in cols if c in dfp.columns]

    st.dataframe(dfp[cols].reset_index(drop=True))


# -----------------------
# DECISION ENGINE UI
# -----------------------
st.header("🧠 FPL Decision Engine")

# Captain Picks
st.subheader("👑 Best Captain Picks")
captains = get_captain_picks(pred)
st.dataframe(captains[["web_name", "team_name", "xPts_per_match"]])

# Differentials
st.subheader("💎 Hidden Gems (Differentials)")
diffs = get_differentials(pred)
st.dataframe(diffs[["web_name", "team_name", "selected_by_percent", "xPts_per_match"]])

# Safe Picks
st.subheader("🛡️ Safe Picks (Template Players)")
safe = get_safe_picks(pred)
st.dataframe(safe[["web_name", "team_name", "selected_by_percent", "xPts_per_match"]])

# Avoid Players
st.subheader("🚨 Avoid These Players")
avoid = get_avoid_players(pred)
st.dataframe(avoid[["web_name", "team_name", "selected_by_percent", "xPts_per_match"]])

# Fixture Swings
st.subheader("📅 Fixture Swings")

good_teams, bad_teams = get_fixture_swing_teams(fixtures, teams, horizon)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ Good Fixture Runs")
    st.dataframe(good_teams)

with col2:
    st.markdown("### ❌ Tough Fixture Runs")
    st.dataframe(bad_teams)


st.success("✅ v3 Model + Decision Engine Active. You now have a competitive edge.")
