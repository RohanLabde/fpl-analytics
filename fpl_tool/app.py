# app.py
import pandas as pd
import numpy as np
import requests
import streamlit as st

from fpl_tool.model import build_player_master, v2_expected_points

try:
    from fpl_tool.model import add_value_columns
except Exception:
    add_value_columns = None


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
# Streamlit UI
# -----------------------
st.set_page_config(page_title="FPL Analytics – Smarter Expected Points", layout="wide")
st.title("⚽ FPL Analytics – Smarter Expected Points")

players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

pm = build_player_master(players.copy(), teams.copy(), element_types.copy())

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Model & display settings")

horizon = st.sidebar.slider("Fixture horizon (matches)", 1, 10, 5)

rank_by_choice = st.sidebar.selectbox("Rank by", ["xPts_total", "xPts_per_match"], index=0)

min_minutes_for_leaderboards = st.sidebar.slider("Min historical minutes filter", 0, 2000, 300)
top_n_per_position = st.sidebar.number_input("Top N per position", 1, 20, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("Form & Momentum Controls")

form_window = st.sidebar.slider("Recent form window (matches)", 3, 10, 5)
form_weight = st.sidebar.slider("Form influence weight", 0.0, 1.0, 0.3, 0.05)
bonus_weight = st.sidebar.slider("Bonus influence weight", 0.0, 0.5, 0.2, 0.05)

# -----------------------
# Run model
# -----------------------
pred = v2_expected_points(
    pm.copy(),
    fixtures.copy(),
    teams.copy(),
    horizon=horizon,
    form_weight=form_weight,
    bonus_weight=bonus_weight,
)

# Safety: recompute total
pred["xPts_total"] = pred["xPts_per_match"] * horizon

if add_value_columns:
    pred = add_value_columns(pred)

pred["£m"] = pred["now_cost"] / 10
pred["selected_by_percent"] = pd.to_numeric(pred["selected_by_percent"], errors="coerce").fillna(0)

# -----------------------
# TEAM DASHBOARD
# -----------------------
st.header("📊 Team Strength & Form Dashboard")

st.subheader("🏆 Season Team Strength (All Matches)")
season_table = build_team_table_from_fixtures(fixtures, teams, last_n=None)
st.dataframe(season_table.reset_index(drop=True))

st.subheader("🔥 Recent Form (Team)")

recent_n = st.slider("Recent team form window (matches)", 3, 10, 5)

recent_form = build_team_table_from_fixtures(fixtures, teams, last_n=recent_n)
st.dataframe(recent_form.reset_index(drop=True))

st.subheader("🎯 FPL Production by Team (from player stats)")
prod = build_team_fpl_production(players, teams)
st.dataframe(prod.reset_index(drop=True))

# -----------------------
# PLAYER LEADERBOARDS
# -----------------------
st.header("🎯 Player Leaderboards (Form-adjusted)")

for pos in ["GKP", "DEF", "MID", "FWD"]:
    dfp = pred[pred["pos"] == pos].copy()

    if min_minutes_for_leaderboards > 0:
        dfp = dfp[dfp["minutes"] >= min_minutes_for_leaderboards]

    dfp = dfp.sort_values(rank_by_choice, ascending=False).head(top_n_per_position)

    st.subheader(f"Top {len(dfp)} {pos}s by {rank_by_choice}")

    cols = [
        "web_name", "team_name", "pos", "£m", "selected_by_percent",
        "xPts_per_match", "xPts_total", "form_factor", "bonus_factor", "minutes_factor"
    ]
    cols = [c for c in cols if c in dfp.columns]

    st.dataframe(dfp[cols].reset_index(drop=True))

st.success("✅ Form-adjusted xPts, team strength & momentum controls are live.")
