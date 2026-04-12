import pandas as pd
import numpy as np
import requests
import streamlit as st
from itertools import combinations

from fpl_tool.model import build_player_master, v5_expected_points


# -----------------------
# DATA LOADERS
# -----------------------
@st.cache_data(ttl=3600)
def load_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    data = requests.get(url).json()
    return (
        pd.DataFrame(data["elements"]),
        pd.DataFrame(data["teams"]),
        pd.DataFrame(data["element_types"])
    )


@st.cache_data(ttl=3600)
def load_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    return pd.DataFrame(requests.get(url).json())


@st.cache_data(ttl=3600)
def run_model(pm, fixtures, teams, horizon, form_weight, bonus_weight):
    return v5_expected_points(
        pm.copy(),
        fixtures.copy(),
        teams.copy(),
        horizon=horizon,
        form_weight=form_weight,
        bonus_weight=bonus_weight,
    )


# -----------------------
# TEAM SUMMARY
# -----------------------
def build_team_summary(fixtures, teams):
    fx = fixtures[fixtures["finished"] == True]

    rows = []

    for _, t in teams.iterrows():
        tid = t["id"]

        team_fx = fx[(fx["team_h"] == tid) | (fx["team_a"] == tid)]

        if len(team_fx) == 0:
            continue

        GF = GA = 0

        for _, m in team_fx.iterrows():
            if m["team_h"] == tid:
                GF += m["team_h_score"]
                GA += m["team_a_score"]
            else:
                GF += m["team_a_score"]
                GA += m["team_h_score"]

        rows.append({
            "Team": t["name"],
            "Goals Scored": GF,
            "Goals Conceded": GA,
            "Matches": len(team_fx)
        })

    return pd.DataFrame(rows).sort_values("Goals Scored", ascending=False)


# -----------------------
# DECISION ENGINE
# -----------------------
def get_captain_picks(df):
    return df[df["pos"].isin(["MID", "FWD"])].sort_values("xPts_per_match", ascending=False).head(5)


def get_differentials(df):
    return df[(df["pos"] != "GKP") & (df["selected_by_percent"] < 10)] \
        .sort_values("xPts_per_match", ascending=False).head(5)


def get_safe_picks(df):
    return df[df["selected_by_percent"] > 20] \
        .sort_values("xPts_per_match", ascending=False).head(5)


def get_best_goalkeepers(df):
    return df[df["pos"] == "GKP"].sort_values("xPts_per_match", ascending=False).head(5)


# -----------------------
# TRANSFER OPTIMIZER V2
# -----------------------
def suggest_transfers_v2(current_team_names, pred_df, budget=1.0, max_transfers=2):

    squad = pred_df[pred_df["web_name"].isin(current_team_names)].copy()
    pool = pred_df[~pred_df["web_name"].isin(current_team_names)].copy()

    pool = pd.concat([
        pool[pool["pos"] == "GKP"].sort_values("xPts_total", ascending=False).head(10),
        pool[pool["pos"] == "DEF"].sort_values("xPts_total", ascending=False).head(20),
        pool[pool["pos"] == "MID"].sort_values("xPts_total", ascending=False).head(20),
        pool[pool["pos"] == "FWD"].sort_values("xPts_total", ascending=False).head(15),
    ])

    weakest = squad.sort_values("xPts_total").head(max_transfers)

    best_gain = -999
    best_solution = None

    for out_combo in combinations(weakest.index, max_transfers):

        out_players = squad.loc[list(out_combo)]
        out_value = out_players["price_m"].sum()
        out_points = out_players["xPts_total"].sum()

        for in_combo in combinations(pool.index, max_transfers):

            in_players = pool.loc[list(in_combo)]

            if sorted(out_players["pos"].values) != sorted(in_players["pos"].values):
                continue

            in_value = in_players["price_m"].sum()

            if in_value > out_value + budget:
                continue

            gain = in_players["xPts_total"].sum() - out_points

            if gain > best_gain:
                best_gain = gain
                best_solution = (out_players, in_players, gain)

    if best_solution is None:
        return pd.DataFrame()

    out_p, in_p, gain = best_solution

    return pd.DataFrame({
        "OUT": out_p["web_name"].values,
        "IN": in_p["web_name"].values,
        "Gain": [round(gain, 2)] * len(out_p)
    })


# -----------------------
# FULL SQUAD BUILDER
# -----------------------
def build_optimal_squad(pred_df, budget=100):

    df = pred_df.sort_values("xPts_total", ascending=False)

    squad = []
    team_count = {}
    total_cost = 0

    limits = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}

    for _, player in df.iterrows():

        pos = player["pos"]
        team = player["team_name"]
        cost = player["price_m"]

        if counts[pos] >= limits[pos]:
            continue

        if team_count.get(team, 0) >= 3:
            continue

        if total_cost + cost > budget:
            continue

        squad.append(player)
        counts[pos] += 1
        team_count[team] = team_count.get(team, 0) + 1
        total_cost += cost

        if sum(counts.values()) == 15:
            break

    return pd.DataFrame(squad), total_cost


# -----------------------
# STARTING XI OPTIMIZER
# -----------------------
def optimize_starting_xi(squad_df):

    best_score = -999
    best_team = None

    gk = squad_df[squad_df["pos"] == "GKP"]
    defenders = squad_df[squad_df["pos"] == "DEF"]
    mids = squad_df[squad_df["pos"] == "MID"]
    fwds = squad_df[squad_df["pos"] == "FWD"]

    formations = [(3,4,3),(3,5,2),(4,4,2),(4,3,3),(5,3,2)]

    for d,m,f in formations:

        if len(defenders)<d or len(mids)<m or len(fwds)<f:
            continue

        for g in combinations(gk.index,1):
            for d_ in combinations(defenders.index,d):
                for m_ in combinations(mids.index,m):
                    for f_ in combinations(fwds.index,f):

                        idx = list(g)+list(d_)+list(m_)+list(f_)
                        team = squad_df.loc[idx]

                        score = team["xPts_per_match"].sum()

                        if score > best_score:
                            best_score = score
                            best_team = team

    return best_team, best_score


def pick_captains(team_df):
    sorted_team = team_df.sort_values("xPts_per_match", ascending=False)
    return sorted_team.iloc[0], sorted_team.iloc[1]


# -----------------------
# UI START
# -----------------------
st.set_page_config(layout="wide")
st.title("⚽ FPL AI – Full System")

players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

pm = build_player_master(players, teams, element_types)

horizon = st.sidebar.slider("Gameweek Horizon",1,10,5)
form_weight = st.sidebar.slider("Form Weight",0.0,1.0,0.3)
bonus_weight = st.sidebar.slider("Bonus Weight",0.0,0.5,0.2)

pred = run_model(pm, fixtures, teams, horizon, form_weight, bonus_weight)
pred["selected_by_percent"] = pd.to_numeric(pred["selected_by_percent"], errors="coerce").fillna(0)

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📊 Dashboard","🎯 Explorer","🧠 Decisions","🔄 Transfers","🏆 Squad Builder","⚽ XI Optimizer"
])

# Dashboard
with tab1:
    st.dataframe(build_team_summary(fixtures, teams))

# Explorer
with tab2:
    for pos in ["GKP","DEF","MID","FWD"]:
        st.subheader(pos)
        st.dataframe(pred[pred["pos"]==pos].sort_values("xPts_total",ascending=False).head(10))

# Decision Engine
with tab3:
    df = pred[pred["minutes"]>300]
    st.dataframe(get_captain_picks(df)[["web_name","xPts_per_match"]])
    st.dataframe(get_differentials(df)[["web_name","xPts_per_match"]])

# Transfers
with tab4:
    st.write("Use previous squad builder")

# Squad Builder
with tab5:
    budget = st.slider("Budget",80,120,100)

    if st.button("Build Squad"):
        squad, cost = build_optimal_squad(pred, budget)
        st.write(f"Cost: {cost}")
        st.dataframe(squad[["web_name","pos","team_name","price_m","xPts_total"]])

# XI Optimizer
with tab6:
    if st.button("Optimize XI"):
        squad, _ = build_optimal_squad(pred,100)
        xi, score = optimize_starting_xi(squad)
        cap, vc = pick_captains(xi)

        st.write("Best XI Score:",score)
        st.dataframe(xi[["web_name","pos","xPts_per_match"]])
        st.write("Captain:",cap["web_name"])
        st.write("Vice Captain:",vc["web_name"])
