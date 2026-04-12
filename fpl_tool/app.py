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


def get_avoid_players(df):
    return df[(df["pos"] != "GKP") &
              (df["selected_by_percent"] > 15) &
              (df["xPts_per_match"] < df["xPts_per_match"].quantile(0.4))].head(5)


def get_best_goalkeepers(df):
    return df[df["pos"] == "GKP"].sort_values("xPts_per_match", ascending=False).head(5)


# -----------------------
# TRANSFER OPTIMIZER V2
# -----------------------
def suggest_transfers_v2(current_team_names, pred_df, budget=1.0, max_transfers=2):

    squad = pred_df[pred_df["web_name"].isin(current_team_names)].copy()
    pool = pred_df[~pred_df["web_name"].isin(current_team_names)].copy()

    # Reduce search space
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

            # Position match
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
# UI START
# -----------------------
st.set_page_config(layout="wide")
st.title("⚽ FPL Analytics – v5 AI (DGW/BGW + Optimizer)")

players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

pm = build_player_master(players, teams, element_types)

# Sidebar
st.sidebar.header("⚙️ Settings")

horizon = st.sidebar.slider("Gameweek Horizon", 1, 10, 5)
form_weight = st.sidebar.slider("Form Weight", 0.0, 1.0, 0.3)
bonus_weight = st.sidebar.slider("Bonus Weight", 0.0, 0.5, 0.2)

pred = run_model(pm, fixtures, teams, horizon, form_weight, bonus_weight)

pred["selected_by_percent"] = pd.to_numeric(pred["selected_by_percent"], errors="coerce").fillna(0)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🎯 Player Explorer",
    "🧠 Decision Engine",
    "🔄 Transfer Optimizer"
])


# -----------------------
# TAB 1: DASHBOARD
# -----------------------
with tab1:
    st.header("📊 Team Overview")

    st.dataframe(
        fixtures[fixtures["finished"] == True][
            ["team_h", "team_a", "team_h_score", "team_a_score"]
        ].head(50)
    )


# -----------------------
# TAB 2: PLAYER EXPLORER
# -----------------------
with tab2:
    st.header("🎯 Player Rankings")

    min_minutes = st.slider("Minimum Minutes", 0, 2000, 500)

    for pos in ["GKP", "DEF", "MID", "FWD"]:
        st.subheader(pos)

        dfp = pred[(pred["pos"] == pos) & (pred["minutes"] >= min_minutes)]
        dfp = dfp.sort_values("xPts_total", ascending=False).head(10)

        st.dataframe(dfp[[
            "web_name", "team_name",
            "minutes", "exp_minutes",
            "fixtures_in_horizon",
            "xPts_total"
        ]])


# -----------------------
# TAB 3: DECISION ENGINE
# -----------------------
with tab3:
    st.header("🧠 Smart Decisions")

    df_filtered = pred[pred["minutes"] > 300]

    st.subheader("👑 Captain Picks")
    st.dataframe(get_captain_picks(df_filtered)[["web_name", "xPts_per_match"]])

    st.subheader("💎 Differentials")
    st.dataframe(get_differentials(df_filtered)[["web_name", "xPts_per_match"]])

    st.subheader("🛡 Safe Picks")
    st.dataframe(get_safe_picks(df_filtered)[["web_name", "xPts_per_match"]])

    st.subheader("🚨 Avoid Players")
    st.dataframe(get_avoid_players(df_filtered)[["web_name", "xPts_per_match"]])

    st.subheader("🧤 Best Goalkeepers")
    st.dataframe(get_best_goalkeepers(df_filtered)[["web_name", "xPts_per_match"]])


# -----------------------
# TAB 4: TRANSFER OPTIMIZER
# -----------------------
with tab4:
    st.header("🔄 Transfer Optimizer v2")

    squad_input = st.text_area(
        "Enter your squad (comma separated)",
        "Saka, Haaland, Salah"
    )

    budget = st.number_input("Extra Budget (£m)", 0.0, 10.0, 1.0)
    transfers = st.selectbox("Transfers", [1, 2], index=1)

    if st.button("Optimize Transfers"):

        squad_list = [x.strip() for x in squad_input.split(",")]

        result = suggest_transfers_v2(
            squad_list,
            pred,
            budget,
            transfers
        )

        if result.empty:
            st.warning("No better combination found.")
        else:
            st.dataframe(result)


st.success("✅ v5 AI + Optimizer v2 Active 🚀")
