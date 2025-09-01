import streamlit as st
import pandas as pd
import requests

from fpl_tool.model import build_player_master, baseline_expected_points, v2_expected_points


# --- Load FPL API data ---
@st.cache_data(ttl=3600)
def load_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    data = r.json()

    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    element_types = pd.DataFrame(data["element_types"])

    return players, teams, element_types


@st.cache_data(ttl=3600)
def load_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    r = requests.get(url)
    return pd.DataFrame(r.json())


# --- Streamlit UI ---
st.set_page_config(page_title="FPL Analytics – Fast Decisions", layout="wide")

st.title("⚽ FPL Analytics – Fast Decisions")
st.caption(
    "Data: Official Fantasy Premier League API. "
    "Toggle V2 for smarter xPts (minutes + Poisson clean sheets + attacking proxy)."
)

# Load data
players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

pm = build_player_master(players, teams, element_types)

# Sidebar settings
st.sidebar.header("Model Settings")
use_v2 = st.sidebar.checkbox("Use V2 xPts (Poisson + minutes + roles)", value=True)
horizon = st.sidebar.slider("Fixture horizon (matches) [V2]", 1, 10, 5)

# Run model
if use_v2:
    pred = v2_expected_points(pm, fixtures, teams, horizon=horizon)
else:
    pred = baseline_expected_points(pm)

# Add value metric
pred["xPts_per_m"] = pred["xPts"] / (pred["now_cost"] / 10)

# --- Display: Top picks by position ---
st.subheader("🎯 Captaincy picks (Top by xPts per position)")

def show_top_by_position(df, col, top_n=5, gk_n=3):
    """Display top players per position"""
    pos_map = {"GK": gk_n, "DEF": top_n, "MID": top_n, "FWD": top_n}
    tables = []
    for pos, n in pos_map.items():
        subset = df[df["pos"] == pos].sort_values(col, ascending=False).head(n)
        tables.append(subset[["web_name", "pos", "team_name", "now_cost", "form", col]])
    return pd.concat(tables)

top_captaincy = show_top_by_position(pred, "xPts")
st.dataframe(top_captaincy.reset_index(drop=True))

st.subheader("💼 Value picks (Top by xPts per million per position)")
top_value = show_top_by_position(pred, "xPts_per_m")
st.dataframe(top_value.reset_index(drop=True))
