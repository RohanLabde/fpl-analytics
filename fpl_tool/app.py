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
    "V2 adds minutes + Poisson clean sheets + attacking proxy, with fixture horizon support."
)

# Load data
players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

pm = build_player_master(players, teams, element_types)

# Sidebar settings
st.sidebar.header("Model Settings")
horizon = st.sidebar.slider("Fixture horizon (matches) [V2]", 1, 10, 5)

# Always use V2 for expected points
pred = v2_expected_points(pm, fixtures, teams, horizon=horizon)

# Add value metric
pred["xPts_per_m"] = pred["xPts"] / (pred["now_cost"] / 10)


# --- Helper: Top picks by position ---
def show_top_by_position(df, col, top_n=5, gk_n=3):
    """Display top players per position"""
    pos_map = {"GK": gk_n, "DEF": top_n, "MID": top_n, "FWD": top_n}
    tables = []
    for pos, n in pos_map.items():
        subset = df[df["pos"] == pos].sort_values(col, ascending=False).head(n)
        tables.append(subset[["web_name", "pos", "team_name", "now_cost", "form", col]])
    return pd.concat(tables)


# --- Captaincy Picks ---
st.subheader("🎯 Captaincy picks (Top by xPts per position)")
top_captaincy = show_top_by_position(pred, "xPts")
st.dataframe(top_captaincy.reset_index(drop=True))

# --- Value Picks ---
st.subheader("💼 Value picks (Top by xPts per million per position)")
top_value = show_top_by_position(pred, "xPts_per_m")
st.dataframe(top_value.reset_index(drop=True))


# --- Analyze My Squad (Best XI) ---
st.subheader("🧩 Analyze My 15-man Squad")

player_options = {
    int(r.id): f"{r.web_name} ({r.team_name}, {r.pos}, £{r.now_cost/10}m)"
    for r in pred.itertuples()
}

squad_ids = st.multiselect("Select your 15 players", options=list(player_options.keys()), 
                           format_func=lambda x: player_options[x])

bank = st.number_input("Bank (money in the bank, £m)", min_value=0.0, step=0.1)

if len(squad_ids) == 15:
    squad_df = pred[pred["id"].isin(squad_ids)].copy()

    # Pick best XI (2-5 DEF, 2-5 MID, 1-3 FWD, 1 GK)
    best_xi = []
    best_xi.append(squad_df[squad_df["pos"] == "GK"].sort_values("xPts", ascending=False).head(1))
    best_xi.append(squad_df[squad_df["pos"] == "DEF"].sort_values("xPts", ascending=False).head(3))
    best_xi.append(squad_df[squad_df["pos"] == "MID"].sort_values("xPts", ascending=False).head(4))
    best_xi.append(squad_df[squad_df["pos"] == "FWD"].sort_values("xPts", ascending=False).head(3))

    best_xi = pd.concat(best_xi).sort_values("xPts", ascending=False).head(11)

    st.markdown("### ✅ Best XI (sorted by xPts):")
    st.dataframe(best_xi[["web_name", "pos", "team_name", "now_cost", "xPts"]])

    captain = best_xi.iloc[0]["web_name"]
    vice_captain = best_xi.iloc[1]["web_name"]
    st.success(f"⭐ Recommended Captain: **{captain}** | Vice Captain: **{vice_captain}**")

else:
    st.info("Please select exactly 15 players to analyze your squad.")
