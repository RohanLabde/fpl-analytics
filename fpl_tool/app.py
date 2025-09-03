import streamlit as st
import pandas as pd
import requests

from fpl_tool.model import build_player_master, v2_expected_points, add_value_columns


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
st.set_page_config(page_title="FPL Analytics – Expected Points Model", layout="wide")

st.title("⚽ FPL Analytics – Smarter Expected Points")
st.caption(
    "Data: Official Fantasy Premier League API. "
    "Model uses xG, xA, clean sheet probability, saves & fixture horizon adjustments."
)

# Load data
players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

pm = build_player_master(players, teams, element_types)

# Sidebar settings
st.sidebar.header("Model Settings")
horizon = st.sidebar.slider("Fixture horizon (matches)", 1, 10, 5)

# Always use new V2 logic
pred = v2_expected_points(pm, fixtures, teams, horizon=horizon)
pred = add_value_columns(pred)


# --- Helper: format for display ---
def format_for_display(df, cols):
    df = df.copy()
    if "now_cost" in df.columns:
        df["now_cost"] = (df["now_cost"] / 10).round(1)  # show as £m
    if "selected_by_percent" in df.columns:
        df["sel_by_%"] = df["selected_by_percent"].astype(float).round(1)  # selection %
    return df[cols]


# --- Helper: Top picks by position ---
def show_top_by_position(df, top_n=10, gk_n=3):
    pos_map = {"GKP": gk_n, "DEF": top_n, "MID": top_n, "FWD": top_n}
    result = {}
    for pos, n in pos_map.items():
        subset = df[df["pos"] == pos].sort_values("xPts", ascending=False).head(n)
        result[pos] = subset
    return result


# --- Captaincy Picks ---
st.subheader("🎯 Captaincy picks (Top by xPts per position)")

captaincy_tables = show_top_by_position(pred, top_n=10, gk_n=3)
for pos, table in captaincy_tables.items():
    st.markdown(f"**Top {len(table)} {pos}s by xPts**")

    if pos in ["MID", "FWD"]:
        display_cols = ["web_name", "team_name", "pos", "now_cost", "sel_by_%", "xAttack", "att_factor", "xPts"]
    elif pos == "DEF":
        display_cols = ["web_name", "team_name", "pos", "now_cost", "sel_by_%", "xAttack", "cs_prob", "xPts"]
    elif pos == "GKP":
        display_cols = ["web_name", "team_name", "pos", "now_cost", "sel_by_%", "cs_prob", "xSaves", "xPts"]
    else:
        display_cols = ["web_name", "team_name", "pos", "now_cost", "sel_by_%", "xPts"]

    st.dataframe(format_for_display(table, display_cols).reset_index(drop=True))


# --- Value Picks ---
st.subheader("💼 Value picks (Top by xPts per million per position)")

value_tables = show_top_by_position(pred, top_n=10, gk_n=3)
for pos, table in value_tables.items():
    st.markdown(f"**Top {len(table)} {pos}s by xPts per million**")

    if pos in ["MID", "FWD"]:
        display_cols = ["web_name", "team_name", "pos", "now_cost", "sel_by_%", "xAttack", "att_factor", "xPts_per_m"]
    elif pos == "DEF":
        display_cols = ["web_name", "team_name", "pos", "now_cost", "sel_by_%", "xAttack", "cs_prob", "xPts_per_m"]
    elif pos == "GKP":
        display_cols = ["web_name", "team_name", "pos", "now_cost", "sel_by_%", "cs_prob", "xSaves", "xPts_per_m"]
    else:
        display_cols = ["web_name", "team_name", "pos", "now_cost", "sel_by_%", "xPts_per_m"]

    st.dataframe(format_for_display(table, display_cols).reset_index(drop=True))


# --- Analyze My Squad ---
st.subheader("🧩 Analyze My 15-man Squad")

player_options = {
    int(r.id): f"{r.web_name} ({r.team_name}, {r.pos}, £{r.now_cost/10}m, {r.selected_by_percent}% selected)"
    for r in pred.itertuples()
}

squad_ids = st.multiselect(
    "Select your 15 players",
    options=list(player_options.keys()),
    format_func=lambda x: player_options[x]
)

bank = st.number_input("Bank (money in the bank, £m)", min_value=0.0, step=0.1)

if len(squad_ids) == 15:
    squad_df = pred[pred["id"].isin(squad_ids)].copy()

    # Pick Best XI (basic heuristic: 1 GKP, 3 DEF, 4 MID, 3 FWD)
    best_xi = []
    best_xi.append(squad_df[squad_df["pos"] == "GKP"].sort_values("xPts", ascending=False).head(1))
    best_xi.append(squad_df[squad_df["pos"] == "DEF"].sort_values("xPts", ascending=False).head(3))
    best_xi.append(squad_df[squad_df["pos"] == "MID"].sort_values("xPts", ascending=False).head(4))
    best_xi.append(squad_df[squad_df["pos"] == "FWD"].sort_values("xPts", ascending=False).head(3))

    best_xi = pd.concat(best_xi).sort_values("xPts", ascending=False).head(11)

    st.markdown("### ✅ Best XI (sorted by xPts):")
    st.dataframe(format_for_display(best_xi, ["web_name", "pos", "team_name", "now_cost", "sel_by_%", "xPts"]))

    captain = best_xi.iloc[0]["web_name"]
    vice_captain = best_xi.iloc[1]["web_name"]
    st.success(f"⭐ Recommended Captain: **{captain}** | Vice Captain: **{vice_captain}**")
else:
    st.info("Please select exactly 15 players to analyze transfers.")
