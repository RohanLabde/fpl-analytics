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


# --- Format helper ---
def format_for_display(df, cols):
    out = df.copy()
    if "now_cost" in out.columns:
        out["now_cost"] = out["now_cost"] / 10  # convert to millions
        out.rename(columns={"now_cost": "£m"}, inplace=True)
    if "selected_by_percent" in out.columns:
        out["sel_by_%"] = out["selected_by_percent"].astype(float)
    return out[cols]


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


# --- Helper: Top picks by position ---
def show_top_by_position(df, col, top_n=10, gk_n=3):
    pos_map = {"GKP": gk_n, "DEF": top_n, "MID": top_n, "FWD": top_n}
    result = {}
    for pos, n in pos_map.items():
        subset = df[df["pos"] == pos].sort_values(col, ascending=False).head(n)
        result[pos] = subset
    return result


# --- Captaincy Picks ---
st.subheader("🎯 Captaincy picks (Top by xPts per position)")

captaincy_tables = show_top_by_position(pred, "xPts", top_n=10, gk_n=3)
for pos, table in captaincy_tables.items():
    st.markdown(f"**Top {len(table)} {pos}s by xPts**")

    if pos in ["MID", "FWD"]:
        display_cols = ["web_name", "team_name", "pos", "£m", "xAttack", "att_factor", "xPts"]
    elif pos == "DEF":
        display_cols = ["web_name", "team_name", "pos", "£m", "xAttack", "cs_prob", "xPts"]
    elif pos == "GKP":
        display_cols = ["web_name", "team_name", "pos", "£m", "cs_prob", "xSaves", "xPts"]
    else:
        display_cols = ["web_name", "team_name", "pos", "£m", "xPts"]

    st.dataframe(format_for_display(table, display_cols).reset_index(drop=True))


# --- Value Picks ---
st.subheader("💼 Value picks (Top by xPts per million per position)")

value_tables = show_top_by_position(pred, "xPts_per_m", top_n=10, gk_n=3)
for pos, table in value_tables.items():
    st.markdown(f"**Top {len(table)} {pos}s by xPts per million**")

    if pos in ["MID", "FWD"]:
        display_cols = ["web_name", "team_name", "pos", "£m", "xAttack", "att_factor", "xPts_per_m"]
    elif pos == "DEF":
        display_cols = ["web_name", "team_name", "pos", "£m", "xAttack", "cs_prob", "xPts_per_m"]
    elif pos == "GKP":
        display_cols = ["web_name", "team_name", "pos", "£m", "cs_prob", "xSaves", "xPts_per_m"]
    else:
        display_cols = ["web_name", "team_name", "pos", "£m", "xPts_per_m"]

    st.dataframe(format_for_display(table, display_cols).reset_index(drop=True))


# --- Analyze My Squad ---
st.subheader("🧩 Analyze My 15-man Squad")

player_options = {
    int(r.id): f"{r.web_name} ({r.team_name}, {r.pos}, £{r.now_cost/10}m)"
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

    # Pick Best XI (heuristic)
    best_xi = []
    best_xi.append(squad_df[squad_df["pos"] == "GKP"].sort_values("xPts", ascending=False).head(1))
    best_xi.append(squad_df[squad_df["pos"] == "DEF"].sort_values("xPts", ascending=False).head(3))
    best_xi.append(squad_df[squad_df["pos"] == "MID"].sort_values("xPts", ascending=False).head(4))
    best_xi.append(squad_df[squad_df["pos"] == "FWD"].sort_values("xPts", ascending=False).head(3))
    best_xi = pd.concat(best_xi).sort_values("xPts", ascending=False).head(11)

    st.markdown("### ✅ Best XI (sorted by xPts):")
    st.dataframe(format_for_display(best_xi, ["web_name", "pos", "team_name", "£m", "sel_by_%", "xPts"]))

    captain = best_xi.iloc[0]["web_name"]
    vice_captain = best_xi.iloc[1]["web_name"]
    st.success(f"⭐ Recommended Captain: **{captain}** | Vice Captain: **{vice_captain}**")

    # Subs
    subs = squad_df[~squad_df["id"].isin(best_xi["id"])].sort_values("xPts", ascending=False)
    st.markdown("### 🪑 Subs (bench, sorted by xPts):")
    st.dataframe(format_for_display(subs, ["web_name", "pos", "team_name", "£m", "sel_by_%", "xPts"]))

    # --- Transfer Suggestions ---
    st.markdown("---")
    st.subheader("🔁 Suggested Transfers")

    current_xi_pts = best_xi["xPts"].sum()
    transfer_candidates = []

    for out_id in squad_ids:
        out_player = pred[pred["id"] == out_id].iloc[0]
        budget_available = bank * 10 + out_player["now_cost"]

        candidates = pred[
            (pred["pos"] == out_player["pos"]) &
            (~pred["id"].isin(squad_ids)) &
            (pred["now_cost"] <= budget_available)
        ]

        if candidates.empty:
            continue

        in_player = candidates.sort_values("xPts", ascending=False).iloc[0]

        # Simulate new XI
        new_squad_ids = [pid for pid in squad_ids if pid != out_id] + [in_player["id"]]
        new_squad_df = pred[pred["id"].isin(new_squad_ids)]

        new_xi = []
        new_xi.append(new_squad_df[new_squad_df["pos"] == "GKP"].sort_values("xPts", ascending=False).head(1))
        new_xi.append(new_squad_df[new_squad_df["pos"] == "DEF"].sort_values("xPts", ascending=False).head(3))
        new_xi.append(new_squad_df[new_squad_df["pos"] == "MID"].sort_values("xPts", ascending=False).head(4))
        new_xi.append(new_squad_df[new_squad_df["pos"] == "FWD"].sort_values("xPts", ascending=False).head(3))
        new_xi = pd.concat(new_xi).head(11)

        new_pts = new_xi["xPts"].sum()
        gain = new_pts - current_xi_pts

        if gain > 0:
            transfer_candidates.append((gain, out_player, in_player, new_pts))

    transfer_candidates = sorted(transfer_candidates, key=lambda x: x[0], reverse=True)

    if transfer_candidates:
        st.markdown("#### 💡 Top 2 Transfer Suggestions:")
        for gain, out_p, in_p, new_pts in transfer_candidates[:2]:
            st.success(
                f"**{out_p['web_name']} ➝ {in_p['web_name']}** "
                f"(+{gain:.2f} xPts, new XI total = {new_pts:.2f})"
            )

        # Dropdown for custom OUT
        st.markdown("#### 🎯 Choose a player to transfer OUT:")
        out_choice = st.selectbox("Select player to sell", [p[1]["web_name"] for p in transfer_candidates])
        if out_choice:
            chosen = [t for t in transfer_candidates if t[1]["web_name"] == out_choice][0]
            gain, out_p, in_p, new_pts = chosen
            st.info(
                f"Best replacement for **{out_p['web_name']}** ➝ **{in_p['web_name']}** "
                f"(+{gain:.2f} xPts, new XI total = {new_pts:.2f})"
            )
    else:
        st.info("No beneficial transfers found within your squad & budget.")
else:
    st.info("Please select exactly 15 players to analyze transfers.")
