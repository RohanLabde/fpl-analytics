# app.py
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

    # create a consistent selected_by_percent column name if present
    if "selected_by_percent" in players.columns:
        players["selected_by_percent"] = pd.to_numeric(players["selected_by_percent"], errors="coerce").fillna(0.0)
    elif "selected_by_percent" not in players.columns and "selected_by" in players.columns:
        players["selected_by_percent"] = pd.to_numeric(players["selected_by"], errors="coerce").fillna(0.0)
    else:
        players["selected_by_percent"] = 0.0

    return players, teams, element_types


@st.cache_data(ttl=3600)
def load_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    r = requests.get(url)
    return pd.DataFrame(r.json())


# --- UI helpers ---
def pretty_money_series(s: pd.Series) -> pd.Series:
    """Convert tenths to £m and format floats for display."""
    out = (s / 10).round(1)
    return out


def pretty_sel_percent(s: pd.Series) -> pd.Series:
    """Format selected_by_percent values (float 0..100 or 0..1 handled)."""
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    # if value looks like 0..1 scale convert to pct
    if s.max() <= 1.0:
        s = s * 100.0
    return s.round(1)


def format_for_display(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    if "now_cost" in out.columns:
        out["£m"] = pretty_money_series(out["now_cost"])
    if "selected_by_percent" in out.columns:
        out["sel_by_%"] = pretty_sel_percent(out["selected_by_percent"])
    # make sure both xPts columns exist (model should provide them)
    if "xPts_per_match" not in out.columns:
        out["xPts_per_match"] = 0.0
    if "xPts_total" not in out.columns:
        out["xPts_total"] = out.get("xPts", 0.0)
    # reorder/return only requested columns that exist
    cols_existing = [c for c in cols if c in out.columns]
    return out[cols_existing]


# --- Streamlit UI ---
st.set_page_config(page_title="FPL Analytics – Expected Points Model", layout="wide")
st.title("⚽ FPL Analytics – Smarter Expected Points")
st.caption("Data: Official Fantasy Premier League API. Model uses xG/xA fallbacks, clean-sheet proxy, saves & horizon.")

# Load data
players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

pm = build_player_master(players, teams, element_types)

# Sidebar settings
st.sidebar.header("Model Settings")
horizon = st.sidebar.slider("Fixture horizon (matches)", 1, 10, 5)

# Compute predictions
pred = v2_expected_points(pm, fixtures, teams, horizon=horizon)
pred = add_value_columns(pred)

# Helper: top by position
def show_top_by_position(df: pd.DataFrame, sort_col: str, top_n=10, gk_n=3) -> dict:
    pos_map = {"GKP": gk_n, "DEF": top_n, "MID": top_n, "FWD": top_n}
    result = {}
    for pos, n in pos_map.items():
        subset = df[df["pos"] == pos].sort_values(sort_col, ascending=False).head(n)
        result[pos] = subset
    return result


# --- Captaincy Picks (show both xPts_per_match and xPts_total) ---
st.subheader("🎯 Captaincy picks (Top by xPts per position)")
captaincy_tables = show_top_by_position(pred, "xPts", top_n=10, gk_n=3)

for pos, table in captaincy_tables.items():
    st.markdown(f"**Top {len(table)} {pos}s by xPts**")
    # choose sensible display columns per position (include both metrics)
    if pos in ["MID", "FWD"]:
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", "att_factor", "xPts_per_match", "xPts_total"]
    elif pos == "DEF":
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", "cs_prob", "xPts_per_match", "xPts_total"]
    elif pos == "GKP":
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "cs_prob", "xSaves", "xPts_per_match", "xPts_total"]
    else:
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xPts_per_match", "xPts_total"]

    st.dataframe(format_for_display(table, display_cols).reset_index(drop=True))


# --- Value Picks (show both per-match and total value) ---
st.subheader("💼 Value picks (Top by xPts per million per position)")
value_tables = show_top_by_position(pred, "xPts_per_m", top_n=10, gk_n=3)

for pos, table in value_tables.items():
    st.markdown(f"**Top {len(table)} {pos}s by xPts per million**")
    if pos in ["MID", "FWD"]:
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", "att_factor", "xPts_per_match", "xPts_total", "xPts_per_m", "xPts_per_m_match"]
    elif pos == "DEF":
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", "cs_prob", "xPts_per_match", "xPts_total", "xPts_per_m", "xPts_per_m_match"]
    elif pos == "GKP":
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "cs_prob", "xSaves", "xPts_per_match", "xPts_total", "xPts_per_m", "xPts_per_m_match"]
    else:
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xPts_per_match", "xPts_total", "xPts_per_m", "xPts_per_m_match"]

    st.dataframe(format_for_display(table, display_cols).reset_index(drop=True))


# --- Analyze My Squad ---
st.subheader("🧩 Analyze My 15-man Squad")

player_options = {
    int(r.id): f"{r.web_name} ({r.team_name}, {r.pos}, £{r.now_cost/10:.1f}m, {r.selected_by_percent:.1f}%)"
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

    # Best XI heuristic: 1 GKP, 3 DEF, 4 MID, 3 FWD
    best_xi = []
    best_xi.append(squad_df[squad_df["pos"] == "GKP"].sort_values("xPts", ascending=False).head(1))
    best_xi.append(squad_df[squad_df["pos"] == "DEF"].sort_values("xPts", ascending=False).head(3))
    best_xi.append(squad_df[squad_df["pos"] == "MID"].sort_values("xPts", ascending=False).head(4))
    best_xi.append(squad_df[squad_df["pos"] == "FWD"].sort_values("xPts", ascending=False).head(3))
    best_xi = pd.concat(best_xi).sort_values("xPts", ascending=False).head(11)

    st.markdown("### ✅ Best XI (sorted by xPts_total):")
    st.dataframe(format_for_display(best_xi, ["web_name", "pos", "team_name", "£m", "sel_by_%", "xPts_per_match", "xPts_total"]).reset_index(drop=True))

    captain = best_xi.iloc[0]["web_name"]
    vice_captain = best_xi.iloc[1]["web_name"]
    st.success(f"⭐ Recommended Captain: **{captain}** | Vice Captain: **{vice_captain}**")

    # Subs (bench)
    subs = squad_df[~squad_df["id"].isin(best_xi["id"])].sort_values("xPts", ascending=False)
    st.markdown("### 🪑 Subs (bench, sorted by xPts_total):")
    st.dataframe(format_for_display(subs, ["web_name", "pos", "team_name", "£m", "sel_by_%", "xPts_per_match", "xPts_total"]).reset_index(drop=True))

    # --- Transfer Suggestions (top 2 by gain in xPts_total) ---
    st.markdown("---")
    st.subheader("🔁 Suggested Transfers")

    current_xi_pts = best_xi["xPts"].sum()
    transfer_candidates = []

    for out_id in squad_ids:
        out_player = pred[pred["id"] == out_id].iloc[0]
        budget_available = bank * 10 + out_player["now_cost"]  # note now_cost in tenths

        candidates = pred[
            (pred["pos"] == out_player["pos"]) &
            (~pred["id"].isin(squad_ids)) &
            (pred["now_cost"] <= budget_available)
        ]

        if candidates.empty:
            continue

        in_player = candidates.sort_values("xPts", ascending=False).iloc[0]

        # simulate new best XI using xPts_total
        new_squad_ids = [pid for pid in squad_ids if pid != out_id] + [in_player["id"]]
        new_squad_df = pred[pred["id"].isin(new_squad_ids)]

        new_xi = []
        new_xi.append(new_squad_df[new_squad_df["pos"] == "GKP"].sort_values("xPts", ascending=False).head(1))
        new_xi.append(new_squad_df[new_squad_df["pos"] == "DEF"].sort_values("xPts", ascending=False).head(3))
        new_xi.append(new_squad_df[new_squad_df["pos"] == "MID"].sort_values("xPts", ascending=False).head(4))
        new_xi.append(new_squad_df[new_squad_df["pos"] == "FWD"].sort_values("xPts", ascending=False).head(3))
        new_xi = pd.concat(new_xi).head(11)

        new_pts_total = new_xi["xPts"].sum()
        gain = new_pts_total - current_xi_pts

        if gain > 0:
            transfer_candidates.append((gain, out_player, in_player, new_pts_total))

    transfer_candidates = sorted(transfer_candidates, key=lambda x: x[0], reverse=True)

    if transfer_candidates:
        st.markdown("#### 💡 Top 2 Transfer Suggestions:")
        for gain, out_p, in_p, new_pts in transfer_candidates[:2]:
            st.success(
                f"**{out_p['web_name']} ➝ {in_p['web_name']}** "
                f"(+{gain:.2f} xPts_total, new XI total = {new_pts:.2f})"
            )

        # allow user to choose an OUT from those candidate outs
        outs_list = [t[1]["web_name"] for t in transfer_candidates]
        out_choice = st.selectbox("Select player to sell (from suggested outs):", [""] + outs_list)
        if out_choice:
            chosen = [t for t in transfer_candidates if t[1]["web_name"] == out_choice][0]
            gain, out_p, in_p, new_pts = chosen
            st.info(
                f"Best replacement for **{out_p['web_name']}** ➝ **{in_p['web_name']}** "
                f"(+{gain:.2f} xPts_total, new XI total = {new_pts:.2f})"
            )
    else:
        st.info("No beneficial transfers found within your squad & budget.")
else:
    st.info("Please select exactly 15 players to analyze transfers.")
