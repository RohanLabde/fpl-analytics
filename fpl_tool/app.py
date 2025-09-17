# app.py
import streamlit as st
import pandas as pd
import requests

from fpl_tool.model import build_player_master, v2_expected_points

# --- Data loaders ---
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


# --- Small display helper ---
def format_for_display(df: pd.DataFrame, cols):
    out = df.copy()
    # convert now_cost (tenths) to £m
    if "now_cost" in out.columns and "£m" not in out.columns:
        out["£m"] = out["now_cost"] / 10.0
    # selected_by_percent -> nice column
    if "selected_by_percent" in out.columns and "sel_by_%" not in out.columns:
        # sometimes it's string — coerce
        out["sel_by_%"] = pd.to_numeric(out["selected_by_percent"], errors="coerce").fillna(0.0).map(lambda x: f"{x:.1f}%")
    # xPts formatting (optional): keep raw numeric but round for display
    for c in ["xPts_per_match", "xPts_total", "xPts_per_m", "xPts_per_m_match"]:
        if c in out.columns:
            out[c] = out[c].astype(float).round(3)
    return out[cols]


# --- Streamlit UI setup ---
st.set_page_config(page_title="FPL Analytics – Smarter Expected Points", layout="wide")
st.title("⚽ FPL Analytics – Smarter Expected Points")
st.caption("Model uses shrinkage on per-90 metrics, fixture horizon, and returns per-match & total xPts.")

# Load API data
players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

# Sidebar controls
st.sidebar.header("Model settings")
horizon = st.sidebar.slider("Fixture horizon (matches)", 1, 10, value=5)
prior_minutes = st.sidebar.slider("Shrinkage prior (minutes, higher = more shrink)", 0, 2000, value=270)
min_minutes_for_lists = st.sidebar.slider("Min minutes to show in leaderboards (0 = show all)", 0, 900, value=90)
hide_low_minutes = st.sidebar.checkbox("Hide low-minute players from leaderboards", value=True)

st.sidebar.markdown("---")
rank_metric = st.sidebar.radio(
    "Ranking metric for captain picks",
    ("xPts_per_match", "xPts_total"),
    help="Per-match = expected for next game (accounts for appearance probability); Total = horizon total."
)

# top_n settings
top_n = st.sidebar.number_input("Top N per position", min_value=1, max_value=30, value=10, step=1)

# Build player master and model predictions
pm = build_player_master(players, teams, element_types)
pred = v2_expected_points(pm, fixtures, teams, horizon=horizon, prior_minutes=float(prior_minutes))

# Some compatibility renames if needed
# (model may provide both xPts_per_m and xPts_per_m_match -> prefer per-match for value picks when ranking per-match)
if "xPts_per_m_match" not in pred.columns and "xPts_per_m" in pred.columns:
    pred["xPts_per_m_match"] = pred["xPts_per_m"]

# Optionally filter leaderboard dataset
if hide_low_minutes and min_minutes_for_lists > 0:
    leaderboard_df = pred[pred["minutes"].fillna(0) >= min_minutes_for_lists].copy()
else:
    leaderboard_df = pred.copy()

# Helper: show top per position dictionary
def top_by_position(df: pd.DataFrame, score_col: str, top_n: int = 10, gk_n: int = 3):
    pos_map = {"GKP": gk_n, "DEF": top_n, "MID": top_n, "FWD": top_n}
    result = {}
    for pos, n in pos_map.items():
        subset = df[df["pos"] == pos].sort_values(score_col, ascending=False).head(n)
        result[pos] = subset
    return result

# --- Captaincy picks ---
st.subheader("🎯 Captaincy picks (Top by {} per position)".format("per match" if rank_metric=="xPts_per_match" else "total"))

capt_tables = top_by_position(leaderboard_df, rank_metric, top_n=top_n, gk_n=3)

for pos, table in capt_tables.items():
    st.markdown(f"**Top {len(table)} {pos}s by {'per match' if rank_metric=='xPts_per_match' else 'total'}**")
    if pos == "GKP":
        cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "cs_prob", "xSaves_per_match", rank_metric]
    elif pos == "DEF":
        cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", "att_factor", "cs_prob", rank_metric]
    elif pos in ("MID", "FWD"):
        cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", "att_factor", rank_metric]
    else:
        cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", rank_metric]

    # keep only columns that exist
    cols = [c for c in cols if c in table.columns]
    st.dataframe(format_for_display(table, cols).reset_index(drop=True))


# --- Value picks ---
st.subheader("💼 Value picks (Top by xPts per million per position)")

# Choose value ranking depending on chosen metric:
if rank_metric == "xPts_per_match":
    value_score = "xPts_per_m_match"  # per-match value
else:
    value_score = "xPts_per_m"  # total value

value_tables = top_by_position(leaderboard_df, value_score, top_n=top_n, gk_n=3)

for pos, table in value_tables.items():
    st.markdown(f"**Top {len(table)} {pos}s by value ({value_score})**")
    if pos == "GKP":
        cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "cs_prob", "xSaves_per_match", value_score]
    elif pos == "DEF":
        cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", "cs_prob", value_score]
    elif pos in ("MID", "FWD"):
        cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", value_score]
    else:
        cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", value_score]

    cols = [c for c in cols if c in table.columns]
    st.dataframe(format_for_display(table, cols).reset_index(drop=True))


# --- Analyze My Squad ---
st.header("🧩 Analyze My 15-man Squad")

player_options = {
    int(r.id): f"{r.web_name} ({r.team_name}, {r.pos}, £{(r.now_cost/10):.1f}m, {float(r.selected_by_percent) if 'selected_by_percent' in r._fields else 0:.1f}%)"
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

    # build best XI (heuristic)
    best_xi_parts = []
    best_xi_parts.append(squad_df[squad_df["pos"] == "GKP"].sort_values("xPts_per_match", ascending=False).head(1))
    best_xi_parts.append(squad_df[squad_df["pos"] == "DEF"].sort_values("xPts_per_match", ascending=False).head(3))
    best_xi_parts.append(squad_df[squad_df["pos"] == "MID"].sort_values("xPts_per_match", ascending=False).head(4))
    best_xi_parts.append(squad_df[squad_df["pos"] == "FWD"].sort_values("xPts_per_match", ascending=False).head(3))

    best_xi = pd.concat(best_xi_parts).sort_values("xPts_per_match", ascending=False).head(11)

    st.markdown("### ✅ Best XI (sorted by xPts_per_match):")
    st.dataframe(format_for_display(best_xi, ["web_name", "pos", "team_name", "£m", "sel_by_%", "xPts_per_match", "xPts_total"]).reset_index(drop=True))

    captain = best_xi.iloc[0]["web_name"]
    vice_captain = best_xi.iloc[1]["web_name"]
    st.success(f"⭐ Recommended Captain: **{captain}** | Vice Captain: **{vice_captain}**")

    # Subs
    subs = squad_df[~squad_df["id"].isin(best_xi["id"])].sort_values("xPts_per_match", ascending=False)
    st.markdown("### 🪑 Subs (bench, sorted by xPts_per_match):")
    cols_subs = [c for c in ["web_name", "pos", "team_name", "£m", "sel_by_%", "xPts_per_match", "xPts_total"] if c in subs.columns]
    st.dataframe(format_for_display(subs, cols_subs).reset_index(drop=True))

    # Transfer suggestions (top 2 improvements)
    st.markdown("---")
    st.subheader("🔁 Suggested Transfers")

    current_xi_pts = best_xi["xPts_total"].sum() if "xPts_total" in best_xi.columns else best_xi["xPts_per_match"].sum()

    transfer_candidates = []
    for out_id in squad_ids:
        out_row = pred[pred["id"] == out_id].iloc[0]
        budget_available_tenths = bank * 10 + out_row["now_cost"]

        # candidate replacements same pos, not currently in squad, within budget
        candidates = pred[
            (pred["pos"] == out_row["pos"]) &
            (~pred["id"].isin(squad_ids)) &
            (pred["now_cost"] <= budget_available_tenths)
        ]
        if candidates.empty:
            continue

        # pick best by xPts_total (or by xPts_per_match if you prefer)
        in_row = candidates.sort_values("xPts_total", ascending=False).iloc[0]

        # simulate
        new_ids = [pid for pid in squad_ids if pid != out_id] + [int(in_row["id"])]
        new_df = pred[pred["id"].isin(new_ids)].copy()

        new_xi_parts = []
        new_xi_parts.append(new_df[new_df["pos"] == "GKP"].sort_values("xPts_total", ascending=False).head(1))
        new_xi_parts.append(new_df[new_df["pos"] == "DEF"].sort_values("xPts_total", ascending=False).head(3))
        new_xi_parts.append(new_df[new_df["pos"] == "MID"].sort_values("xPts_total", ascending=False).head(4))
        new_xi_parts.append(new_df[new_df["pos"] == "FWD"].sort_values("xPts_total", ascending=False).head(3))
        new_xi = pd.concat(new_xi_parts).head(11)

        new_pts = new_xi["xPts_total"].sum()
        gain = new_pts - current_xi_pts

        if gain > 0:
            transfer_candidates.append((gain, out_row, in_row, new_pts))

    transfer_candidates = sorted(transfer_candidates, key=lambda x: x[0], reverse=True)
    if transfer_candidates:
        st.markdown("#### 💡 Top 2 transfer suggestions (by xPts_total improvement):")
        for gain, out_r, in_r, new_pts in transfer_candidates[:2]:
            st.success(f"**{out_r['web_name']} ➝ {in_r['web_name']}** (+{gain:.2f} xPts_total, new XI total = {new_pts:.2f})")

        st.markdown("#### 🎯 Suggest replacement for a chosen OUT player")
        out_names = [t[1]["web_name"] for t in transfer_candidates]
        choice = st.selectbox("Choose an OUT player from suggestions", options=out_names)
        if choice:
            chosen = [t for t in transfer_candidates if t[1]["web_name"] == choice][0]
            gain, out_r, in_r, new_pts = chosen
            st.info(f"Best replacement for **{out_r['web_name']}** ➝ **{in_r['web_name']}** (+{gain:.2f} xPts_total)")
    else:
        st.info("No beneficial transfers found within your squad & budget.")
else:
    st.info("Please select exactly 15 players to analyze transfers.")

# --- Footer note ---
st.markdown("---")
st.caption("Notes: Model uses shrinkage to stabilize per-90 metrics for low-minute players. Use the 'Min minutes' slider to hide players with tiny samples from leaderboards.")
