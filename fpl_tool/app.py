# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests

from fpl_tool.model import build_player_master, v2_expected_points, add_value_columns

# ---------------------------
# Data loading (cached)
# ---------------------------
@st.cache_data(ttl=3600)
def load_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    data = r.json()

    players = pd.DataFrame(data.get("elements", []))
    teams = pd.DataFrame(data.get("teams", []))
    element_types = pd.DataFrame(data.get("element_types", []))

    return players, teams, element_types

@st.cache_data(ttl=3600)
def load_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    r = requests.get(url)
    return pd.DataFrame(r.json())

# ---------------------------
# Formatting / helpers
# ---------------------------
def fmt_df_for_display(df: pd.DataFrame, cols):
    """
    Format DataFrame for display and ensure unique columns.
    - converts money to £m (column 'price_m' if available or converts now_cost)
    - formats selected_by_percent to sel_by_% if available
    - rounds numeric columns
    - removes duplicate columns
    - preserves requested column order (and dedupes requested cols)
    """
    d = df.copy()

    # create price_m column cleanly if not present
    if "price_m" not in d.columns:
        if "now_cost" in d.columns:
            d["price_m"] = d["now_cost"] / 10.0
        elif "now_cost/10" in d.columns:
            # Just in case some computations produced this name
            d["price_m"] = d["now_cost/10"]
    # selection percentage
    if "sel_by_%" not in d.columns:
        if "selected_by_percent" in d.columns:
            d["sel_by_%"] = d["selected_by_percent"].astype(float).map(lambda x: f"{x:.1f}%")
        elif "sel_by" in d.columns:
            d["sel_by_%"] = d["sel_by"].astype(float).map(lambda x: f"{x:.1f}%")

    # Round / normalize numeric columns commonly used in UI
    numeric_cols = [
        "price_m", "xAttack_per90", "xAttack", "att_factor", "cs_prob",
        "xSaves", "xSaves_per_match", "xPts_per_match", "xPts_total",
        "xPts", "xPts_per_m"
    ]
    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").round(3)

    # Remove duplicate columns that could exist in the DataFrame (keep first occurrence)
    d = d.loc[:, ~d.columns.duplicated()]

    # Ensure requested columns are unique and in order and exist in df
    final_cols = []
    for c in cols:
        if c in d.columns and c not in final_cols:
            final_cols.append(c)

    # If none of requested cols exist, return original df truncated to first 10 cols as fallback
    if not final_cols:
        final_cols = list(d.columns[:10])

    return d[final_cols]

# Best XI picker that tries a small set of formations and chooses best total metric
def build_best_xi_from_squad(squad_df: pd.DataFrame, metric_col: str):
    """
    Given squad_df (must contain pos and metric_col), try common formations and return
    the highest-scoring 11 according to metric_col.
    Enforces 1 GK always.
    """
    # Ensure metric exists
    if metric_col not in squad_df.columns:
        metric_col = "xPts" if "xPts" in squad_df.columns else squad_df.columns[0]

    # Candidate formations as tuples (DEF, MID, FWD) - common choices
    formations = [
        (4, 4, 2),
        (4, 3, 3),
        (3, 5, 2),
        (3, 4, 3),
        (5, 3, 2)
    ]

    best_sum = -1e9
    best_lineup = None
    for d_count, m_count, f_count in formations:
        parts = []
        # GK
        gk = squad_df[squad_df["pos"] == "GKP"].sort_values(metric_col, ascending=False).head(1)
        if gk.shape[0] < 1:
            continue  # can't form lineup without GK
        parts.append(gk)

        # DEF
        defs = squad_df[squad_df["pos"] == "DEF"].sort_values(metric_col, ascending=False).head(d_count)
        if defs.shape[0] < d_count:
            continue
        parts.append(defs)

        # MID
        mids = squad_df[squad_df["pos"] == "MID"].sort_values(metric_col, ascending=False).head(m_count)
        if mids.shape[0] < m_count:
            continue
        parts.append(mids)

        # FWD
        fwds = squad_df[squad_df["pos"] == "FWD"].sort_values(metric_col, ascending=False).head(f_count)
        if fwds.shape[0] < f_count:
            continue
        parts.append(fwds)

        try:
            lineup = pd.concat(parts)
        except ValueError:
            continue

        if lineup.shape[0] != 11:
            continue

        total = lineup[metric_col].sum()
        if total > best_sum:
            best_sum = total
            best_lineup = lineup.copy()

    # As fallback: if no formation found, pick 11 highest by metric but keep 1 GK
    if best_lineup is None:
        gk = squad_df[squad_df["pos"] == "GKP"].sort_values(metric_col, ascending=False).head(1)
        others = squad_df[squad_df["pos"] != "GKP"].sort_values(metric_col, ascending=False).head(10)
        best_lineup = pd.concat([gk, others]).head(11)

    return best_lineup.sort_values(metric_col, ascending=False)

# Transfer suggestion helper: single-player swaps only
def compute_transfer_suggestions(pred_df: pd.DataFrame, squad_ids: list, bank: float, metric_col="xPts"):
    bank_tenths = bank * 10.0
    current_squad = pred_df[pred_df["id"].isin(squad_ids)]
    # compute current best XI by metric (simple heuristic: best_xi_from_squad)
    current_xi = build_best_xi_from_squad(current_squad, metric_col)
    current_total = float(current_xi[metric_col].sum())

    candidates = []
    for out_id in squad_ids:
        out_player = pred_df[pred_df["id"] == out_id].iloc[0]
        # available budget in tenths
        budget_available = bank_tenths + float(out_player.get("now_cost", 0))
        same_pos_candidates = pred_df[
            (pred_df["pos"] == out_player["pos"]) &
            (~pred_df["id"].isin(squad_ids)) &
            (pred_df["now_cost"] <= budget_available)
        ]
        if same_pos_candidates.empty:
            continue
        # choose best candidate by metric
        best_in = same_pos_candidates.sort_values(metric_col, ascending=False).iloc[0]
        # simulate new squad ids
        new_ids = [pid for pid in squad_ids if pid != out_id] + [int(best_in["id"])]
        new_squad = pred_df[pred_df["id"].isin(new_ids)]
        new_xi = build_best_xi_from_squad(new_squad, metric_col)
        new_total = float(new_xi[metric_col].sum())
        gain = new_total - current_total
        candidates.append((gain, out_player, best_in, new_total))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    return candidates, current_total

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="FPL Analytics – Smarter Expected Points", layout="wide")
st.title("⚽ FPL Analytics – Smarter Expected Points")
st.caption("Model: uses minutes/xG/xA + fixture horizon -> xPts_per_match and xPts_total. Filter by min minutes to avoid outliers.")

# Load data
players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

# Build player master
pm = build_player_master(players, teams, element_types)

# Sidebar settings
st.sidebar.header("Model & display settings")
horizon = st.sidebar.slider("Fixture horizon (matches)", 1, 10, 5)

rank_by = st.sidebar.selectbox("Rank by (per match or total)", options=["xPts_per_match", "xPts_total"])
min_minutes = st.sidebar.slider("Min historical minutes for leaderboards (0 = no filter)", 0, 2000, 270)
top_n = st.sidebar.number_input("Top N per position", min_value=3, max_value=20, value=10, step=1)

# Run model
pred = v2_expected_points(pm, fixtures, teams, horizon=horizon)  # returns df with xPts etc.
pred = add_value_columns(pred)  # adds xPts_per_m etc.

# Enrich / normalize column names we expect
# price_m (convert)
if "price_m" not in pred.columns:
    if "now_cost" in pred.columns:
        pred["price_m"] = pred["now_cost"] / 10.0
    else:
        pred["price_m"] = np.nan

# selection %
if "selected_by_percent" in pred.columns and "sel_by_%" not in pred.columns:
    pred["sel_by_%"] = pred["selected_by_percent"].astype(float).map(lambda x: f"{x:.1f}%")
elif "sel_by" in pred.columns and "sel_by_%" not in pred.columns:
    pred["sel_by_%"] = pred["sel_by"].astype(float).map(lambda x: f"{x:.1f}%")
else:
    if "sel_by_%" not in pred.columns:
        pred["sel_by_%"] = ""

# Provide both per-match and total columns if model provided only one
if "xPts_per_match" not in pred.columns:
    # if model provided xPts as total over horizon, make a per-match average
    if "xPts" in pred.columns and "games_proj" in pred.columns and pred["games_proj"].sum() > 0:
        pred["xPts_total"] = pd.to_numeric(pred["xPts"], errors="coerce")
        pred["xPts_per_match"] = pred["xPts_total"] / pred["games_proj"].replace({0: np.nan})
    else:
        pred["xPts_per_match"] = pd.to_numeric(pred.get("xPts", 0)) / np.maximum(pred.get("games_proj", 1), 1)

if "xPts_total" not in pred.columns:
    # assume xPts_total = xPts_per_match * games_proj if xPts_per_match exists
    if "xPts_per_match" in pred.columns and "games_proj" in pred.columns:
        pred["xPts_total"] = pred["xPts_per_match"] * pred["games_proj"]
    else:
        pred["xPts_total"] = pred.get("xPts", 0)

# Provide convenient columns used by UI
pred["id"] = pred["id"].astype(int)
pred["pos"] = pred["pos"].astype(str)
pred["team_name"] = pred.get("team_name", pred.get("team", "")).astype(str)

# Apply min minutes filter for leaderboards where requested
leaderboard_df = pred.copy()
if min_minutes > 0 and "minutes" in leaderboard_df.columns:
    leaderboard_df = leaderboard_df[leaderboard_df["minutes"].astype(float) >= min_minutes]

# --- Top by position function ---
def top_by_position(df, metric_col, top_n=10, gk_n=3):
    pos_map = {"GKP": gk_n, "DEF": top_n, "MID": top_n, "FWD": top_n}
    out = {}
    for pos, n in pos_map.items():
        sub = df[df["pos"] == pos].sort_values(metric_col, ascending=False).head(n)
        out[pos] = sub
    return out

# --- Captaincy & Value picks display ---
st.subheader(f"🎯 Captaincy picks (Top by {rank_by} per position)")

captaincy = top_by_position(leaderboard_df, rank_by, top_n, gk_n=3)
for pos, tbl in captaincy.items():
    st.markdown(f"**Top {min(len(tbl), top_n) if pos != 'GKP' else min(len(tbl), 3)} {pos}s by {rank_by}**")
    # choose columns appropriate for positions
    if pos == "GKP":
        cols = ["web_name", "team_name", "pos", "price_m", "sel_by_%", "cs_prob", "xSaves", "xPts_per_match", "xPts_total"]
    elif pos == "DEF":
        cols = ["web_name", "team_name", "pos", "price_m", "sel_by_%", "xAttack_per90", "att_factor", "cs_prob", "xPts_per_match", "xPts_total"]
    else:
        cols = ["web_name", "team_name", "pos", "price_m", "sel_by_%", "xAttack_per90", "att_factor", "xPts_per_match", "xPts_total"]
    st.dataframe(fmt_df_for_display(tbl, cols).reset_index(drop=True), use_container_width=True)

# Value picks (xPts_per_m)
st.subheader("💼 Value picks (Top by xPts per million per position)")
value_tables = top_by_position(leaderboard_df, "xPts_per_m", top_n, gk_n=3)
for pos, tbl in value_tables.items():
    st.markdown(f"**Top {min(len(tbl), top_n)} {pos}s by xPts_per_m**")
    if pos == "GKP":
        cols = ["web_name", "team_name", "pos", "price_m", "sel_by_%", "cs_prob", "xSaves", "xPts_per_m", "xPts_total"]
    elif pos == "DEF":
        cols = ["web_name", "team_name", "pos", "price_m", "sel_by_%", "xAttack_per90", "att_factor", "cs_prob", "xPts_per_m", "xPts_total"]
    else:
        cols = ["web_name", "team_name", "pos", "price_m", "sel_by_%", "xAttack_per90", "att_factor", "xPts_per_m", "xPts_total"]
    st.dataframe(fmt_df_for_display(tbl, cols).reset_index(drop=True), use_container_width=True)

# --- Analyze My 15-man Squad ---
st.subheader("🧩 Analyze My 15-man Squad")
player_options = {
    int(r.id): f"{r.web_name} ({r.team_name}, {r.pos}, £{(r.price_m if 'price_m' in r._fields else (r.now_cost/10 if 'now_cost' in r._fields else 0)):.1f}m, {getattr(r, 'selected_by_percent', getattr(r, 'sel_by', 0))}%)"
    for r in pred.itertuples()
}

squad_ids = st.multiselect(
    "Select your 15 players",
    options=list(player_options.keys()),
    format_func=lambda x: player_options[x]
)

bank = st.number_input("Bank (money in the bank, £m)", min_value=0.0, step=0.1, value=0.0)

if len(squad_ids) == 15:
    squad_df = pred[pred["id"].isin(squad_ids)].copy()

    # Pick best XI using formations search and rank_by (per match or total)
    best_xi = build_best_xi_from_squad(squad_df, rank_by)

    st.markdown(f"### ✅ Best XI (sorted by {rank_by}):")
    st.dataframe(fmt_df_for_display(best_xi, ["web_name", "pos", "team_name", "price_m", "sel_by_%", rank_by, "xPts_total"]).reset_index(drop=True), use_container_width=True)

    # captain / vice
    captain = best_xi.iloc[0]["web_name"]
    vice_captain = best_xi.iloc[1]["web_name"]
    st.success(f"⭐ Recommended Captain: **{captain}** | Vice Captain: **{vice_captain}**")

    # Subs: those in squad not in best XI (sorted by rank metric)
    subs = squad_df[~squad_df["id"].isin(best_xi["id"])].sort_values(rank_by, ascending=False)
    st.markdown("### 🪑 Subs (bench, sorted by chosen metric):")
    st.dataframe(fmt_df_for_display(subs, ["web_name", "pos", "team_name", "price_m", "sel_by_%", rank_by, "xPts_total"]).reset_index(drop=True), use_container_width=True)

    # --- Transfer suggestions (top 2) ---
    st.markdown("---")
    st.subheader("🔁 Suggested Transfers (single-player swaps)")

    transfer_candidates, current_total = compute_transfer_suggestions(pred, squad_ids, bank, rank_by)

    if transfer_candidates:
        st.markdown("#### 💡 Top 2 Transfer Suggestions:")
        for gain, out_p, in_p, new_pts in transfer_candidates[:2]:
            st.success(
                f"**{out_p['web_name']} ➝ {in_p['web_name']}** "
                f"(+{gain:.2f} {rank_by}, new XI total = {new_pts:.2f})"
            )

        st.markdown("#### 🎯 Choose a player to transfer OUT:")
        # build readable list for selectbox (unique names)
        out_list = [f"{t[1]['web_name']} ({t[1]['pos']}, {t[1]['team_name']})" for t in transfer_candidates]
        choice = st.selectbox("Select player to sell", options=out_list)
        if choice:
            idx = out_list.index(choice)
            gain, out_p, in_p, new_pts = transfer_candidates[idx]
            st.info(
                f"Best replacement for **{out_p['web_name']}** ➝ **{in_p['web_name']}** "
                f"(+{gain:.2f} {rank_by}, new XI total = {new_pts:.2f})"
            )
    else:
        st.info("No beneficial transfers found within your squad & budget.")

else:
    st.info("Please select exactly 15 players to analyze transfers.")
