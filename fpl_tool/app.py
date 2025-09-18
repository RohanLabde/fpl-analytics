# app.py
import streamlit as st
import pandas as pd
import requests
from typing import List, Tuple

# Import model (assumed present in fpl_tool/model.py)
from fpl_tool.model import build_player_master, v2_expected_points, add_value_columns

# -------------------------
# Data loading (cached)
# -------------------------
@st.cache_data(ttl=3600)
def load_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url, timeout=10)
    data = r.json()

    players = pd.DataFrame(data.get("elements", []))
    teams = pd.DataFrame(data.get("teams", []))
    element_types = pd.DataFrame(data.get("element_types", []))

    # ensure selected_by_percent exists and numeric
    if "selected_by_percent" in players.columns:
        players["selected_by_percent"] = pd.to_numeric(players["selected_by_percent"], errors="coerce").fillna(0.0)
    else:
        players["selected_by_percent"] = 0.0

    return players, teams, element_types


@st.cache_data(ttl=3600)
def load_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    r = requests.get(url, timeout=10)
    return pd.DataFrame(r.json())


# -------------------------
# Display helpers
# -------------------------
def format_for_display(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return a copy of df with friendly columns for display (price as £m, selected_by percent string)."""
    out = df.copy()
    if "now_cost" in out.columns:
        # create human readable price column (millions)
        out["£m"] = out["now_cost"].astype(float) / 10.0
    if "selected_by_percent" in out.columns:
        out["sel_by_%"] = out["selected_by_percent"].astype(float).map(lambda x: f"{x:.1f}%")
    # keep requested columns only if they exist
    existing = [c for c in cols if c in out.columns]
    return out[existing]


# -------------------------
# Best XI formation solver
# -------------------------
def build_best_xi_from_squad(squad_df: pd.DataFrame, metric_col: str = "xPts_per_match") -> pd.DataFrame:
    """
    Attempt several common formations and select the XI that maximizes the sum of metric_col.
    - squad_df: dataframe of the 15-man squad (must include pos, id and the metric_col)
    - metric_col: column used to rank players inside each formation
    Returns: DataFrame of selected XI (11 rows)
    """
    # ensure columns exist
    if metric_col not in squad_df.columns:
        # fallback to xPts or xPts_per_match or last numeric column
        fallback = None
        for c in ["xPts", "xPts_per_match", "xPts_per_m", "xPts_per_match"]:
            if c in squad_df.columns:
                fallback = c
                break
        metric_col = fallback if fallback else squad_df.columns[-1]

    # standard formation templates (GK, DEF, MID, FWD)
    formations = [
        (1, 3, 4, 3),
        (1, 4, 4, 2),
        (1, 4, 3, 2),
        (1, 4, 2, 3),
        (1, 5, 3, 2),
        (1, 3, 5, 1),
    ]

    best_sum = -1e9
    best_xi = None

    for form in formations:
        gk_n, def_n, mid_n, fwd_n = form
        parts = []
        try:
            parts.append(squad_df[squad_df["pos"] == "GKP"].sort_values(metric_col, ascending=False).head(gk_n))
            parts.append(squad_df[squad_df["pos"] == "DEF"].sort_values(metric_col, ascending=False).head(def_n))
            parts.append(squad_df[squad_df["pos"] == "MID"].sort_values(metric_col, ascending=False).head(mid_n))
            parts.append(squad_df[squad_df["pos"] == "FWD"].sort_values(metric_col, ascending=False).head(fwd_n))
            xi = pd.concat(parts)
            if len(xi) == 11:
                s = xi[metric_col].sum()
                if s > best_sum:
                    best_sum = s
                    best_xi = xi.copy()
        except Exception:
            continue

    # If nothing chosen (lack of players), choose top 11 by metric as fallback
    if best_xi is None or best_xi.empty:
        best_xi = squad_df.sort_values(metric_col, ascending=False).head(11)

    return best_xi.sort_values(metric_col, ascending=False)


# -------------------------
# Transfer suggestions
# -------------------------
def compute_transfer_suggestions(pred_df: pd.DataFrame,
                                 squad_ids: List[int],
                                 bank: float,
                                 metric_col: str = "xPts_per_match",
                                 min_minutes: int = 0) -> Tuple[List[Tuple[float, pd.Series, pd.Series, float]], float]:
    """
    Compute single-player swap suggestions.
    - pred_df: full players DataFrame (must contain id, pos, now_cost, minutes, etc.)
    - squad_ids: list of ints in the user's 15-man squad
    - bank: float (in £m)
    - metric_col: column used to rank players (e.g. 'xPts_per_match' or 'xPts_total')
    - min_minutes: minimum historical minutes required for an incoming candidate
    Returns:
      (sorted_candidates, current_total) where sorted_candidates is list of tuples
      (gain, out_player_series, in_player_series, new_total)
    """
    df = pred_df.copy()
    # ensure numeric minutes
    if "minutes" not in df.columns:
        df["minutes"] = 0.0
    else:
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)

    # ensure now_cost numeric
    if "now_cost" in df.columns:
        df["now_cost"] = pd.to_numeric(df["now_cost"], errors="coerce").fillna(0.0)
    else:
        df["now_cost"] = 0.0

    bank_tenths = float(bank) * 10.0
    current_squad = df[df["id"].isin(squad_ids)]

    # compute current XI and its metric total
    current_xi = build_best_xi_from_squad(current_squad, metric_col)
    current_total = float(current_xi[metric_col].sum())

    candidates = []
    for out_id in squad_ids:
        out_player = df[df["id"] == out_id].iloc[0]
        budget_available = bank_tenths + float(out_player.get("now_cost", 0))

        # candidate pool: same position, not in squad, within budget, above minutes threshold
        pool = df[
            (df["pos"] == out_player["pos"]) &
            (~df["id"].isin(squad_ids)) &
            (df["now_cost"].astype(float) <= budget_available) &
            (df["minutes"].astype(float) >= min_minutes)
        ]

        if pool.empty:
            continue

        # pick top candidate by metric_col
        pick_col = metric_col if metric_col in pool.columns else "xPts"
        if pick_col not in pool.columns:
            in_player = pool.iloc[0]
        else:
            in_player = pool.sort_values(pick_col, ascending=False).iloc[0]

        # simulate new squad and recompute best XI
        new_ids = [pid for pid in squad_ids if pid != out_id] + [int(in_player["id"])]
        new_squad = df[df["id"].isin(new_ids)]
        new_xi = build_best_xi_from_squad(new_squad, metric_col)
        new_total = float(new_xi[metric_col].sum())

        gain = new_total - current_total
        candidates.append((gain, out_player, in_player, new_total))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    return candidates, current_total


# -------------------------
# UI & App layout
# -------------------------
st.set_page_config(page_title="FPL Analytics – Smarter Expected Points", layout="wide")
st.title("⚽ FPL Analytics – Smarter Expected Points")
st.caption("Data: Official Fantasy Premier League API. Model uses V2 logic (xG/xA + cs prob + minutes + fixture horizon).")

# load API data
players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

# build player master via model.py helper
pm = build_player_master(players, teams, element_types)

# -------------------------
# Sidebar settings
# -------------------------
st.sidebar.header("Model & display settings")
horizon = st.sidebar.slider("Fixture horizon (matches)", min_value=1, max_value=10, value=5)
rank_by = st.sidebar.selectbox("Rank by (per match or total)", options=["xPts_per_match", "xPts_total"])
min_minutes = st.sidebar.slider("Min historical minutes for leaderboards (0 = no filter)", min_value=0, max_value=1500, value=0)
top_n = st.sidebar.number_input("Top N per position", min_value=1, max_value=20, value=10)

# Compute predictions (V2)
pred = v2_expected_points(pm, fixtures, teams, horizon=horizon)
pred = add_value_columns(pred)

# Add safety defaults for commonly used display columns
for col in ["now_cost", "selected_by_percent", "minutes"]:
    if col not in pred.columns:
        pred[col] = 0.0

# compute per-match and total columns if model returned only one form
if "xPts" in pred.columns and "games_proj" in pred.columns:
    pred["xPts_per_match"] = pred["xPts"] / pred["games_proj"].replace({0: 1})
    pred["xPts_total"] = pred["xPts"]  # xPts is already a total projection over horizon in many implementations
else:
    # try fallback names
    if "xPts_per_match" not in pred.columns:
        pred["xPts_per_match"] = pred.get("xPts", 0.0)
    if "xPts_total" not in pred.columns:
        pred["xPts_total"] = pred.get("xPts", 0.0)

# Apply min_minutes filter for leaderboards (but keep full pred for transfers - transfer will also use min_minutes)
leaderboard_df = pred[pred["minutes"].astype(float) >= float(min_minutes)] if min_minutes > 0 else pred.copy()

# -------------------------
# Leaderboards - per position
# -------------------------
def show_top_by_position(df: pd.DataFrame, metric: str, top_n_per_pos: int = 10, gk_n: int = 3):
    pos_map = {"GKP": gk_n, "DEF": top_n_per_pos, "MID": top_n_per_pos, "FWD": top_n_per_pos}
    result = {}
    for pos, n in pos_map.items():
        subset = df[df["pos"] == pos].sort_values(metric, ascending=False).head(n)
        result[pos] = subset
    return result

st.subheader(f"🎯 Captaincy picks (Top by {rank_by} per position)")
cap_tables = show_top_by_position(leaderboard_df, rank_by, top_n_per_pos=top_n, gk_n=3)

for pos, table in cap_tables.items():
    st.markdown(f"**Top {len(table)} {pos}s by {rank_by}**")
    # choose display columns smartly
    if pos == "GKP":
        cols = ["web_name", "team_name", "pos", "now_cost", "selected_by_percent", "cs_prob", "xSaves_per_match" if "xSaves_per_match" in table.columns else "xSaves", rank_by]
    elif pos == "DEF":
        cols = ["web_name", "team_name", "pos", "now_cost", "selected_by_percent", "xAttack_per90" if "xAttack_per90" in table.columns else "xAttack", "att_factor" if "att_factor" in table.columns else "att_factor", "cs_prob", rank_by]
    else:  # MID/FWD
        cols = ["web_name", "team_name", "pos", "now_cost", "selected_by_percent", "xAttack_per90" if "xAttack_per90" in table.columns else "xAttack", "att_factor" if "att_factor" in table.columns else "att_factor", rank_by]

    st.dataframe(format_for_display(table, cols).reset_index(drop=True), use_container_width=True)

# Value picks (by xPts_per_m)
st.subheader("💼 Value picks (Top by xPts_per_m per position)")
val_tables = show_top_by_position(leaderboard_df, "xPts_per_m", top_n_per_pos=top_n, gk_n=3)
for pos, table in val_tables.items():
    st.markdown(f"**Top {len(table)} {pos}s by xPts_per_m**")
    cols = ["web_name", "team_name", "pos", "now_cost", "selected_by_percent", "xAttack_per90" if "xAttack_per90" in table.columns else "xAttack", "cs_prob", "xPts_per_m"]
    st.dataframe(format_for_display(table, cols).reset_index(drop=True), use_container_width=True)

# -------------------------
# Analyze My Squad
# -------------------------
st.subheader("🧩 Analyze My 15-man Squad")

player_options = {
    int(r.id): f"{r.web_name} ({r.team_name}, {r.pos}, £{(r.now_cost/10) if 'now_cost' in r._fields else 0:.1f}m, {getattr(r, 'selected_by_percent', 0):.1f}%)"
    for r in pred.itertuples()
}

squad_ids = st.multiselect(
    "Select your 15 players",
    options=list(player_options.keys()),
    format_func=lambda x: player_options[x],
    help="Pick exactly 15 players from your squad"
)

bank = st.number_input("Bank (money in the bank, £m)", min_value=0.0, step=0.1)

if len(squad_ids) == 15:
    squad_df = pred[pred["id"].isin(squad_ids)].copy()

    # compute best XI (formation-aware) using chosen rank metric
    best_xi = build_best_xi_from_squad(squad_df, rank_by)
    st.markdown(f"### ✅ Best XI (sorted by {rank_by}):")
    st.dataframe(format_for_display(best_xi, ["web_name", "pos", "team_name", "now_cost", "selected_by_percent", rank_by, "xPts_total"]).reset_index(drop=True), use_container_width=True)

    captain = best_xi.iloc[0]["web_name"]
    vice_captain = best_xi.iloc[1]["web_name"]
    st.success(f"⭐ Recommended Captain: **{captain}** | Vice Captain: **{vice_captain}**")

    # Subs (bench)
    subs = squad_df[~squad_df["id"].isin(best_xi["id"])].sort_values(rank_by, ascending=False)
    st.markdown("### 🪑 Subs (bench, sorted by metric):")
    st.dataframe(format_for_display(subs, ["web_name", "pos", "team_name", "now_cost", "selected_by_percent", rank_by, "xPts_total"]).reset_index(drop=True), use_container_width=True)

    # Transfer suggestions: top 2 swaps that obey min_minutes filter
    st.markdown("---")
    st.subheader("🔁 Suggested Transfers (single-player swaps)")

    transfer_candidates, current_total = compute_transfer_suggestions(pred, squad_ids, bank, metric_col=rank_by, min_minutes=min_minutes)

    if transfer_candidates:
        st.markdown("#### 💡 Top 2 Transfer Suggestions:")
        for gain, out_p, in_p, new_total in transfer_candidates[:2]:
            label = f"**{out_p['web_name']} ➝ {in_p['web_name']}** (+{gain:.2f} {rank_by}, new XI total = {new_total:.2f})"
            st.success(label)

        # allow user to pick an out player from suggestions and show best replacement
        st.markdown("#### 🎯 Choose a player to transfer OUT:")
        out_choice = st.selectbox("Select player to sell", [f"{t[1]['web_name']} ({t[1]['pos']}, {t[1]['team_name']})" for t in transfer_candidates])
        if out_choice:
            chosen = [t for t in transfer_candidates if f"{t[1]['web_name']} ({t[1]['pos']}, {t[1]['team_name']})" == out_choice][0]
            gain, out_p, in_p, new_total = chosen
            st.info(f"Best replacement for **{out_p['web_name']}** ➝ **{in_p['web_name']}** (+{gain:.2f} {rank_by}, new XI total = {new_total:.2f})")
    else:
        st.info("No beneficial transfers found within your squad & budget (respecting min minutes).")

else:
    st.info("Please select exactly 15 players to analyze transfers.")

# -------------------------
# End
# -------------------------
st.markdown("---")
st.caption("Notes: 'min historical minutes' prevents tiny-sample players being considered. The solver tries several formations to select the best XI by the chosen metric.")
