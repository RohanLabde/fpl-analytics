import streamlit as st
import pandas as pd
import requests

from fpl_tool.model import build_player_master, v2_expected_points


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

    # Round numeric columns used for display
    round_cols = [
        "xPts_per_match", "xPts_total", "xPts_per_m", "xAttack_per90", "xSaves_per_match",
        "cs_prob", "att_factor", "xPts_per_m_match"
    ]
    for rc in round_cols:
        if rc in out.columns:
            out[rc] = pd.to_numeric(out[rc], errors="coerce").round(3)

    # Ensure sel_by_% and £m look nice (they should exist on pred already)
    if "sel_by_%" in out.columns:
        # Already prettified when created on pred; keep as-is
        pass
    if "£m" in out.columns:
        out["£m"] = pd.to_numeric(out["£m"], errors="coerce").round(1)

    # final safe col selection
    final_cols = [c for c in cols if c in out.columns]
    return out[final_cols]


# --- Streamlit UI ---
st.set_page_config(page_title="FPL Analytics – Expected Points Model", layout="wide")
st.title("⚽ FPL Analytics – Smarter Expected Points")
st.caption("Data: Official Fantasy Premier League API. Model uses xG, xA, clean sheet probability, saves & fixture horizon adjustments.")

# Load data
players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

pm = build_player_master(players, teams, element_types)

# Sidebar settings
st.sidebar.header("Model Settings")
horizon = st.sidebar.slider("Fixture horizon (matches)", 1, 10, 5)
prior_minutes = st.sidebar.slider("Shrinkage prior (minutes)", 0, 2000, 270)
min_minutes_for_lists = st.sidebar.slider("Min minutes to show in leaderboards (0 = show all)", 0, 900, 90)
hide_low_minutes = st.sidebar.checkbox("Hide low-minute players from leaderboards", value=True)

st.sidebar.markdown("---")
rank_metric = st.sidebar.radio(
    "Ranking metric for captain picks",
    ("xPts_per_match", "xPts_total"),
    help="Per-match = expected for next game; Total = horizon total across matches."
)

top_n = st.sidebar.number_input("Top N per position", min_value=1, max_value=30, value=10, step=1)


# --- Run model ---
pred = v2_expected_points(pm, fixtures, teams, horizon=horizon, prior_minutes=float(prior_minutes))

# Make sure price and selected% are available everywhere
# now_cost is the FPL tenths (e.g. 41 -> £4.1m). Create consistent columns for display.
if "now_cost" in pred.columns:
    pred["£m"] = pred["now_cost"].astype(float) / 10.0
else:
    # if model renamed price, attempt common alternatives
    if "now_cost" not in pred.columns and "now_cost" in players.columns:
        pred["£m"] = players.set_index("id")["now_cost"].reindex(pred["id"]).astype(float) / 10.0
    else:
        pred["£m"] = pd.NA

# Add pretty selected by percent column (if raw exists)
if "selected_by_percent" in pred.columns:
    # keep original numeric as well, create prettier string
    pred["selected_by_percent"] = pd.to_numeric(pred["selected_by_percent"], errors="coerce").fillna(0.0)
    pred["sel_by_%"] = pred["selected_by_percent"].map(lambda x: f"{x:.1f}%")
else:
    pred["selected_by_percent"] = 0.0
    pred["sel_by_%"] = "0.0%"

# Some model naming compatibility: ensure xPts_per_match and xPts_total exist
# Many model versions used different names; create consistent names if possible.
if "xPts_per_m" in pred.columns and "xPts_per_match" not in pred.columns:
    pred["xPts_per_match"] = pred["xPts_per_m"]
if "xPts" in pred.columns and "xPts_total" not in pred.columns:
    # If xPts in your model was total, use it; else compute if per-match exists
    if pred["xPts"].dtype != object:
        pred["xPts_total"] = pred["xPts"]
elif "xPts_per_match" in pred.columns and "xPts_total" not in pred.columns:
    pred["xPts_total"] = pred["xPts_per_match"] * pred.get("proj_matches", horizon)

# Safety conversions
for col in ["xPts_per_match", "xPts_total", "xPts_per_m"]:
    if col in pred.columns:
        pred[col] = pd.to_numeric(pred[col], errors="coerce").fillna(0.0)

# Leaderboard df (optionally hide low-minutes players)
if hide_low_minutes and min_minutes_for_lists > 0:
    leaderboard_df = pred[pred["minutes"].fillna(0) >= min_minutes_for_lists].copy()
else:
    leaderboard_df = pred.copy()


# --- Helper: Top picks by position ---
def top_by_position(df: pd.DataFrame, score_col: str, top_n: int = 10, gk_n: int = 3):
    pos_map = {"GKP": gk_n, "DEF": top_n, "MID": top_n, "FWD": top_n}
    result = {}
    for pos, n in pos_map.items():
        subset = df[df["pos"] == pos].sort_values(score_col, ascending=False).head(n)
        result[pos] = subset
    return result


# --- Captaincy Picks ---
st.subheader(f"🎯 Captaincy picks (Top by {'per match' if rank_metric == 'xPts_per_match' else 'total'} per position)")

capt_tables = top_by_position(leaderboard_df, rank_metric, top_n=top_n, gk_n=3)

for pos, table in capt_tables.items():
    st.markdown(f"**Top {len(table)} {pos}s by {'per match' if rank_metric == 'xPts_per_match' else 'total'}**")

    # Compose display columns and guarantee '£m' present
    if pos == "GKP":
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "cs_prob", "xSaves_per_match", rank_metric]
    elif pos == "DEF":
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", "att_factor", "cs_prob", rank_metric]
    elif pos in ("MID", "FWD"):
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", "att_factor", rank_metric]
    else:
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", rank_metric]

    display_cols = [c for c in display_cols if c in table.columns]
    st.dataframe(format_for_display(table, display_cols).reset_index(drop=True))


# --- Value Picks ---
st.subheader("💼 Value picks (Top by xPts per million per position)")
value_score = "xPts_per_m" if "xPts_per_m" in leaderboard_df.columns else ("xPts_per_match" if rank_metric == "xPts_per_match" else "xPts_total")
value_tables = top_by_position(leaderboard_df, value_score, top_n=top_n, gk_n=3)

for pos, table in value_tables.items():
    st.markdown(f"**Top {len(table)} {pos}s by value ({value_score})**")

    if pos == "GKP":
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "cs_prob", "xSaves_per_match", value_score]
    elif pos == "DEF":
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", "cs_prob", value_score]
    elif pos in ("MID", "FWD"):
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", value_score]
    else:
        display_cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", value_score]

    display_cols = [c for c in display_cols if c in table.columns]
    st.dataframe(format_for_display(table, display_cols).reset_index(drop=True))


# --- Analyze My Squad ---
st.subheader("🧩 Analyze My 15-man Squad")

player_options = {
    int(r.id): f"{r.web_name} ({r.team_name}, {r.pos}, £{(r.now_cost/10):.1f}m, {float(r.selected_by_percent) if 'selected_by_percent' in pred.columns else 0:.1f}%)"
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

    # Best XI by per-match (heuristic)
    best_xi = []
    best_xi.append(squad_df[squad_df["pos"] == "GKP"].sort_values("xPts_per_match", ascending=False).head(1))
    best_xi.append(squad_df[squad_df["pos"] == "DEF"].sort_values("xPts_per_match", ascending=False).head(3))
    best_xi.append(squad_df[squad_df["pos"] == "MID"].sort_values("xPts_per_match", ascending=False).head(4))
    best_xi.append(squad_df[squad_df["pos"] == "FWD"].sort_values("xPts_per_match", ascending=False).head(3))
    best_xi = pd.concat(best_xi).sort_values("xPts_per_match", ascending=False).head(11)

    st.markdown("### ✅ Best XI (sorted by xPts_per_match):")
    bcols = [c for c in ["web_name", "pos", "team_name", "£m", "sel_by_%", "xPts_per_match", "xPts_total"] if c in best_xi.columns]
    st.dataframe(format_for_display(best_xi, bcols).reset_index(drop=True))

    captain = best_xi.iloc[0]["web_name"]
    vice_captain = best_xi.iloc[1]["web_name"]
    st.success(f"⭐ Recommended Captain: **{captain}** | Vice Captain: **{vice_captain}**")

    # Subs
    subs = squad_df[~squad_df["id"].isin(best_xi["id"])].sort_values("xPts_per_match", ascending=False)
    sub_cols = [c for c in ["web_name", "pos", "team_name", "£m", "sel_by_%", "xPts_per_match", "xPts_total"] if c in subs.columns]
    st.markdown("### 🪑 Subs (bench, sorted by xPts_per_match):")
    st.dataframe(format_for_display(subs, sub_cols).reset_index(drop=True))

    # Transfer suggestions: use xPts_total to evaluate change across horizon
    st.markdown("---")
    st.subheader("🔁 Suggested Transfers")
    current_xi_pts_total = best_xi["xPts_total"].sum() if "xPts_total" in best_xi.columns else best_xi["xPts_per_match"].sum()

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

        in_player = candidates.sort_values("xPts_total", ascending=False).iloc[0]

        new_squad_ids = [pid for pid in squad_ids if pid != out_id] + [in_player["id"]]
        new_squad_df = pred[pred["id"].isin(new_squad_ids)]

        new_xi = []
        new_xi.append(new_squad_df[new_squad_df["pos"] == "GKP"].sort_values("xPts_total", ascending=False).head(1))
        new_xi.append(new_squad_df[new_squad_df["pos"] == "DEF"].sort_values("xPts_total", ascending=False).head(3))
        new_xi.append(new_squad_df[new_squad_df["pos"] == "MID"].sort_values("xPts_total", ascending=False).head(4))
        new_xi.append(new_squad_df[new_squad_df["pos"] == "FWD"].sort_values("xPts_total", ascending=False).head(3))
        new_xi = pd.concat(new_xi).head(11)

        new_pts_total = new_xi["xPts_total"].sum()
        gain = new_pts_total - current_xi_pts_total

        if gain > 0:
            transfer_candidates.append((gain, out_player, in_player, new_pts_total))

    transfer_candidates = sorted(transfer_candidates, key=lambda x: x[0], reverse=True)

    if transfer_candidates:
        st.markdown("#### 💡 Top 2 Transfer Suggestions:")
        for gain, out_p, in_p, new_total in transfer_candidates[:2]:
            st.success(f"**{out_p['web_name']} ➝ {in_p['web_name']}** (+{gain:.2f} xPts_total, new XI total = {new_total:.2f})")

        st.markdown("#### 🎯 Choose a player to transfer OUT:")
        out_choice = st.selectbox("Select player to sell", [p[1]["web_name"] for p in transfer_candidates])
        if out_choice:
            chosen = [t for t in transfer_candidates if t[1]["web_name"] == out_choice][0]
            gain, out_p, in_p, new_total = chosen
            st.info(f"Best replacement for **{out_p['web_name']}** ➝ **{in_p['web_name']}** (+{gain:.2f} xPts_total)")

    else:
        st.info("No beneficial transfers found within your bank & squad constraints.")

else:
    st.info("Please select exactly 15 players to analyze transfers.")

# Footer
st.markdown("---")
st.caption("Note: '£m' = FPL now_cost / 10. Use the 'Min minutes' slider to exclude low-sample players from leaderboards.")
