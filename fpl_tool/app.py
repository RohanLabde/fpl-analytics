import itertools
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


# --- Helper: formatting
def format_for_display(df, cols):
    out = df.copy()
    if "£m" in out.columns:
        out["£m"] = pd.to_numeric(out["£m"], errors="coerce").round(1)
    if "sel_by_%" in out.columns:
        # keep already formatted strings (e.g. "12.3%")
        pass
    # round numeric metrics for nicer display
    for c in ["xPts_per_match", "xPts_total", "xPts_per_m", "xAttack_per90", "xSaves_per_match", "cs_prob", "att_factor"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(3)
    final = [c for c in cols if c in out.columns]
    return out[final]


# --- formation utilities (valid FPL formations) ---
VALID_FORMATIONS = {
    (3, 4, 3),
    (3, 5, 2),
    (4, 4, 2),
    (4, 3, 3),
    (5, 3, 2),
    (5, 4, 1),
}

def is_valid_outfield_counts(defs, mids, fwds):
    return (defs, mids, fwds) in VALID_FORMATIONS


def best_xi_optimal(squad_df, score_col="xPts_per_match"):
    """
    From a 15-man squad DataFrame, return the optimal Best XI (DataFrame)
    that maximizes score_col subject to FPL formation rules:
      - exactly 1 GK
      - an outfield split that is in VALID_FORMATIONS
    We iterate all combinations C(15,11) (1,365 combos) — cheap.
    """
    # require id column for combinatorics
    if "id" not in squad_df.columns:
        raise ValueError("squad_df must contain 'id' column")
    ids = squad_df["id"].tolist()
    best_score = -1e9
    best_combo = None

    # Pre-snapshot: mapping id -> attributes for speed
    meta = squad_df.set_index("id")
    # iterate all 11-player combos
    for combo in itertools.combinations(ids, 11):
        combo_df = meta.loc[list(combo)]

        # must contain exactly 1 GK
        gk_count = (combo_df["pos"] == "GKP").sum()
        if gk_count != 1:
            continue

        # outfield counts
        defs = (combo_df["pos"] == "DEF").sum()
        mids = (combo_df["pos"] == "MID").sum()
        fwds = (combo_df["pos"] == "FWD").sum()

        if defs + mids + fwds != 10:
            continue  # sanity

        if not is_valid_outfield_counts(defs, mids, fwds):
            continue

        # compute score (use numeric sum; missing => 0)
        score = combo_df[score_col].astype(float).sum()
        if score > best_score:
            best_score = score
            best_combo = list(combo)

    if best_combo is None:
        # fallback to greedy (shouldn't happen)
        # 1 GK, top 10 outfield by score_col (subject to min defs 3)
        gk = squad_df[squad_df["pos"] == "GKP"].sort_values(score_col, ascending=False).head(1)
        outfield = squad_df[squad_df["pos"] != "GKP"].sort_values(score_col, ascending=False).head(10)
        return pd.concat([gk, outfield])
    else:
        return meta.loc[best_combo].reset_index(drop=False)  # keep id as column


# --- Streamlit UI ---
st.set_page_config(page_title="FPL Analytics – Best XI optimizer", layout="wide")
st.title("⚽ FPL Analytics – Smarter Expected Points (Best XI optimizer)")
st.caption("Pick 15 players, app will compute optimal Best XI across valid FPL formations.")

# Load data
players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()
pm = build_player_master(players, teams, element_types)

# Sidebar controls
st.sidebar.header("Model + display settings")
horizon = st.sidebar.slider("Fixture horizon (matches)", 1, 10, 5)
rank_metric = st.sidebar.selectbox("Ranking metric for selection & leaderboards", ["xPts_per_match", "xPts_total"])
min_minutes = st.sidebar.slider("Min minutes to include in leaderboards (0 = include all)", 0, 900, 90)
hide_low_minutes = st.sidebar.checkbox("Hide players below min-minutes in leaderboards", value=True)
top_n = st.sidebar.number_input("Top N per position shown", min_value=1, max_value=20, value=10)

# Run model (expects model to return xPts_per_match and xPts_total; add_value_columns if needed)
pred = v2_expected_points(pm, fixtures, teams, horizon=horizon)
pred = add_value_columns(pred) if "xPts_per_m" not in pred.columns else pred

# Normalize / ensure price & selected percent columns exist for display everywhere
if "now_cost" in pred.columns:
    pred["£m"] = pred["now_cost"].astype(float) / 10.0
else:
    pred["£m"] = (pred.get("now_cost", pd.Series(0, index=pred.index)).astype(float) / 10.0)

if "selected_by_percent" in pred.columns:
    pred["selected_by_percent"] = pd.to_numeric(pred["selected_by_percent"], errors="coerce").fillna(0.0)
else:
    pred["selected_by_percent"] = 0.0
pred["sel_by_%"] = pred["selected_by_percent"].map(lambda x: f"{x:.1f}%")

# Ensure canonical metric names exist
if "xPts_per_match" not in pred.columns and "xPts_per_m" in pred.columns:
    pred["xPts_per_match"] = pred["xPts_per_m"]
if "xPts_total" not in pred.columns:
    # if model produced xPts as total use that; else approximate total = per_match * horizon
    if "xPts" in pred.columns and pred["xPts"].mean() > 0:
        pred["xPts_total"] = pred["xPts"]
    else:
        pred["xPts_total"] = pred.get("xPts_per_match", 0.0) * horizon

# Optional leaderboard df filter
leaderboard_df = pred.copy()
if hide_low_minutes and min_minutes > 0 and "minutes" in leaderboard_df.columns:
    leaderboard_df = leaderboard_df[leaderboard_df["minutes"].fillna(0) >= min_minutes]

# Helper: top by position
def top_by_position(df, score_col, top_n=10, gk_n=3):
    pos_map = {"GKP": gk_n, "DEF": top_n, "MID": top_n, "FWD": top_n}
    out = {}
    for pos, n in pos_map.items():
        out[pos] = df[df["pos"] == pos].sort_values(score_col, ascending=False).head(n)
    return out

# Captaincy / leaderboards
st.subheader(f"🎯 Captaincy picks (Top by {rank_metric} per position)")
captains = top_by_position(leaderboard_df, rank_metric, top_n=top_n, gk_n=3)
for pos, tbl in captains.items():
    st.markdown(f"**Top {len(tbl)} {pos}s by {rank_metric}**")
    if pos == "GKP":
        cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "cs_prob", "xSaves_per_match", rank_metric, "xPts_total"]
    elif pos == "DEF":
        cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", "att_factor", "cs_prob", rank_metric, "xPts_total"]
    else:
        cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", "att_factor", rank_metric, "xPts_total"]
    st.dataframe(format_for_display(tbl, cols).reset_index(drop=True))

# Value picks
st.subheader("💼 Value picks (Top by xPts_per_m per position)")
value_score = "xPts_per_m" if "xPts_per_m" in pred.columns else "xPts_per_match"
values = top_by_position(leaderboard_df, value_score, top_n=top_n, gk_n=3)
for pos, tbl in values.items():
    st.markdown(f"**Top {len(tbl)} {pos}s by value**")
    if pos == "GKP":
        cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "cs_prob", "xSaves_per_match", value_score]
    elif pos == "DEF":
        cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", "cs_prob", value_score]
    else:
        cols = ["web_name", "team_name", "pos", "£m", "sel_by_%", "xAttack_per90", value_score]
    st.dataframe(format_for_display(tbl, cols).reset_index(drop=True))

# Analyze my 15-man squad
st.subheader("🧩 Analyze My 15-man Squad")

player_options = {
    int(r.id): f"{r.web_name} ({r.team_name}, {r.pos}, £{(r.£m if '£m' in r._fields else (r.now_cost/10 if 'now_cost' in r._fields else 0))}m, {getattr(r, 'selected_by_percent', 0):.1f}%)"
    for r in pred.itertuples()
}

squad_ids = st.multiselect("Select your 15 players", options=list(player_options.keys()), format_func=lambda x: player_options[x])
bank = st.number_input("Bank (money in the bank, £m)", min_value=0.0, step=0.1)

if len(squad_ids) == 15:
    squad_df = pred[pred["id"].isin(squad_ids)].copy()

    # Find optimal Best XI (maximize rank_metric while keeping formation valid)
    best_xi = best_xi_optimal(squad_df, score_col=rank_metric)

    # If best_xi returned without index info, ensure we have a DataFrame
    if best_xi is None or best_xi.empty:
        st.error("Could not compute best XI - unexpected error.")
    else:
        # Show best XI sorted by per-match metric
        st.markdown("### ✅ Best XI (optimal across valid formations):")
        show_cols = [c for c in ["web_name", "pos", "team_name", "£m", "sel_by_%", "xPts_per_match", "xPts_total"] if c in best_xi.columns]
        st.dataframe(format_for_display(best_xi.sort_values(rank_metric, ascending=False), show_cols).reset_index(drop=True))

        # Captain & vice (top two from chosen XI by rank_metric)
        ranked = best_xi.sort_values(rank_metric, ascending=False)
        captain = ranked.iloc[0]["web_name"]
        vice = ranked.iloc[1]["web_name"] if ranked.shape[0] > 1 else ""
        st.success(f"⭐ Recommended Captain: **{captain}** | Vice Captain: **{vice}**")

        # Subs = remaining squad members (not in best_xi)
        subs = squad_df[~squad_df["id"].isin(best_xi["id"])].sort_values(rank_metric, ascending=False)
        st.markdown("### 🪑 Subs (bench, sorted by per-match metric):")
        sub_cols = [c for c in ["web_name", "pos", "team_name", "£m", "sel_by_%", "xPts_per_match", "xPts_total"] if c in subs.columns]
        st.dataframe(format_for_display(subs, sub_cols).reset_index(drop=True))

        # Transfers (same logic as before: single-out best replacement)
        st.markdown("---")
        st.subheader("🔁 Suggested Transfers (single out -> in)")

        current_total = best_xi["xPts_total"].sum() if "xPts_total" in best_xi.columns else best_xi[rank_metric].sum()
        transfer_candidates = []

        for out_id in squad_ids:
            out_player = pred[pred["id"] == out_id].iloc[0]
            budget_available = bank * 10 + out_player["now_cost"]  # now_cost in tenths
            candidates = pred[
                (pred["pos"] == out_player["pos"]) &
                (~pred["id"].isin(squad_ids)) &
                (pred["now_cost"] <= budget_available)
            ]
            if candidates.empty:
                continue
            # pick top by total across horizon
            in_player = candidates.sort_values("xPts_total", ascending=False).iloc[0]

            new_squad_ids = [pid for pid in squad_ids if pid != out_id] + [in_player["id"]]
            new_squad_df = pred[pred["id"].isin(new_squad_ids)]

            # compute best XI for the new squad (optimal formation)
            new_best_xi = best_xi_optimal(new_squad_df, score_col="xPts_total")
            new_total = new_best_xi["xPts_total"].sum() if "xPts_total" in new_best_xi.columns else new_best_xi[rank_metric].sum()
            gain = new_total - current_total
            if gain > 0:
                transfer_candidates.append((gain, out_player, in_player, new_total))

        transfer_candidates = sorted(transfer_candidates, key=lambda x: x[0], reverse=True)

        if transfer_candidates:
            st.markdown("#### 💡 Top 2 Transfer Suggestions:")
            for gain, out_p, in_p, new_total in transfer_candidates[:2]:
                st.success(f"**{out_p['web_name']} ➝ {in_p['web_name']}** (+{gain:.2f} xPts_total, new XI total = {new_total:.2f})")

            st.markdown("#### 🎯 Choose a player to transfer OUT (from suggestions):")
            out_choice = st.selectbox("Select player to sell", [p[1]["web_name"] for p in transfer_candidates])
            if out_choice:
                chosen = [t for t in transfer_candidates if t[1]["web_name"] == out_choice][0]
                gain, out_p, in_p, new_total = chosen
                st.info(f"Best replacement for **{out_p['web_name']}** ➝ **{in_p['web_name']}** (+{gain:.2f} xPts_total)")

        else:
            st.info("No beneficial transfer found within your constraints.")
else:
    st.info("Please select exactly 15 players to analyze transfers.")
