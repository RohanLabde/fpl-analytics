# app.py
import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Import model functions
from fpl_tool.model import build_player_master, v2_expected_points

# Optional
try:
    from fpl_tool.model import add_value_columns
except Exception:
    add_value_columns = None


# -----------------------
# CACHED LOADERS
# -----------------------
@st.cache_data(ttl=3600)
def load_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url, timeout=10)
    data = r.json()
    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    element_types = pd.DataFrame(data["element_types"])
    return players, teams, element_types


@st.cache_data(ttl=3600)
def load_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    r = requests.get(url, timeout=10)
    return pd.DataFrame(r.json())


# -----------------------
# DISPLAY HELPERS
# -----------------------
def fmt_df_for_display(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()

    if "now_cost" in out.columns and "£m" not in out.columns:
        out["£m"] = out["now_cost"] / 10.0

    if "selected_by_percent" in out.columns:
        out["sel_by_%"] = pd.to_numeric(out["selected_by_percent"], errors="coerce").fillna(0).map(lambda x: f"{x:.1f}%")
    else:
        out["sel_by_%"] = ""

    if "xPts_total" not in out.columns and "xPts" in out.columns:
        out["xPts_total"] = out["xPts"]

    if "xPts_per_match" not in out.columns and "games_proj" in out.columns:
        out["xPts_per_match"] = out["xPts_total"] / out["games_proj"].replace(0, 1)

    final_cols = [c for c in cols if c in out.columns]
    return out[final_cols]


# -----------------------
# BEST XI (formation search)
# -----------------------
FORMATIONS = [
    (3, 4, 3), (4, 4, 2), (3, 5, 2), (4, 3, 3), (5, 3, 2), (4, 5, 1), (5, 4, 1)
]


def build_best_xi_from_squad(squad_df: pd.DataFrame, rank_by: str) -> pd.DataFrame:
    df = squad_df.copy()
    if df.empty:
        return df

    best_total = -1e12
    best_xi = pd.DataFrame()

    gk_pool = df[df["pos"].isin(["GKP", "GK"])].sort_values(rank_by, ascending=False)
    if gk_pool.empty:
        return df.sort_values(rank_by, ascending=False).head(11)

    for d, m, f in FORMATIONS:
        xi = pd.concat([
            gk_pool.head(1),
            df[df["pos"] == "DEF"].sort_values(rank_by, ascending=False).head(d),
            df[df["pos"] == "MID"].sort_values(rank_by, ascending=False).head(m),
            df[df["pos"] == "FWD"].sort_values(rank_by, ascending=False).head(f),
        ])
        if len(xi) != 1 + d + m + f:
            continue
        total = xi[rank_by].sum()
        if total > best_total:
            best_total = total
            best_xi = xi.copy()

    if best_xi.empty:
        best_xi = df.sort_values(rank_by, ascending=False).head(11)

    return best_xi.sort_values(rank_by, ascending=False)


# -----------------------
# TRANSFER SUGGESTER (greedy)
# -----------------------
def suggest_transfers_greedy(pred_df, squad_ids, bank, rank_by, min_minutes, max_outs=3, top_n_suggestions=2):
    results = []
    pred = pred_df.copy()
    idx = pred.set_index("id", drop=False)

    squad_df = pred[pred["id"].isin(squad_ids)].copy()
    current_xi = build_best_xi_from_squad(squad_df, rank_by)
    current_total = current_xi[rank_by].sum() if not current_xi.empty else 0

    for r in range(1, min(max_outs, len(squad_ids)) + 1):
        for out_combo in itertools.combinations(squad_ids, r):
            freed = sum([idx.loc[o]["now_cost"] for o in out_combo])
            budget = bank * 10 + freed
            base_ids = [x for x in squad_ids if x not in out_combo]

            pool = pred[
                (~pred["id"].isin(base_ids)) &
                (~pred["id"].isin(out_combo)) &
                (pred["minutes"] >= min_minutes)
            ].copy()

            outs = [idx.loc[o].to_dict() for o in out_combo]
            outs = sorted(outs, key=lambda r: r.get(rank_by, 0), reverse=True)

            chosen = []
            budget_left = budget
            ok = True

            for o in outs:
                cands = pool[(pool["pos"] == o["pos"]) & (pool["now_cost"] <= budget_left)]
                if cands.empty:
                    ok = False
                    break
                pick = cands.sort_values(rank_by, ascending=False).iloc[0]
                chosen.append(pick.to_dict())
                pool = pool[pool["id"] != pick["id"]]
                budget_left -= pick["now_cost"]

            if not ok:
                continue

            new_ids = base_ids + [int(p["id"]) for p in chosen]
            new_df = pred[pred["id"].isin(new_ids)]
            new_xi = build_best_xi_from_squad(new_df, rank_by)
            new_total = new_xi[rank_by].sum() if not new_xi.empty else 0
            gain = new_total - current_total

            if gain > 0:
                results.append((gain, outs, chosen, new_total))

    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_n_suggestions]
    return results


# -----------------------
# STREAMLIT UI
# -----------------------
st.set_page_config(page_title="FPL Analytics", layout="wide")
st.title("⚽ FPL Analytics – Smarter Expected Points")

players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

pm = build_player_master(players, teams, element_types)

# Sidebar controls
st.sidebar.header("Model Controls")
horizon = st.sidebar.slider("Fixture horizon", 1, 10, 5)
form_w = st.sidebar.slider("Form weight", 0.0, 1.0, 0.25, 0.01)
bonus_w = st.sidebar.slider("Bonus weight", 0.0, 0.5, 0.05, 0.01)

rank_by_choice = st.sidebar.selectbox("Rank by", ["xPts_total", "xPts_per_match"])
min_minutes = st.sidebar.slider("Min minutes filter", 0, 2000, 270)
top_n = st.sidebar.number_input("Top N per position", 1, 20, 10)

# Run model
pred = v2_expected_points(pm.copy(), fixtures.copy(), teams.copy(), horizon=horizon, form_weight=form_w, bonus_weight=bonus_w)

# Clamp games_proj
pred["games_proj"] = np.minimum(horizon, np.maximum(1, (pred["minutes"] / 90).fillna(1)))

# Value column
if add_value_columns:
    pred = add_value_columns(pred)
else:
    pred["xPts_per_m"] = pred["xPts_total"] / (pred["now_cost"] / 10)

pred["£m"] = pred["now_cost"] / 10


# -----------------------
# PLAYER TABLES
# -----------------------
st.subheader("🎯 Captaincy Picks")

for pos, n in {"GKP": 3, "DEF": top_n, "MID": top_n, "FWD": top_n}.items():
    df = pred[pred["pos"] == pos]
    df = df[df["minutes"] >= min_minutes]
    df = df.sort_values(rank_by_choice, ascending=False).head(n)

    st.markdown(f"**Top {len(df)} {pos}**")
    st.dataframe(fmt_df_for_display(df, ["web_name", "team_name", "pos", "£m", "selected_by_percent", "xPts_per_match", "xPts_total"]))


st.subheader("💼 Value Picks")

for pos, n in {"GKP": 3, "DEF": top_n, "MID": top_n, "FWD": top_n}.items():
    df = pred[pred["pos"] == pos]
    df = df[df["minutes"] >= min_minutes]
    df = df.sort_values("xPts_per_m", ascending=False).head(n)

    st.markdown(f"**Top {len(df)} {pos}**")
    st.dataframe(fmt_df_for_display(df, ["web_name", "team_name", "pos", "£m", "selected_by_percent", "xPts_per_m", "xPts_total"]))


# -----------------------
# TEAM DASHBOARD (NEW SECTION)
# -----------------------
st.markdown("---")
st.header("📊 Team Strength & Form Dashboard (Context & Insights)")

# ---- Table 1: Season Strength ----
st.subheader("🏟️ Season Team Strength")

team_stats = teams[["id", "name", "played", "points", "win", "draw", "loss"]].copy()

# goals from fixtures
played_fixtures = fixtures.dropna(subset=["team_h_score", "team_a_score"])

gf = []
ga = []
cs = []

for _, t in team_stats.iterrows():
    tid = t["id"]
    home = played_fixtures[played_fixtures["team_h"] == tid]
    away = played_fixtures[played_fixtures["team_a"] == tid]

    goals_for = home["team_h_score"].sum() + away["team_a_score"].sum()
    goals_against = home["team_a_score"].sum() + away["team_h_score"].sum()

    clean_sheets = (home["team_a_score"] == 0).sum() + (away["team_h_score"] == 0).sum()

    gf.append(goals_for)
    ga.append(goals_against)
    cs.append(clean_sheets)

team_stats["GF"] = gf
team_stats["GA"] = ga
team_stats["CS"] = cs
team_stats["GF_per_match"] = team_stats["GF"] / team_stats["played"].replace(0, 1)
team_stats["GA_per_match"] = team_stats["GA"] / team_stats["played"].replace(0, 1)
team_stats["CS_%"] = (team_stats["CS"] / team_stats["played"].replace(0, 1)) * 100

st.dataframe(team_stats.sort_values("points", ascending=False))


# ---- Table 2: Recent Form ----
st.subheader("🔥 Recent Form")

recent_n = st.slider("Recent form window (matches)", 3, 10, 5)

recent_rows = []

for _, t in teams.iterrows():
    tid = t["id"]
    tf = played_fixtures[(played_fixtures["team_h"] == tid) | (played_fixtures["team_a"] == tid)].copy()
    tf = tf.sort_values("kickoff_time").tail(recent_n)

    pts = 0
    gf = 0
    ga = 0
    cs = 0

    for _, fx in tf.iterrows():
        if fx["team_h"] == tid:
            s_for, s_against = fx["team_h_score"], fx["team_a_score"]
        else:
            s_for, s_against = fx["team_a_score"], fx["team_h_score"]

        gf += s_for
        ga += s_against
        if s_against == 0:
            cs += 1
        if s_for > s_against:
            pts += 3
        elif s_for == s_against:
            pts += 1

    recent_rows.append({
        "Team": t["name"],
        "Matches": len(tf),
        "Points": pts,
        "GF": gf,
        "GA": ga,
        "CS": cs,
        "GF_per_match": gf / max(1, len(tf)),
        "GA_per_match": ga / max(1, len(tf)),
    })

recent_df = pd.DataFrame(recent_rows)
st.dataframe(recent_df.sort_values("Points", ascending=False))


# ---- Table 3: FPL Production ----
st.subheader("🎯 FPL Production by Team")

prod = pm.groupby("team_name").agg(
    Goals=("goals_scored", "sum"),
    Assists=("assists", "sum"),
    CleanSheets=("clean_sheets", "sum"),
    Bonus=("bonus", "sum"),
    xG=("expected_goals", "sum"),
    xA=("expected_assists", "sum"),
).reset_index()

st.dataframe(prod.sort_values("Goals", ascending=False))

