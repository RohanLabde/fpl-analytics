# app.py
import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# import your model functions (adjust if your module path differs)
from fpl_tool.model import build_player_master, v2_expected_points

# add_value_columns is optional; try to import and fall back to local calculation
try:
    from fpl_tool.model import add_value_columns  # type: ignore
except Exception:
    add_value_columns = None


# -----------------------
# CACHED DATA LOADERS
# -----------------------
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


# -----------------------
# DISPLAY / FORMAT HELPERS
# -----------------------
def fmt_df_for_display(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Prepare DataFrame for display:
    - ensure price column '£m' exists (converted from now_cost)
    - ensure selection percent 'sel_by_%' exists
    - ensure xPts_total and xPts_per_match exist
    - keep only requested columns that exist
    """
    out = df.copy()

    # price
    if "now_cost" in out.columns and "£m" not in out.columns:
        out["£m"] = out["now_cost"] / 10.0

    # selection %
    if "selected_by_percent" in out.columns:
        out["sel_by_%"] = pd.to_numeric(out["selected_by_percent"], errors="coerce").fillna(0).map(
            lambda x: f"{x:.1f}%"
        )
    elif "sel_by" in out.columns:
        out["sel_by_%"] = pd.to_numeric(out["sel_by"], errors="coerce").fillna(0).map(lambda x: f"{x:.1f}%")
    else:
        # create empty col to keep display uniform when requested
        out["sel_by_%"] = ""

    # xPts_total fallback (some model variants use 'xPts' as total)
    if "xPts_total" not in out.columns and "xPts" in out.columns:
        out["xPts_total"] = out["xPts"].astype(float)

    # games_proj fallback (if missing, default to 1)
    if "games_proj" not in out.columns:
        out["games_proj"] = np.maximum(1.0, (pd.to_numeric(out.get("minutes", 0), errors="coerce").fillna(0) / 90.0).clip(upper=1))

    # xPts_per_match
    def calc_per_match(r):
        tot = r.get("xPts_total", None)
        gp = r.get("games_proj", None)
        try:
            if gp and gp > 0 and tot is not None:
                return float(tot) / float(gp)
        except Exception:
            pass
        # fallback try xPts
        if "xPts" in r and r.get("games_proj", 0) > 0:
            return float(r["xPts"]) / float(r["games_proj"])
        return 0.0

    if "xPts_per_match" not in out.columns:
        out["xPts_per_match"] = out.apply(calc_per_match, axis=1)

    # Ensure numeric columns are floats for consistent display
    for c in ["xPts_per_match", "xPts_total", "£m"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    final_cols = [c for c in cols if c in out.columns]
    return out[final_cols]


# -----------------------
# BEST XI / FORMATION SEARCH
# -----------------------
FORMATIONS = [
    (3, 4, 3),
    (4, 4, 2),
    (3, 5, 2),
    (4, 3, 3),
    (5, 3, 2),
    (4, 5, 1),
    (5, 4, 1),
]


def build_best_xi_from_squad(squad_df: pd.DataFrame, rank_by: str = "xPts_per_match") -> pd.DataFrame:
    """
    Try multiple formations and return the 11 players (best XI) maximizing sum of rank_by.
    GK is always 1. If formations fail (insufficient position supply), fallback to top-11 by rank_by.
    """
    df = squad_df.copy()
    if df.empty:
        return df

    if rank_by not in df.columns:
        df[rank_by] = 0.0

    # ensure positions exist
    df["pos"] = df["pos"].fillna("UNK")

    best_total = -1e12
    best_xi = pd.DataFrame()

    # GK pool
    gk_pool = df[df["pos"].isin(["GKP", "GK"])].sort_values(rank_by, ascending=False)
    if gk_pool.empty:
        # fallback to top 11 overall
        return df.sort_values(rank_by, ascending=False).head(11)

    for def_c, mid_c, fwd_c in FORMATIONS:
        # pick GK
        sel_gk = gk_pool.head(1)
        defs = df[df["pos"] == "DEF"].sort_values(rank_by, ascending=False).head(def_c)
        mids = df[df["pos"] == "MID"].sort_values(rank_by, ascending=False).head(mid_c)
        fwds = df[df["pos"] == "FWD"].sort_values(rank_by, ascending=False).head(fwd_c)

        combo = pd.concat([sel_gk, defs, mids, fwds])
        needed = 1 + def_c + mid_c + fwd_c
        if len(combo) != needed:
            continue
        total = combo[rank_by].sum()
        if total > best_total:
            best_total = total
            best_xi = combo.copy()

    if best_xi.empty:
        # fallback
        best_xi = df.sort_values(rank_by, ascending=False).head(11)

    return best_xi.sort_values(rank_by, ascending=False)


# -----------------------
# TRANSFER SUGGESTIONS (greedy)
# -----------------------
def suggest_transfers_greedy(
    pred_df: pd.DataFrame,
    squad_ids: List[int],
    bank: float,
    rank_by: str,
    min_minutes: int,
    max_outs: int = 3,
    top_n_suggestions: int = 2,
) -> Tuple[List[Tuple[float, List[dict], List[dict], float]], List[dict]]:
    """
    Greedy search for best replacements for every combination of outs up to `max_outs`.
    Returns sorted list of suggestions: (gain, outs_list, ins_list, new_total)
    Also returns sample ins for UI convenience (ins for best suggestion).
    """
    results = []
    pred = pred_df.copy()
    pred_indexed = pred.set_index("id", drop=False)

    # current best XI total
    squad_df = pred[pred["id"].isin(squad_ids)].copy()
    current_best_xi = build_best_xi_from_squad(squad_df, rank_by)
    current_total = float(current_best_xi[rank_by].sum()) if not current_best_xi.empty else 0.0

    # candidate pool (global constraints)
    def candidate_pool_mask(df: pd.DataFrame, excluded_ids: List[int]):
        m = ~df["id"].isin(excluded_ids) & (pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0) >= min_minutes)
        return m

    squad_list = list(squad_ids)

    # all combinations of 1..max_outs
    combs = []
    for r in range(1, min(max_outs, len(squad_list)) + 1):
        combs.extend(itertools.combinations(squad_list, r))

    for out_combo in combs:
        # freed budget (tenths of £m)
        try:
            freed = sum([float(pred_indexed.loc[out_id]["now_cost"]) for out_id in out_combo])
        except Exception:
            continue
        budget_available = bank * 10 + freed

        base_ids = [pid for pid in squad_list if pid not in out_combo]

        # pool excludes base players and outs
        pool = pred[candidate_pool_mask(pred, excluded_ids=base_ids + list(out_combo))].copy()
        # sort outs by lost value (rank_by) so we replace the most impactful outs first
        outs_rows = [pred_indexed.loc[o].to_dict() for o in out_combo]
        outs_rows_sorted = sorted(outs_rows, key=lambda r: r.get(rank_by, 0), reverse=True)

        chosen_ins = []
        budget_remaining = budget_available

        # pick replacements greedily
        ok = True
        for out_row in outs_rows_sorted:
            pos = out_row.get("pos", None)
            if pos is None:
                ok = False
                break
            # candidates same pos and within budget_remaining
            cands = pool[(pool["pos"] == pos) & (pool["now_cost"] <= budget_remaining)].copy()
            if cands.empty:
                # try relaxing the budget to full available (rare)
                cands = pool[(pool["pos"] == pos) & (pool["now_cost"] <= budget_available)].copy()
            if cands.empty:
                ok = False
                break
            cands = cands.sort_values(rank_by, ascending=False)
            pick = cands.iloc[0]
            chosen_ins.append(pick.to_dict())
            # remove pick from pool
            pool = pool[pool["id"] != pick["id"]]
            budget_remaining -= float(pick["now_cost"])

        if not ok:
            continue

        new_ids = base_ids + [int(p["id"]) for p in chosen_ins]
        if len(new_ids) != len(squad_list):
            continue

        new_squad_df = pred[pred["id"].isin(new_ids)].copy()
        new_best_xi = build_best_xi_from_squad(new_squad_df, rank_by)
        new_total = float(new_best_xi[rank_by].sum()) if not new_best_xi.empty else 0.0
        gain = new_total - current_total

        if gain > 0:
            results.append((gain, outs_rows_sorted, chosen_ins, new_total))

    results_sorted = sorted(results, key=lambda x: x[0], reverse=True)[:top_n_suggestions]
    sample_ins = results_sorted[0][2] if results_sorted else []
    return results_sorted, sample_ins


# -----------------------
# STREAMLIT UI
# -----------------------
st.set_page_config(page_title="FPL Analytics – Smarter Expected Points", layout="wide")
st.title("⚽ FPL Analytics – Smarter Expected Points")
st.caption("Data: Official Fantasy Premier League API. Model uses v2 (minutes + Poisson clean sheets + attacking proxy).")

# load data
players_df, teams_df, element_types_df = load_fpl_data()
fixtures_df = load_fixtures()

# build player master (adds team_name and pos)
pm = build_player_master(players_df.copy(), teams_df.copy(), element_types_df.copy())

# Sidebar controls
st.sidebar.header("Model & display settings")
horizon = st.sidebar.slider("Fixture horizon (matches)", 1, 10, 5)

rank_by_choice = st.sidebar.selectbox("Rank by (per match or total)", ["xPts_per_match", "xPts_total"], index=1)
min_minutes_for_leaderboards = st.sidebar.slider("Min historical minutes for leaderboards (0 = no filter)", 0, 2000, 270)
top_n_per_position = st.sidebar.number_input("Top N per position", min_value=1, max_value=20, value=10, step=1)
max_outs_allowed = st.sidebar.number_input("Max outs for multi-transfer suggestions", min_value=1, max_value=3, value=3, step=1)

# run model (v2) - expects build_player_master & v2_expected_points in model.py
pred = v2_expected_points(pm.copy(), fixtures_df.copy(), teams_df.copy(), horizon=horizon)

# Ensure games_proj exists and is clamped between 1 and horizon
minutes_series = pd.to_numeric(pred.get("minutes", 0), errors="coerce").fillna(0)
calc_games = (minutes_series / 90.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
# clamp lower bound to 1 to avoid tiny denominators, and upper to horizon
pred["games_proj"] = np.minimum(horizon, np.maximum(1.0, calc_games))

# Ensure xPts_total exists (some model versions produce 'xPts' instead)
if "xPts_total" not in pred.columns and "xPts" in pred.columns:
    pred["xPts_total"] = pd.to_numeric(pred["xPts"], errors="coerce").fillna(0.0)
elif "xPts_total" not in pred.columns:
    # fallback zero
    pred["xPts_total"] = 0.0

# xPts_per_match (safe: divide by clamped games_proj)
pred["xPts_per_match"] = pred.apply(
    lambda r: float(r["xPts_total"]) / float(r["games_proj"]) if (r.get("games_proj", 0) > 0) else float(r.get("xPts_total", 0.0)),
    axis=1,
)

# value metric
if add_value_columns is not None:
    try:
        pred = add_value_columns(pred)
    except Exception:
        pred["xPts_per_m"] = pred["xPts_total"] / (pred["now_cost"] / 10.0).replace(0, np.nan)
else:
    pred["xPts_per_m"] = pred["xPts_total"] / (pred["now_cost"] / 10.0).replace(0, np.nan)

# numeric selected_by_percent if present
if "selected_by_percent" in pred.columns:
    pred["selected_by_percent"] = pd.to_numeric(pred["selected_by_percent"], errors="coerce").fillna(0.0)
elif "sel_by" in pred.columns:
    pred["selected_by_percent"] = pd.to_numeric(pred["sel_by"], errors="coerce").fillna(0.0)
else:
    pred["selected_by_percent"] = 0.0

# price column for display
if "now_cost" in pred.columns:
    pred["£m"] = pred["now_cost"] / 10.0
else:
    pred["£m"] = np.nan

# -----------------------
# LEADERBOARDS
# -----------------------
st.subheader(f"🎯 Captaincy picks (Top by {rank_by_choice} per position)")

pos_map = {"GKP": 3, "DEF": top_n_per_position, "MID": top_n_per_position, "FWD": top_n_per_position}
for pos, n in pos_map.items():
    df_pos = pred[pred["pos"] == pos].copy()
    if min_minutes_for_leaderboards > 0:
        df_pos = df_pos[pd.to_numeric(df_pos.get("minutes", 0), errors="coerce").fillna(0) >= min_minutes_for_leaderboards]
    df_pos = df_pos.sort_values(rank_by_choice, ascending=False).head(n)
    st.markdown(f"**Top {len(df_pos)} {pos}s by {rank_by_choice}**")

    if pos in ["MID", "FWD"]:
        display_cols = ["web_name", "team_name", "pos", "£m", "selected_by_percent", "xAttack", "att_factor", "xPts_per_match", "xPts_total"]
    elif pos == "DEF":
        display_cols = ["web_name", "team_name", "pos", "£m", "selected_by_percent", "xAttack", "att_factor", "cs_prob", "xPts_per_match", "xPts_total"]
    else:  # GKP
        display_cols = ["web_name", "team_name", "pos", "£m", "selected_by_percent", "cs_prob", "xSaves", "xPts_per_match", "xPts_total"]

    st.dataframe(fmt_df_for_display(df_pos, display_cols).reset_index(drop=True))


# -----------------------
# VALUE PICKS (by xPts_per_m)
# -----------------------
st.subheader("💼 Value picks (Top by xPts_per_m per position)")
for pos, n in pos_map.items():
    df_pos = pred[pred["pos"] == pos].copy()
    if min_minutes_for_leaderboards > 0:
        df_pos = df_pos[pd.to_numeric(df_pos.get("minutes", 0), errors="coerce").fillna(0) >= min_minutes_for_leaderboards]
    df_pos = df_pos.sort_values("xPts_per_m", ascending=False).head(n)
    st.markdown(f"**Top {len(df_pos)} {pos}s by xPts_per_m**")
    if pos in ["MID", "FWD"]:
        display_cols = ["web_name", "team_name", "pos", "£m", "selected_by_percent", "xAttack", "att_factor", "xPts_per_m", "xPts_per_match", "xPts_total"]
    elif pos == "DEF":
        display_cols = ["web_name", "team_name", "pos", "£m", "selected_by_percent", "xAttack", "att_factor", "cs_prob", "xPts_per_m", "xPts_per_match", "xPts_total"]
    else:
        display_cols = ["web_name", "team_name", "pos", "£m", "selected_by_percent", "cs_prob", "xSaves", "xPts_per_m", "xPts_per_match", "xPts_total"]

    st.dataframe(fmt_df_for_display(df_pos, display_cols).reset_index(drop=True))


# -----------------------
# ANALYZE MY SQUAD
# -----------------------
st.subheader("🧩 Analyze My 15-man Squad")

# Build player options safely (use records to avoid weird f-string attribute errors)
player_options_map = {}
for rec in pred.to_dict("records"):
    pid = int(rec.get("id"))
    price_m = (rec.get("now_cost") / 10.0) if rec.get("now_cost") is not None else None
    sel_pct = rec.get("selected_by_percent", 0.0)
    label = f"{rec.get('web_name')} ({rec.get('team_name')}, {rec.get('pos')}, " + (
        f"£{price_m:.1f}m, {sel_pct:.1f}%)" if price_m is not None else f"{sel_pct:.1f}%)"
    )
    player_options_map[pid] = label

squad_ids = st.multiselect(
    "Select your 15 players",
    options=list(player_options_map.keys()),
    format_func=lambda x: player_options_map.get(x, str(x)),
    max_selections=15,
)

bank = st.number_input("Bank (money in the bank, £m)", min_value=0.0, step=0.1)

if len(squad_ids) == 15:
    squad_df = pred[pred["id"].isin(squad_ids)].copy()

    # Build best XI by trying formations (using the chosen ranking metric)
    best_xi = build_best_xi_from_squad(squad_df, rank_by=rank_by_choice)
    st.markdown(f"### ✅ Best XI (sorted by {rank_by_choice}):")
    st.dataframe(fmt_df_for_display(best_xi, ["web_name", "pos", "team_name", "£m", "sel_by_%", "xPts_per_match", "xPts_total"]).reset_index(drop=True))

    # captain / vice
    if not best_xi.empty and len(best_xi) >= 2:
        captain = best_xi.iloc[0]["web_name"]
        vice = best_xi.iloc[1]["web_name"]
        st.success(f"⭐ Recommended Captain: **{captain}** | Vice Captain: **{vice}**")

    # Subs (bench)
    subs = squad_df[~squad_df["id"].isin(best_xi["id"])].sort_values(rank_by_choice, ascending=False)
    st.markdown(f"### 🪑 Subs (bench, sorted by {rank_by_choice}):")
    st.dataframe(fmt_df_for_display(subs, ["web_name", "pos", "team_name", "£m", "sel_by_%", "xPts_per_match", "xPts_total"]).reset_index(drop=True))

    # -----------------------
    # TRANSFER SUGGESTIONS
    # -----------------------
    st.markdown("---")
    st.subheader("🔁 Suggested Transfers (single-player + multi-player up to 3 outs supported)")

    current_xi_total = float(best_xi[rank_by_choice].sum()) if not best_xi.empty else 0.0

    suggestions, sample_ins = suggest_transfers_greedy(
        pred_df=pred,
        squad_ids=squad_ids,
        bank=bank,
        rank_by=rank_by_choice,
        min_minutes=min_minutes_for_leaderboards,
        max_outs=int(max_outs_allowed),
        top_n_suggestions=3,
    )

    if suggestions:
        st.markdown("#### 💡 Top transfer suggestions (greedy):")
        for gain, outs, ins, new_total in suggestions:
            out_names = ", ".join([o.get("web_name", "") for o in outs])
            in_names = ", ".join([i.get("web_name", "") for i in ins])
            st.success(f"**{out_names} ➝ {in_names}** (+{gain:.2f} {rank_by_choice}, new XI total = {new_total:.2f})")
    else:
        st.info("No beneficial transfers found within your squad/minutes/budget (greedy search).")

    # UI to choose up to 3 outs and get recommendations for exactly those outs
    st.markdown("#### 🎯 Choose up to 3 players to transfer OUT (we'll suggest replacements):")
    out_choices = st.multiselect(
        "Select players to sell (up to 3)",
        options=squad_ids,
        format_func=lambda x: player_options_map.get(x, str(x)),
        max_selections=3,
    )

    if out_choices:
        # compute freed budget
        freed = sum([float(pred.set_index("id").loc[o]["now_cost"]) for o in out_choices])
        budget_available = bank * 10 + freed

        base_ids = [pid for pid in squad_ids if pid not in out_choices]

        pool = pred[
            (~pred["id"].isin(base_ids))
            & (~pred["id"].isin(out_choices))
            & (pd.to_numeric(pred.get("minutes", 0), errors="coerce").fillna(0) >= min_minutes_for_leaderboards)
        ].copy()

        outs_rows = [pred.set_index("id").loc[o].to_dict() for o in out_choices]
        outs_rows_sorted = sorted(outs_rows, key=lambda r: r.get(rank_by_choice, 0), reverse=True)

        chosen_ins_local = []
        budget_remaining = budget_available
        ok = True

        for out_row in outs_rows_sorted:
            pos = out_row.get("pos")
            if pos is None:
                ok = False
                break
            cands = pool[(pool["pos"] == pos) & (pool["now_cost"] <= budget_remaining)].sort_values(rank_by_choice, ascending=False)
            if cands.empty:
                # try full budget if necessary
                cands = pool[(pool["pos"] == pos) & (pool["now_cost"] <= budget_available)].sort_values(rank_by_choice, ascending=False)
            if cands.empty:
                ok = False
                break
            pick = cands.iloc[0]
            chosen_ins_local.append(pick.to_dict())
            pool = pool[pool["id"] != pick["id"]]
            budget_remaining -= float(pick["now_cost"])

        if not ok:
            st.warning("No available replacements found for the selected outs under current minutes/budget constraints.")
        else:
            new_ids = base_ids + [int(p["id"]) for p in chosen_ins_local]
            new_squad_df = pred[pred["id"].isin(new_ids)].copy()
            new_xi = build_best_xi_from_squad(new_squad_df, rank_by_choice)
            new_total = float(new_xi[rank_by_choice].sum()) if not new_xi.empty else 0.0
            gain = new_total - current_xi_total
            out_list_names = ", ".join([o.get("web_name", "") for o in outs_rows_sorted])
            in_list_names = ", ".join([i.get("web_name", "") for i in chosen_ins_local])
            st.info(f"Best replacement for **{out_list_names}** ➝ **{in_list_names}** (+{gain:.2f} {rank_by_choice}, new XI total = {new_total:.2f})")
else:
    st.info("Please select exactly 15 players to analyze transfers.")
