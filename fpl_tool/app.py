# app.py
import itertools
from typing import List, Tuple

import pandas as pd
import numpy as np
import requests
import streamlit as st

# Import model functions from your model.py
from fpl_tool.model import build_player_master, v2_expected_points

# add_value_columns is optional in model.py — try import, otherwise compute locally
try:
    from fpl_tool.model import add_value_columns  # type: ignore
except Exception:
    add_value_columns = None


# -----------------------
# Data loaders (cached)
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
# Helpers: Formatting
# -----------------------
def format_money_col(df: pd.DataFrame, col_name: str = "now_cost") -> pd.DataFrame:
    out = df.copy()
    if col_name in out.columns:
        out["£m"] = out[col_name] / 10
    else:
        out["£m"] = np.nan
    return out


def format_selected_by(df: pd.DataFrame, col_name: str = "selected_by_percent") -> pd.DataFrame:
    out = df.copy()
    if col_name in out.columns:
        # ensure numeric
        out[col_name] = pd.to_numeric(out[col_name], errors="coerce").fillna(0)
        out["sel_by_%"] = out[col_name].map(lambda x: f"{x:.1f}%")
    elif "sel_by" in out.columns:
        out["sel_by_%"] = out["sel_by"].map(lambda x: f"{float(x):.1f}%")
    else:
        out["sel_by_%"] = ""
    return out


def fmt_df_for_display(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    # Add price column
    if "now_cost" in out.columns and "£m" not in out.columns:
        out["£m"] = out["now_cost"] / 10
    # selection percent
    if "selected_by_percent" in out.columns:
        out["sel_by_%"] = pd.to_numeric(out["selected_by_percent"], errors="coerce").fillna(0).map(
            lambda x: f"{x:.1f}%"
        )
    elif "sel_by" in out.columns:
        out["sel_by_%"] = pd.to_numeric(out["sel_by"], errors="coerce").fillna(0).map(lambda x: f"{x:.1f}%")
    else:
        out["sel_by_%"] = ""
    # xPts per match & total if not present
    if "xPts_total" not in out.columns and "xPts" in out.columns:
        out["xPts_total"] = out["xPts"]
    if "xPts_per_match" not in out.columns:
        out["xPts_per_match"] = out.apply(
            lambda r: (r["xPts_total"] / r["games_proj"]) if ("xPts_total" in r and r.get("games_proj", 0) > 0) else r.get("xPts_total", 0),
            axis=1,
        )
    # Keep desired columns in order, but guard against missing columns
    final_cols = [c for c in cols if c in out.columns]
    return out[final_cols]


# -----------------------
# Best XI by trying formations
# -----------------------
# Common formations to try (DEF, MID, FWD) — GK is always 1
FORMATIONS = [
    (3, 4, 3),
    (4, 4, 2),
    (3, 5, 2),
    (4, 3, 3),
    (5, 3, 2),
    (4, 5, 1),
]


def build_best_xi_from_squad(squad_df: pd.DataFrame, rank_by: str = "xPts_per_match") -> pd.DataFrame:
    """
    Try several formations, pick the XI (11 players) that maximizes sum of `rank_by`.
    squad_df must contain 'pos' column and the `rank_by` column.
    """
    # ensure numeric
    df = squad_df.copy()
    if rank_by not in df.columns:
        df[rank_by] = 0

    best_total = -1e9
    best_xi = pd.DataFrame()
    # GK first
    gk_pool = df[df["pos"].isin(["GKP", "GK"])].sort_values(rank_by, ascending=False)
    if gk_pool.empty:
        # no keeper; return empty
        return best_xi

    for def_c, mid_c, fwd_c in FORMATIONS:
        # select GK
        try:
            sel_gk = gk_pool.head(1)
        except Exception:
            continue
        # select defenders
        defs = df[df["pos"] == "DEF"].sort_values(rank_by, ascending=False).head(def_c)
        mids = df[df["pos"] == "MID"].sort_values(rank_by, ascending=False).head(mid_c)
        fwds = df[df["pos"] == "FWD"].sort_values(rank_by, ascending=False).head(fwd_c)
        combo = pd.concat([sel_gk, defs, mids, fwds])
        # if combo is not 11 (some positions insufficient) skip
        if len(combo) != 1 + def_c + mid_c + fwd_c:
            continue
        total = combo[rank_by].sum()
        if total > best_total:
            best_total = total
            best_xi = combo.copy()
    # Return best_xi sorted by rank_by
    if best_xi.empty:
        # fallback: top 11 by rank_by
        best_xi = df.sort_values(rank_by, ascending=False).head(11)
    return best_xi.sort_values(rank_by, ascending=False)


# -----------------------
# Transfer suggestion logic (greedy replacements)
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
    Generate up to `top_n_suggestions` transfer suggestions.
    Greedy strategy:
      - For each combination of 1..max_outs players from your squad (we examine combinations of outs),
      - remove them (free budget = sum now_cost of outs + bank),
      - for each out (one-by-one) choose the best replacement in candidate pool that matches position,
        respecting that a chosen replacement cannot be reused and must meet minutes & cost <= budget_remaining.
      - After forming new squad, compute best XI (tries formations) and measure total rank_by.
      - Suggest the combinations that yield the highest gain (new_total - current_total).
    """
    results = []
    pred = pred_df.copy()
    # mapping id->row
    pred_indexed = pred.set_index("id", drop=False)

    # compute current best XI for given squad:
    squad_df = pred[pred["id"].isin(squad_ids)].copy()
    current_best_xi = build_best_xi_from_squad(squad_df, rank_by)
    current_total = float(current_best_xi[rank_by].sum()) if not current_best_xi.empty else 0.0

    # Candidate pool constraints (global)
    def candidate_pool_mask(df: pd.DataFrame):
        return (
            (~df["id"].isin(squad_ids))
            & (pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0) >= min_minutes)
        )

    # We'll consider all combinations of 1..max_outs outs from the provided squad_ids
    from itertools import combinations

    squad_list = list(squad_ids)
    combs = []
    for r in range(1, min(max_outs, len(squad_list)) + 1):
        combs.extend(combinations(squad_list, r))

    for out_combo in combs:
        # freed budget in tenths of £m (now_cost uses same scale)
        freed = sum([float(pred_indexed.loc[out_id]["now_cost"]) for out_id in out_combo])
        budget_available = bank * 10 + freed  # both in tenths of million

        # new squad baseline (removed outs)
        base_ids = [pid for pid in squad_list if pid not in out_combo]

        # Build a working candidate pool (exclude mins filter, remove outs and already in base_ids)
        pool = pred[candidate_pool_mask(pred)].copy()
        # ensure same-position replacement: we'll process each out in descending order of lost value
        outs_rows = [pred_indexed.loc[o].to_dict() for o in out_combo]
        # sort outs so the most impactful out is replaced first (by rank_by)
        outs_rows_sorted = sorted(outs_rows, key=lambda r: r.get(rank_by, 0), reverse=True)

        chosen_ins = []
        budget_remaining = budget_available

        for out_row in outs_rows_sorted:
            pos = out_row["pos"]
            # candidates for this out: same position, not already chosen, cost <= budget_remaining
            cands = pool[(pool["pos"] == pos) & (pool["now_cost"] <= budget_remaining)].copy()
            if cands.empty:
                # no candidate for this pos within current budget; try relaxing minutes filter? skip
                # For now skip this out combo
                chosen_ins = None
                break
            # choose best candidate by rank_by
            cands = cands.sort_values(rank_by, ascending=False)
            pick = cands.iloc[0]
            chosen_ins.append(pick.to_dict())
            # remove pick from pool and reduce budget
            pool = pool[pool["id"] != pick["id"]]
            budget_remaining -= float(pick["now_cost"])

        if chosen_ins is None:
            continue

        # Build new full 15-man IDs and dataframe
        new_ids = base_ids + [int(p["id"]) for p in chosen_ins]
        if len(new_ids) != len(squad_list):
            # something off - skip
            continue
        new_squad_df = pred[pred["id"].isin(new_ids)].copy()
        new_best_xi = build_best_xi_from_squad(new_squad_df, rank_by)
        new_total = float(new_best_xi[rank_by].sum()) if not new_best_xi.empty else 0.0
        gain = new_total - current_total

        if gain > 0:
            results.append((gain, [out_row for out_row in outs_rows_sorted], chosen_ins, new_total))

    # sort and return top_n_suggestions
    results_sorted = sorted(results, key=lambda x: x[0], reverse=True)[:top_n_suggestions]
    # return suggestions and also the default best_ins for top suggestion level (for UI convenience)
    return results_sorted, (results_sorted[0][2] if results_sorted else [])


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="FPL Analytics – Smarter Expected Points", layout="wide")
st.title("⚽ FPL Analytics – Smarter Expected Points")
st.caption(
    "Data: Official Fantasy Premier League API. Model uses v2 (minutes + Poisson clean sheets + attacking proxy)."
)

# load data
players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

# build player master (adds team_name and pos)
pm = build_player_master(players.copy(), teams.copy(), element_types.copy())

# Sidebar: model & display controls
st.sidebar.header("Model & display settings")
horizon = st.sidebar.slider("Fixture horizon (matches)", 1, 10, 5)

rank_by_choice = st.sidebar.selectbox("Rank by (per match or total)", ["xPts_per_match", "xPts_total"])

min_minutes_for_leaderboards = st.sidebar.slider("Min historical minutes for leaderboards (0 = no filter)", 0, 1500, 270)
top_n_per_position = st.sidebar.number_input("Top N per position", min_value=1, max_value=20, value=10, step=1)

# Always use v2 logic
pred = v2_expected_points(pm.copy(), fixtures.copy(), teams.copy(), horizon=horizon)

# Ensure we have helpful columns:
# xPts_total (total over horizon) and games_proj (how many matches projected)
if "xPts" in pred.columns and "xPts_total" not in pred.columns:
    pred["xPts_total"] = pred["xPts"]
if "games_proj" not in pred.columns:
    # some model variants may name it differently; fallback:
    pred["games_proj"] = np.maximum(1, (pd.to_numeric(pred.get("minutes", 0), errors="coerce").fillna(0) / 90).clip(upper=horizon))

# xPts_per_match
pred["xPts_per_match"] = pred.apply(
    lambda r: (r["xPts_total"] / r["games_proj"]) if (r.get("games_proj", 0) > 0) else r.get("xPts_total", 0), axis=1
)

# Add value column either via model helper or compute local fallback
if add_value_columns is not None:
    try:
        pred = add_value_columns(pred)
    except Exception:
        pred["xPts_per_m"] = pred["xPts_total"] / (pred["now_cost"] / 10).replace(0, np.nan)
else:
    pred["xPts_per_m"] = pred["xPts_total"] / (pred["now_cost"] / 10).replace(0, np.nan)

# optional selection-by percent column alias
if "selected_by_percent" in pred.columns:
    pred["selected_by_percent"] = pd.to_numeric(pred["selected_by_percent"], errors="coerce").fillna(0)
else:
    # some ingestion earlier used 'sel_by' — attempt to copy if exists
    if "sel_by" in pred.columns:
        pred["selected_by_percent"] = pred["sel_by"]
    else:
        pred["selected_by_percent"] = 0.0

# Format price column (now_cost -> £m) for display uses
pred["£m"] = pred["now_cost"] / 10

# -----------------------
# Leaderboards: top by position
# -----------------------
st.subheader(f"🎯 Captaincy picks (Top by {rank_by_choice} per position)")

pos_map = {"GKP": 3, "DEF": top_n_per_position, "MID": top_n_per_position, "FWD": top_n_per_position}
for pos, n in pos_map.items():
    # apply minutes filter
    df_pos = pred[pred["pos"] == pos].copy()
    if min_minutes_for_leaderboards > 0:
        df_pos = df_pos[pd.to_numeric(df_pos.get("minutes", 0), errors="coerce").fillna(0) >= min_minutes_for_leaderboards]
    df_pos = df_pos.sort_values(rank_by_choice, ascending=False).head(n)
    st.markdown(f"**Top {len(df_pos)} {pos}s by {rank_by_choice}**")
    # display columns vary by pos
    if pos in ["MID", "FWD"]:
        display_cols = ["web_name", "team_name", "pos", "£m", "selected_by_percent", "xAttack" , "att_factor", "xPts_per_match", "xPts_total"]
    elif pos == "DEF":
        display_cols = ["web_name", "team_name", "pos", "£m", "selected_by_percent", "xAttack", "att_factor", "cs_prob", "xPts_per_match", "xPts_total"]
    else:  # GKP
        display_cols = ["web_name", "team_name", "pos", "£m", "selected_by_percent", "cs_prob", "xSaves", "xPts_per_match", "xPts_total"]

    st.dataframe(fmt_df_for_display(df_pos, display_cols).reset_index(drop=True))

# -----------------------
# Value picks (by xPts_per_m)
# -----------------------
st.subheader("💼 Value picks (Top by xPts_per_m per position)")
for pos, n in pos_map.items():
    df_pos = pred[pred["pos"] == pos].copy()
    if min_minutes_for_leaderboards > 0:
        df_pos = df_pos[pd.to_numeric(df_pos.get("minutes", 0), errors="coerce").fillna(0) >= min_minutes_for_leaderboards]
    df_pos = df_pos.sort_values("xPts_per_m", ascending=False).head(n)
    st.markdown(f"**Top {len(df_pos)} {pos}s by xPts_per_m**")
    # reuse columns selection but show xPts_per_m
    if pos in ["MID", "FWD"]:
        display_cols = ["web_name", "team_name", "pos", "£m", "selected_by_percent", "xAttack", "att_factor", "xPts_per_m", "xPts_per_match", "xPts_total"]
    elif pos == "DEF":
        display_cols = ["web_name", "team_name", "pos", "£m", "selected_by_percent", "xAttack", "att_factor", "cs_prob", "xPts_per_m", "xPts_per_match", "xPts_total"]
    else:
        display_cols = ["web_name", "team_name", "pos", "£m", "selected_by_percent", "cs_prob", "xSaves", "xPts_per_m", "xPts_per_match", "xPts_total"]

    st.dataframe(fmt_df_for_display(df_pos, display_cols).reset_index(drop=True))


# -----------------------
# Analyze my squad: pick 15 players and analyze
# -----------------------
st.subheader("🧩 Analyze My 15-man Squad")

# build player options string safely (avoid f-string formatting complexity inside widget)
player_options = {int(r.id): f"{r.web_name} ({r.team_name}, {r.pos}, £{r.now_cost/10:.1f}m, {r.selected_by_percent:.1f}%)" for r in pred.itertuples()}

squad_ids = st.multiselect(
    "Select your 15 players",
    options=list(player_options.keys()),
    format_func=lambda x: player_options[x],
)

bank = st.number_input("Bank (money in the bank, £m)", min_value=0.0, step=0.1)

if len(squad_ids) == 15:
    # squad DF
    squad_df = pred[pred["id"].isin(squad_ids)].copy()
    # Build best XI using selected ranking metric
    best_xi = build_best_xi_from_squad(squad_df, rank_by_choice)
    st.markdown("### ✅ Best XI (sorted by {}):".format(rank_by_choice))
    st.dataframe(fmt_df_for_display(best_xi, ["web_name", "pos", "team_name", "£m", "sel_by_%", "xPts_per_match", "xPts_total"]).reset_index(drop=True))

    # recommended captain / vice by top two xPts_per_match (or chosen metric)
    if not best_xi.empty and len(best_xi) >= 2:
        captain = best_xi.iloc[0]["web_name"]
        vice_captain = best_xi.iloc[1]["web_name"]
        st.success(f"⭐ Recommended Captain: **{captain}** | Vice Captain: **{vice_captain}**")

    # subs (bench) — players in squad but not in best_xi
    subs = squad_df[~squad_df["id"].isin(best_xi["id"])].sort_values(rank_by_choice, ascending=False)
    st.markdown("### 🪑 Subs (bench, sorted by {}):".format(rank_by_choice))
    st.dataframe(fmt_df_for_display(subs, ["web_name", "pos", "team_name", "£m", "sel_by_%", "xPts_per_match", "xPts_total"]).reset_index(drop=True))

    # -----------------------
    # Transfer suggestions
    # -----------------------
    st.markdown("---")
    st.subheader("🔁 Suggested Transfers (single-player + multi-player up to 3 outs supported)")

    # build transfer candidates (only players in squad considered as potential outs)
    current_xi_total = float(best_xi[rank_by_choice].sum()) if not best_xi.empty else 0.0

    # create all greedy suggestions (this can take a little time for big squads & max_outs=3)
    max_outs_allowed = 3
    suggestions, sample_ins = suggest_transfers_greedy(
        pred_df=pred,
        squad_ids=squad_ids,
        bank=bank,
        rank_by=rank_by_choice,
        min_minutes=min_minutes_for_leaderboards,
        max_outs=max_outs_allowed,
        top_n_suggestions=2,
    )

    if suggestions:
        st.markdown("#### 💡 Top transfer suggestions (greedy):")
        for gain, outs, ins, new_total in suggestions:
            out_names = ", ".join([o["web_name"] for o in outs])
            in_names = ", ".join([i["web_name"] for i in ins])
            st.success(f"**{out_names} ➝ {in_names}** (+{gain:.2f} {rank_by_choice}, new XI total = {new_total:.2f})")
    else:
        st.info("No beneficial transfers found within your squad, minutes filter & budget (greedy search).")

    # UI: allow user to pick up to 3 outs (from their squad) to ask for best replacement(s)
    st.markdown("#### 🎯 Choose up to 3 players to transfer OUT (we'll suggest the best replacements):")
    out_choices = st.multiselect(
        "Select players to sell (up to 3)",
        options=squad_ids,
        format_func=lambda x: player_options[x],
        max_selections=3,
    )

    if out_choices:
        # call greedy suggestion but restrict to combos that match the selected outs
        # We simulate removing exactly those outs (if length > 3, UI prevents it)
        outs_tuple = tuple(out_choices)
        # We'll call the greedy routine tailored to that single combination:
        # compute freed budget and then choose greedy replacements
        freed = sum([float(pred.set_index("id").loc[o]["now_cost"]) for o in outs_tuple])
        budget_available = bank * 10 + freed

        # base remain ids
        base_ids = [pid for pid in squad_ids if pid not in outs_tuple]
        # candidate pool (respect min minutes)
        pool = pred[
            (~pred["id"].isin(base_ids))
            & (~pred["id"].isin(outs_tuple))
            & (pd.to_numeric(pred.get("minutes", 0), errors="coerce").fillna(0) >= min_minutes_for_leaderboards)
        ].copy()

        # process outs in order of lost value (rank_by)
        outs_rows = [pred.set_index("id").loc[o].to_dict() for o in outs_tuple]
        outs_rows_sorted = sorted(outs_rows, key=lambda r: r.get(rank_by_choice, 0), reverse=True)

        chosen_ins_local = []
        budget_remaining = budget_available

        for out_row in outs_rows_sorted:
            pos = out_row["pos"]
            cands = pool[(pool["pos"] == pos) & (pool["now_cost"] <= budget_remaining)].sort_values(rank_by_choice, ascending=False)
            if cands.empty:
                cands = pool[(pool["pos"] == pos) & (pool["now_cost"] <= budget_available)].sort_values(rank_by_choice, ascending=False)
            if cands.empty:
                chosen_ins_local = None
                break
            pick = cands.iloc[0]
            chosen_ins_local.append(pick.to_dict())
            pool = pool[pool["id"] != pick["id"]]
            budget_remaining -= float(pick["now_cost"])

        if chosen_ins_local is None:
            st.warning("No available replacements found for the selected outs under current minutes/budget constraints.")
        else:
            new_ids = base_ids + [int(p["id"]) for p in chosen_ins_local]
            new_squad_df = pred[pred["id"].isin(new_ids)].copy()
            new_xi = build_best_xi_from_squad(new_squad_df, rank_by_choice)
            new_total = float(new_xi[rank_by_choice].sum()) if not new_xi.empty else 0.0
            gain = new_total - current_xi_total
            out_list_names = ", ".join([o["web_name"] for o in outs_rows_sorted])
            in_list_names = ", ".join([i["web_name"] for i in chosen_ins_local])
            st.info(f"Best replacement for **{out_list_names}** ➝ **{in_list_names}** (+{gain:.2f} {rank_by_choice}, new XI total = {new_total:.2f})")
else:
    st.info("Please select exactly 15 players to analyze transfers.")
