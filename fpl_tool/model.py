import numpy as np
import pandas as pd

from fpl_tool.features import (
    simple_expected_goals_horizon,
    status_to_start_prob,
)

# ---------------------------
# Utilities
# ---------------------------
def _safe_norm(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or (mx - mn) < 1e-9:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


# ---------------------------
# V1 baseline (kept for the toggle)
# ---------------------------
def baseline_expected_points(
    players_df: pd.DataFrame,
    events: pd.DataFrame,
    fixtures_softness: dict,
    horizon: int = 1,
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1,
) -> pd.DataFrame]:
    if players_df is None or players_df.empty:
        return pd.DataFrame()

    df = players_df.copy()

    # infer next GW
    try:
        if "is_next" in events.columns and (events["is_next"] == True).any():
            next_gw = int(events.loc[events["is_next"] == True, "id"].iloc[0])
        elif "finished" in events.columns and (~events["finished"]).any():
            next_gw = int(events.loc[~events["finished"], "id"].min())
        else:
            next_gw = int(events["id"].max()) if "id" in events.columns else 1
    except Exception:
        next_gw = 1

    df["form"] = pd.to_numeric(df.get("form", 0), errors="coerce").fillna(0.0)
    df["mins"] = pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0.0)
    df["mins_norm"] = _safe_norm(df["mins"])

    # fixture softness (invert: easier → higher)
    fdr_vals = []
    for _, r in df.iterrows():
        team_id = r.get("team", None)
        series = [fixtures_softness.get(team_id, {}).get(next_gw + i, np.nan) for i in range(horizon)]
        m = float(np.nanmean(series)) if len(series) else np.nan
        fdr_vals.append(m if not np.isnan(m) else np.nan)

    df["fdr_raw"] = pd.Series(fdr_vals, index=df.index).fillna(3.0)
    df["fdr_norm_inv"] = 1.0 - _safe_norm(df["fdr_raw"])

    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    df["pos"] = df.get("pos").fillna(df.get("element_type", 0).map(pos_map))

    df["xPts"] = alpha * df["form"] + beta * df["mins_norm"] + gamma * df["fdr_norm_inv"]
    pos_adj = {"GKP": 0.15, "DEF": 0.10, "MID": 0.05, "FWD": 0.00}
    df["xPts"] = df.apply(lambda r: r["xPts"] + pos_adj.get(str(r["pos"]), 0.0), axis=1)

    df["price"] = pd.to_numeric(df.get("price", df.get("now_cost", 0) / 10.0), errors="coerce").fillna(0.0)
    df["xPts_per_m"] = df["xPts"] / df["price"].replace(0, np.nan)
    return df.sort_values("xPts", ascending=False)


# ---------------------------
# V2 — Horizon-aware version
# ---------------------------
def expected_points_v2(
    players_df: pd.DataFrame,
    teams: pd.DataFrame,
    events: pd.DataFrame,
    fixtures: pd.DataFrame,
    w_attack: float = 0.9,
    w_cs: float = 1.0,
    w_bonus: float = 0.2,
    horizon: int = 1,
    aggregate: str = "average",  # "average" per match OR "total" across horizon
) -> pd.DataFrame:
    """
    V2 xPts with manual fixture horizon:

    • We compute per-team expected-goals FOR/AGAINST averaged across the next `horizon`
      fixtures (handles DGW naturally).  (lam_for_mean, lam_against_mean, n fixtures)

    • Player xPts is then computed with those horizon-mean lambdas.
      If aggregate == "total", we multiply the final xPts by n fixtures for that team.
      If "average", we leave it per-match.

    Typical outputs:
      - top captains ~5–9 (average) for a good single fixture
      - total across 2 fixtures ≈ 2× that number (if minutes are stable)
    """
    if players_df is None or players_df.empty:
        return pd.DataFrame()

    df = players_df.copy()

    # --- team horizon context ---
    ctx = simple_expected_goals_horizon(fixtures, events, horizon=horizon)  # team-> dict with lam_for_mean, lam_against_mean, n

    # --- minutes share baselining ---
    try:
        finished_gws = int(events["finished"].sum()) if "finished" in events.columns else int(events["id"].max() - 1)
    except Exception:
        finished_gws = 1
    finished_gws = max(1, finished_gws)
    denom_mins = finished_gws * 90.0

    # start prob & minutes share
    st_col = df.get("status", "a")
    cnr_col = df.get("chance_of_playing_next_round", np.nan)
    start_probs = [status_to_start_prob(st, cnr) for st, cnr in zip(st_col, cnr_col)]
    df["start_prob"] = start_probs

    df["minutes"] = pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0.0)
    df["mins_ratio"] = np.clip(df["minutes"] / max(1.0, denom_mins), 0.0, 1.0)

    sel = pd.to_numeric(df.get("selected_by_percent", 0), errors="coerce").fillna(0.0) / 100.0
    sel_adj = 0.25 + 0.75 * np.clip(sel, 0.0, 1.0)

    df["p60"] = np.minimum(np.array(df["start_prob"]) * 0.9 * sel_adj, df["mins_ratio"])

    # position
    if "pos" not in df.columns:
        pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        df["pos"] = df.get("element_type", 0).map(pos_map)

    # --- map each player to horizon lambdas ---
    lam_for = []
    lam_against = []
    n_fix = []
    for _, r in df.iterrows():
        c = ctx.get(int(r["team"]), None)
        if c:
            lam_for.append(float(c["lam_for_mean"]))
            lam_against.append(float(c["lam_against_mean"]))
            n_fix.append(int(c["n"]))
        else:
            lam_for.append(1.2)
            lam_against.append(1.2)
            n_fix.append(1)
    df["lam_for"] = lam_for
    df["lam_against"] = lam_against
    df["n_fixtures"] = n_fix

    # --- components ---
    # CS
    df["cs_prob"] = np.exp(-df["lam_against"])
    cs_points_by_pos = df["pos"].map({"GKP": 4.0, "DEF": 4.0}).fillna(0.0)
    df["cs_component"] = w_cs * df["cs_prob"] * cs_points_by_pos * df["p60"]

    # attacking proxy (log scaled)
    goal_pts_map = {"GKP": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
    goal_pts = df["pos"].map(goal_pts_map).fillna(4.0)
    assist_pts = 3.0

    mins = df["minutes"].replace(0, np.nan)
    g_p90 = pd.to_numeric(df.get("goals_scored", 0), errors="coerce") / mins * 90.0
    a_p90 = pd.to_numeric(df.get("assists", 0), errors="coerce") / mins * 90.0
    g_p90 = g_p90.fillna(0.0)
    a_p90 = a_p90.fillna(0.0)

    ict = pd.to_numeric(df.get("ict_index", 0), errors="coerce").fillna(0.0)
    ict_norm = (ict - ict.min()) / (ict.max() - ict.min() + 1e-9)

    atk_raw = (g_p90 * goal_pts) + (a_p90 * assist_pts) + 0.5 * ict_norm
    atk_scaled = np.log1p(np.maximum(atk_raw, 0.0))

    df["atk_component"] = w_attack * df["lam_for"] * df["p60"] * atk_scaled

    # appearance + bonus + GK saves
    df["appear_pts"] = 2.0 * df["start_prob"]
    df["gk_saves_proxy"] = np.where(df["pos"] == "GKP", (1.0 - df["cs_prob"]) * 0.5 * df["p60"], 0.0)

    bps = pd.to_numeric(df.get("bps", 0), errors="coerce").fillna(0.0)
    bps_p90 = (bps / mins * 90.0).fillna(0.0)
    bonus_proxy = (bps_p90 + ict_norm) / 2.0
    bonus_scaled = np.log1p(np.maximum(bonus_proxy, 0.0))
    df["bonus_component"] = w_bonus * bonus_scaled * df["p60"]

    # combine
    df["xPts_v2"] = (
        df["appear_pts"]
        + df["cs_component"]
        + df["gk_saves_proxy"]
        + df["atk_component"]
        + df["bonus_component"]
    )

    # aggregate choice: average per match vs total across horizon
    if aggregate.lower().startswith("total"):
        df["xPts_v2"] = df["xPts_v2"] * df["n_fixtures"]

    # value per million (safe)
    df["price"] = pd.to_numeric(df.get("price", df.get("now_cost", 0) / 10.0), errors="coerce")
    df["price"] = df["price"].where(df["price"] > 0, np.nan)
    df["xPts_v2_per_m"] = (df["xPts_v2"] / df["price"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # light clipping
    cap = 18.0 if aggregate.lower().startswith("average") else 36.0
    df["xPts_v2"] = df["xPts_v2"].clip(lower=0.0, upper=cap)

    return df.sort_values("xPts_v2", ascending=False)
