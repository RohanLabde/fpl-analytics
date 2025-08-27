import numpy as np
import pandas as pd

# V2 helpers (minutes + Poisson model) come from features.py
from fpl_tool.features import simple_expected_goals, status_to_start_prob


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
# V1 (baseline) — kept for toggle/back-compat
# ---------------------------
def baseline_expected_points(
    players_df: pd.DataFrame,
    events: pd.DataFrame,
    fixtures_softness: dict,
    horizon: int = 1,
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1,
) -> pd.DataFrame:
    """
    Simple blend: form + minutes + inverse fixture 'softness'.
    """
    if players_df is None or players_df.empty:
        return pd.DataFrame()

    df = players_df.copy()

    # Determine next GW
    try:
        if "is_next" in events.columns and (events["is_next"] == True).any():
            next_gw = int(events.loc[events["is_next"] == True, "id"].iloc[0])
        elif "finished" in events.columns and (~events["finished"]).any():
            next_gw = int(events.loc[~events["finished"], "id"].min())
        else:
            next_gw = int(events["id"].max()) if "id" in events.columns else 1
    except Exception:
        next_gw = 1

    # Core features
    df["form"] = pd.to_numeric(df.get("form", 0), errors="coerce").fillna(0.0)
    df["mins"] = pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0.0)
    df["mins_norm"] = _safe_norm(df["mins"])

    # Fixture softness (lower is harder; invert for easier = higher)
    fdr_vals = []
    for _, r in df.iterrows():
        team_id = r.get("team", None)
        series = [fixtures_softness.get(team_id, {}).get(next_gw + i, np.nan) for i in range(horizon)]
        m = float(np.nanmean(series)) if len(series) else np.nan
        fdr_vals.append(m if not np.isnan(m) else np.nan)

    df["fdr_raw"] = pd.Series(fdr_vals, index=df.index).fillna(3.0)
    df["fdr_norm_inv"] = 1.0 - _safe_norm(df["fdr_raw"])

    # Position mapping
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    df["pos"] = df.get("pos").fillna(df.get("element_type", 0).map(pos_map))

    # xPts blend
    df["xPts"] = alpha * df["form"] + beta * df["mins_norm"] + gamma * df["fdr_norm_inv"]

    # Modest priors by position
    pos_adj = {"GKP": 0.15, "DEF": 0.10, "MID": 0.05, "FWD": 0.00}
    df["xPts"] = df.apply(lambda r: r["xPts"] + pos_adj.get(str(r["pos"]), 0.0), axis=1)

    # Value metric
    df["price"] = pd.to_numeric(df.get("price", df.get("now_cost", 0) / 10.0), errors="coerce").fillna(0.0)
    df["xPts_per_m"] = df["xPts"] / df["price"].replace(0, np.nan)

    return df.sort_values("xPts", ascending=False)


# ---------------------------
# V2 — Minutes + Poisson clean sheets + attacking proxy (scaled)
# ---------------------------
def expected_points_v2(
    players_df: pd.DataFrame,
    teams: pd.DataFrame,
    events: pd.DataFrame,
    fixtures: pd.DataFrame,
    w_attack: float = 0.9,
    w_cs: float = 1.0,
    w_bonus: float = 0.2,
) -> pd.DataFrame:
    """
    V2 xPts (scaled to realistic 0–10 per match):
      • Minutes: start probability -> p60 (probability to reach 60’)
      • Team context: expected goals for & against from a tiny Poisson prep
      • Clean sheets (GKP/DEF): P(goals_conceded = 0)
      • Attacking proxy: per-90 goals/assists + ICT, scaled down and by team xG
      • Bonus proxy: small scaled bonus/BPS/ICT signal
    """
    if players_df is None or players_df.empty:
        return pd.DataFrame()

    df = players_df.copy()

    # --- match context per team (expected goals for/against)
    ctx = simple_expected_goals(fixtures, events)  # {team: {lam_for, lam_against, is_home, opp}}

    # --- start probability -> p60 (probability to play 60+)
    start_probs = []
    st_col = df.get("status", "a")
    cnr_col = df.get("chance_of_playing_next_round", np.nan)
    for st, cnr in zip(st_col, cnr_col):
        start_probs.append(status_to_start_prob(st, cnr))
    df["start_prob"] = start_probs
    df["p60"] = df["start_prob"] * 0.9  # starters average ~81 mins → use 0.9 as a simple proxy

    # --- team lambdas for each player ---
    lam_for, lam_against = [], []
    for _, r in df.iterrows():
        c = ctx.get(int(r["team"]), None)
        lam_for.append(c["lam_for"] if c else 1.2)
        lam_against.append(c["lam_against"] if c else 1.2)
    df["lam_for"] = lam_for
    df["lam_against"] = lam_against

    # --- clean sheet probability (Poisson: P(X=0) with X ~ Poisson(lam_against))
    df["cs_prob"] = np.exp(-df["lam_against"])

    # --- per-90 attacking proxy (scaled) ---
    # Map position if needed
    if "pos" not in df.columns:
        pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        df["pos"] = df.get("element_type", 0).map(pos_map)

    goal_pts_map = {"GKP": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
    goal_pts = df["pos"].map(goal_pts_map).fillna(4.0)
    assist_pts = 3.0

    mins = pd.to_numeric(df.get("minutes", 0), errors="coerce").replace(0, np.nan)
    g_p90 = pd.to_numeric(df.get("goals_scored", 0), errors="coerce") / mins * 90.0
    a_p90 = pd.to_numeric(df.get("assists", 0), errors="coerce") / mins * 90.0
    g_p90 = g_p90.fillna(0.0)
    a_p90 = a_p90.fillna(0.0)

    ict = pd.to_numeric(df.get("ict_index", 0), errors="coerce").fillna(0.0)
    ict_norm = (ict - ict.min()) / (ict.max() - ict.min() + 1e-9)

    # raw attacking points per 90 (unscaled)
    atk_raw = (g_p90 * goal_pts) + (a_p90 * assist_pts) + 0.5 * ict_norm

    # ---- scaling to keep xPts realistic ----
    # This keeps typical attackers around ~0.2–0.6 attacking points per 90 before minutes/team scaling.
    atk_scaled = atk_raw / 10.0

    # combine: minutes (p60) × team xG × scaled per90
    df["atk_component"] = w_attack * df["p60"] * df["lam_for"] * atk_scaled

    # --- appearance points (2 if 60+ very likely)
    df["appear_pts"] = df["p60"] * 2.0

    # --- clean sheet component (GKP/DEF only)
    cs_points_by_pos = df["pos"].map({"GKP": 4.0, "DEF": 4.0}).fillna(0.0)
    df["cs_component"] = w_cs * df["cs_prob"] * cs_points_by_pos

    # --- tiny saves proxy for GK (more saves when CS_prob is low)
    df["gk_saves_proxy"] = np.where(df["pos"] == "GKP", (1.0 - df["cs_prob"]) * 0.5, 0.0)

    # --- bonus proxy (scaled down)
    bps = pd.to_numeric(df.get("bps", 0), errors="coerce").fillna(0.0)
    bps_p90 = (bps / mins * 90.0).fillna(0.0)
    bonus_proxy = (bps_p90 + ict_norm) / 2.0
    df["bonus_component"] = w_bonus * (bonus_proxy / 10.0)

    # --- total xPts (V2) ---
    df["xPts_v2"] = (
        df["appear_pts"]
        + df["cs_component"]
        + df["gk_saves_proxy"]
        + df["atk_component"]
        + df["bonus_component"]
    )

    # Value metric
    df["price"] = pd.to_numeric(df.get("price", df.get("now_cost", 0) / 10.0), errors="coerce").fillna(0.0)
    df["xPts_v2_per_m"] = df["xPts_v2"] / df["price"].replace(0, np.nan)

    # Sanity clip (optional): cap at 0–20 to avoid junk outliers
    df["xPts_v2"] = df["xPts_v2"].clip(lower=0.0, upper=20.0)

    return df.sort_values("xPts_v2", ascending=False)
