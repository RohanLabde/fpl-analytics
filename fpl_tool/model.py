
import numpy as np
import pandas as pd

def _safe_norm(series):
    s = series.copy().astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx - mn < 1e-9:
        return pd.Series([0.5]*len(s), index=s.index)  # neutral
    return (s - mn) / (mx - mn)

def baseline_expected_points(players_df, events, fixtures_softness, horizon=1, alpha=0.7, beta=0.2, gamma=0.1):
    if players_df is None or players_df.empty:
        return pd.DataFrame()
    df = players_df.copy()

    # Determine next GW
    try:
        next_mask = (events["is_next"] == True) if "is_next" in events.columns else None
        if next_mask is not None and next_mask.any():
            next_gw = int(events.loc[next_mask, "id"].iloc[0])
        else:
            if "finished" in events.columns and (~events["finished"]).any():
                next_gw = int(events.loc[~events["finished"], "id"].min())
            else:
                next_gw = int(events["id"].max()) if "id" in events.columns else 1
    except Exception:
        next_gw = 1

    # Features: form, minutes proxy
    df["form"] = pd.to_numeric(df.get("form", 0), errors="coerce").fillna(0.0)
    df["mins"] = pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0.0)
    df["mins_norm"] = _safe_norm(df["mins"])

    # Fixture softness (invert: softer fixture -> higher score)
    fdr_vals = []
    for _, r in df.iterrows():
        team_id = r.get("team", None)
        series = [fixtures_softness.get(team_id, {}).get(next_gw+i, np.nan) for i in range(horizon)]
        m = float(np.nanmean(series)) if len(series)>0 else np.nan
        fdr_vals.append(m if not np.isnan(m) else np.nan)
    df["fdr_raw"] = pd.Series(fdr_vals, index=df.index).fillna(3.0)
    df["fdr_norm_inv"] = 1 - _safe_norm(df["fdr_raw"])  # higher is easier

    # Position names might be NaN; map element_type as fallback
    pos_map = {1:"GKP", 2:"DEF", 3:"MID", 4:"FWD"}
    df["pos"] = df["pos"].fillna(df["element_type"].map(pos_map))

    # xPts
    df["xPts"] = alpha*df["form"] + beta*df["mins_norm"] + gamma*df["fdr_norm_inv"]

    # modest priors by position
    pos_adj = {"GKP":0.15,"DEF":0.10,"MID":0.05,"FWD":0.00}
    df["xPts"] = df.apply(lambda r: r["xPts"] + pos_adj.get(str(r["pos"]),0.0), axis=1)

    # value
    df["price"] = pd.to_numeric(df.get("price", df.get("now_cost", 0)/10.0), errors="coerce").fillna(0.0)
    df["xPts_per_m"] = df["xPts"] / df["price"].replace(0, np.nan)
    return df.sort_values("xPts", ascending=False)

import numpy as np
import pandas as pd
from fpl_tool.features import simple_expected_goals, status_to_start_prob

def expected_points_v2(players_df, teams, events, fixtures,
                       w_attack=0.9, w_cs=1.0, w_bonus=0.2):
    """
    V2 xPts:
      xMins60 = start_prob * 0.9   (assume starters average ~81 mins; proxy by start prob)
      CS_prob = exp(-lambda_against)  (Poisson, 0 goals conceded)
      Team xG_for ~ simple_expected_goals()

      For each position:
        - GKP:  appearance + CS + tiny saves proxy (1 - CS_prob)*0.5
        - DEF:  appearance + CS + small attacking index
        - MID/FWD: appearance + attacking index scaled by team xG

      Attacking index ~ per90 (goals*goal_points + assists*assist_points) + ICT (scaled)
      Then scaled by team expected goals (lam_for).
    """
    if players_df is None or players_df.empty:
        return pd.DataFrame()

    # --- match context ---
    m = simple_expected_goals(fixtures, events)  # {team: {lam_for, lam_against, is_home, opp}}
    df = players_df.copy()

    # --- minutes / start probability ---
    df["start_prob"] = [status_to_start_prob(st, cnr if "chance_of_playing_next_round" in df.columns else None)
                        for st, cnr in zip(df.get("status","a"), df.get("chance_of_playing_next_round", np.nan))]
    # probability to reach 60+ mins ~ start_prob * 0.9
    df["p60"] = df["start_prob"] * 0.9

    # --- team lambdas for each player ---
    lam_for_list, lam_against_list = [], []
    for _, r in df.iterrows():
        ctx = m.get(int(r["team"]), None)
        lam_for_list.append(ctx["lam_for"] if ctx else 1.2)
        lam_against_list.append(ctx["lam_against"] if ctx else 1.2)
    df["lam_for"] = lam_for_list
    df["lam_against"] = lam_against_list

    # --- clean sheet probability (Poisson P[X=0]) ---
    df["cs_prob"] = np.exp(-df["lam_against"])

    # --- per-90 attacking proxy ---
    # points per goal by position
    goal_pts = df["pos"].map({"GKP":6, "DEF":6, "MID":5, "FWD":4}).fillna(4)
    assist_pts = 3.0
    mins = df.get("minutes", 0).replace(0, np.nan)
    g_p90 = df.get("goals_scored", 0) / mins * 90.0
    a_p90 = df.get("assists", 0) / mins * 90.0
    ict = pd.to_numeric(df.get("ict_index", 0), errors="coerce").fillna(0.0)

    # Normalize ICT to ~0-1
    ict_norm = (ict - ict.min()) / (ict.max() - ict.min() + 1e-9)

    atk_raw = (g_p90 * goal_pts) + (a_p90 * assist_pts) + 0.5 * ict_norm
    # Fill NaNs (players with 0 mins)
    atk_raw = atk_raw.fillna(0.0)

    # Scale attacking by team expected goals and by p60 (minutes)
    df["atk_component"] = w_attack * df["p60"] * df["lam_for"] * atk_raw

    # --- appearance points ---
    df["appear_pts"] = df["p60"] * 2.0  # 2 points if 60+ likely

    # --- clean sheet component (GKP/DEF only) ---
    cs_points_by_pos = df["pos"].map({"GKP":4.0, "DEF":4.0}).fillna(0.0)
    df["cs_component"] = w_cs * df["cs_prob"] * cs_points_by_pos

    # --- simple saves proxy for GK (tiny) ---
    df["gk_saves_proxy"] = np.where(df["pos"]=="GKP", (1.0 - df["cs_prob"]) * 0.5, 0.0)

    # --- bonus proxy from recent BPS per 90 (if available) else ICT ---
    bps = pd.to_numeric(df.get("bps", 0), errors="coerce").fillna(0.0)
    bps_p90 = bps / mins * 90.0
    bps_p90 = bps_p90.fillna(0.0)
    bonus_proxy = (bps_p90 + ict_norm) / 2.0
    df["bonus_component"] = w_bonus * bonus_proxy

    # --- sum up ---
    df["xPts_v2"] = df["appear_pts"] + df["cs_component"] + df["gk_saves_proxy"] + df["atk_component"] + df["bonus_component"]
    df["xPts_v2_per_m"] = df["xPts_v2"] / df["price"].replace(0, np.nan)
    return df.sort_values("xPts_v2", ascending=False)
