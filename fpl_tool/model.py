# model.py
import pandas as pd
import numpy as np
from typing import Tuple


def build_player_master(players: pd.DataFrame, teams: pd.DataFrame, element_types: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the raw players DataFrame with team name and short position label (GKP/DEF/MID/FWD).
    """
    df = players.copy()
    # map team id -> name
    team_map = teams.set_index("id")["name"].to_dict()
    df["team_name"] = df["team"].map(team_map)

    # map element_type id -> short name (singular_name_short)
    pos_map = element_types.set_index("id")["singular_name_short"].to_dict()
    df["pos"] = df["element_type"].map(pos_map)

    # ensure id exists and is numeric
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    df["id"] = df["id"].astype(int)

    return df


def v2_expected_points(players: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Produce a DataFrame with expected-points metrics.

    Important outputs (per-player):
      - xAttack_per90        : xG + xA per 90 (raw from API if present)
      - att_factor           : fixture-based attacking multiplier (avg over horizon)
      - cs_prob              : fixture-based avg clean-sheet probability (per match)
      - xSaves_per_match     : expected save points per match (per-90 * save-point-proxy)
      - xPts_per_match       : expected fantasy points per match (modelled)
      - xPts_total           : expected points across the chosen horizon (xPts_per_match * horizon)
      - xPts_per_m           : value (per match) per million (added later by add_value_columns)
    Notes:
      - This model assumes a *per-match* projection (i.e., if a player plays 90 minutes).
      - Use the UI min-minutes filter to remove players with too small minutes (1-min anomalies).
    """
    df = players.copy()

    # ------------- ensure numeric columns exist -------------
    numeric_cols = [
        "expected_goals_per_90",
        "expected_assists_per_90",
        "saves_per_90",
        "minutes",
        "now_cost",
        "selected_by_percent",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # base attacking metric (per 90)
    df["xAttack_per90"] = df["expected_goals_per_90"] + df["expected_assists_per_90"]

    # compute fixture-based multipliers (att_factor and cs_prob) using upcoming fixtures
    att_factors = []
    cs_probs = []

    # make sure fixture difficulty columns exist; if not, fallback to neutral difficulty = 3
    fixtures = fixtures.copy()
    if "team_h_difficulty" not in fixtures.columns or "team_a_difficulty" not in fixtures.columns:
        fixtures["team_h_difficulty"] = 3
        fixtures["team_a_difficulty"] = 3

    # iterate players to compute average across next `horizon` fixtures
    for _, player in df.iterrows():
        team_id = player.get("team", None)
        if pd.isna(team_id):
            att_factors.append(1.0)
            cs_probs.append(0.25)
            continue

        team_fixt = fixtures[
            (fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)
        ].copy()

        # sort by kickoff_time if present else by index
        if "kickoff_time" in team_fixt.columns:
            team_fixt = team_fixt.sort_values("kickoff_time")
        team_fixt = team_fixt.head(horizon)

        per_fx_att = []
        per_fx_cs = []
        for _, fx in team_fixt.iterrows():
            if fx.get("team_h") == team_id:
                diff = fx.get("team_h_difficulty", 3)
            else:
                diff = fx.get("team_a_difficulty", 3)

            # translate difficulty 1-5 into clean-sheet probability and attack factor:
            # cs_prob: (5 - diff) / 5  (clamped to min 0.05)
            cs_prob = max(0.05, (5.0 - float(diff)) / 5.0)
            per_fx_cs.append(cs_prob)

            # attack factor: slightly >1 for easier fixtures, <1 for harder
            att_factor = 1.0 + (3.0 - float(diff)) * 0.10  # ~0.8 to 1.2 ranges
            per_fx_att.append(att_factor)

        # fallback defaults if no fixtures found
        if per_fx_att:
            att_factors.append(float(np.mean(per_fx_att)))
        else:
            att_factors.append(1.0)

        if per_fx_cs:
            cs_probs.append(float(np.mean(per_fx_cs)))
        else:
            cs_probs.append(0.25)

    df["att_factor"] = att_factors
    df["cs_prob"] = cs_probs

    # saves proxy: expected save-point contribution per match (assuming 1 save = 0.33 pts)
    df["xSaves_per_match"] = df["saves_per_90"] * 0.33  # per 90 -> per match (90)

    # xAttack per match (attack per 90 adjusted by fixture factor)
    df["xAttack_per90_adj"] = df["xAttack_per90"] * df["att_factor"]
    # to keep names simple, also provide xAttack per match as same unit as per-90 but called xAttack_per90_adj

    # --- Position-specific per-match expected points ---
    xPts_per_match = []
    for _, r in df.iterrows():
        pos = r.get("pos", "")
        # appearance points assumed = 2.0 for a full match (starts)
        appearance_pts = 2.0

        if pos in ["FWD", "MID"]:
            xp = r["xAttack_per90_adj"] + appearance_pts
        elif pos == "DEF":
            # defenders: attacking involvement + clean-sheet pts + appearance pts
            xp = r["xAttack_per90_adj"] + (r["cs_prob"] * 4.0) + appearance_pts
        elif pos == "GKP" or pos == "GKP " or pos == "GK" or pos == "GKP":
            xp = (r["cs_prob"] * 4.0) + r["xSaves_per_match"] + appearance_pts
        else:
            # fallback
            xp = r["xAttack_per90_adj"] + appearance_pts

        xPts_per_match.append(float(xp))

    df["xPts_per_match"] = xPts_per_match

    # total over horizon
    df["xPts_total"] = df["xPts_per_match"] * float(horizon)

    # a simple "xPts" alias (older code might expect this)
    df["xPts"] = df["xPts_per_match"]

    # helpful columns: price (tenths to millions), selected_by_percent
    df["price_m"] = df["now_cost"].astype(float) / 10.0
    df["selected_by_percent"] = pd.to_numeric(df.get("selected_by_percent", 0.0), errors="coerce").fillna(0.0)

    # keep expected/derived columns
    keep_cols = [
        "id",
        "web_name",
        "team_name",
        "pos",
        "price_m",
        "now_cost",
        "selected_by_percent",
        "minutes",
        "xAttack_per90",
        "xAttack_per90_adj",
        "att_factor",
        "cs_prob",
        "xSaves_per_match",
        "xPts_per_match",
        "xPts_total",
        "xPts",
    ]
    # ensure we don't drop required columns if they don't exist
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing].copy()


def add_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add value columns (xPts per million and xPts per million per match).
    - xPts_per_m: per-match expected points divided by player price (m)
    - xPts_per_m_total: total (horizon) expected points per million
    """
    out = df.copy()
    # price in millions: prefer 'price_m' then 'now_cost'/10
    if "price_m" in out.columns:
        price = out["price_m"].astype(float).replace(0, np.nan)
    elif "now_cost" in out.columns:
        price = out["now_cost"].astype(float) / 10.0
        price = price.replace(0, np.nan)
    else:
        price = pd.Series(np.nan, index=out.index)

    out["xPts_per_m"] = out["xPts_per_match"].astype(float) / price
    out["xPts_per_m"] = out["xPts_per_m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # horizon total-to-price (if xPts_total exists)
    if "xPts_total" in out.columns:
        out["xPts_total_per_m"] = out["xPts_total"].astype(float) / price
        out["xPts_total_per_m"] = out["xPts_total_per_m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        out["xPts_total_per_m"] = 0.0

    return out
