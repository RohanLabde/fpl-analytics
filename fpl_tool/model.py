# model.py
import pandas as pd
import numpy as np
from typing import Tuple


def build_player_master(players: pd.DataFrame, teams: pd.DataFrame, element_types: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the raw players DataFrame with team name and short position label (GKP/DEF/MID/FWD).
    Ensures `id` is present and numeric.
    """
    df = players.copy()

    # map team id -> name (if teams provided)
    if "id" in teams.columns and "name" in teams.columns:
        team_map = teams.set_index("id")["name"].to_dict()
        df["team_name"] = df["team"].map(team_map)
    else:
        df["team_name"] = df.get("team", np.nan)

    # map element_type id -> short name (singular_name_short)
    if "id" in element_types.columns and "singular_name_short" in element_types.columns:
        pos_map = element_types.set_index("id")["singular_name_short"].to_dict()
        df["pos"] = df["element_type"].map(pos_map)
    else:
        # fallback: if 'element_type' already textual
        df["pos"] = df["element_type"]

    # ensure id exists and is numeric
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    df["id"] = df["id"].astype(int)

    return df


def v2_expected_points(players: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame = None, horizon: int = 5) -> pd.DataFrame:
    """
    Produce expected points projections.

    Outputs include:
      - xAttack_per90            : xG + xA per 90 (raw if present)
      - xAttack_per90_adj       : xAttack_per90 adjusted by fixture attack factor
      - xAttack                 : alias for xAttack_per90_adj (convenience for UI)
      - att_factor              : average attacking multiplier over the horizon
      - cs_prob                 : average clean-sheet probability per match over horizon
      - xSaves_per_match        : expected save points per match (GK)
      - games_proj              : projected games (minutes/90) clamped to [1, horizon]
      - xPts_per_match          : expected fantasy points per match (modelled)
      - xPts_total              : expected points across the chosen horizon (xPts_per_match * games_proj)
      - price_m                 : price in millions
      - selected_by_percent     : numeric selection percentage
    """
    df = players.copy()

    # --- ensure numeric columns exist ---
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

    # base attacking per-90
    df["xAttack_per90"] = df["expected_goals_per_90"] + df["expected_assists_per_90"]

    # --- fixture-based multipliers (att_factor, cs_prob) ---
    att_factors = []
    cs_probs = []

    fixtures = fixtures.copy() if fixtures is not None else pd.DataFrame()
    # fallback: neutral difficulties if not present
    if fixtures is not None and not fixtures.empty:
        if "team_h_difficulty" not in fixtures.columns or "team_a_difficulty" not in fixtures.columns:
            fixtures["team_h_difficulty"] = 3
            fixtures["team_a_difficulty"] = 3
    else:
        fixtures = pd.DataFrame(columns=["team_h", "team_a", "team_h_difficulty", "team_a_difficulty", "kickoff_time"])

    # compute per-player average attack factor and cs prob across next `horizon` fixtures
    for _, player in df.iterrows():
        team_id = player.get("team", None)
        if pd.isna(team_id):
            att_factors.append(1.0)
            cs_probs.append(0.25)
            continue

        # select fixtures for that team (home or away)
        team_fixt = fixtures[
            (fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)
        ].copy()

        # sort by kickoff_time if available
        if "kickoff_time" in team_fixt.columns:
            team_fixt = team_fixt.sort_values("kickoff_time")

        team_fixt = team_fixt.head(horizon)

        per_fx_att = []
        per_fx_cs = []
        for _, fx in team_fixt.iterrows():
            # determine which difficulty field to use
            if fx.get("team_h") == team_id:
                diff = fx.get("team_h_difficulty", 3)
            else:
                diff = fx.get("team_a_difficulty", 3)

            # clean-sheet probability proxy: easier difficulty -> higher cs chance
            cs_prob = max(0.05, (5.0 - float(diff)) / 5.0)
            per_fx_cs.append(cs_prob)

            # attack factor: >1 for easier, <1 for harder (small range)
            att_factor = 1.0 + (3.0 - float(diff)) * 0.10  # approx range 0.8..1.2
            per_fx_att.append(att_factor)

        att_factors.append(float(np.mean(per_fx_att)) if per_fx_att else 1.0)
        cs_probs.append(float(np.mean(per_fx_cs)) if per_fx_cs else 0.25)

    df["att_factor"] = att_factors
    df["cs_prob"] = cs_probs

    # saves proxy: convert saves_per_90 to expected save points per match (1 save ≈ 0.33 pts)
    df["xSaves_per_match"] = df["saves_per_90"] * 0.33

    # adjusted xAttack per90 after fixture factor
    df["xAttack_per90_adj"] = df["xAttack_per90"] * df["att_factor"]

    # convenience alias used by older app code
    df["xAttack"] = df["xAttack_per90_adj"]

    # --- games projection (minutes -> games) ---
    # Use minutes/90 but clamp to minimum 1 (so tiny minute-sample players are not overly ranked)
    # and maximum horizon (we only evaluate the upcoming horizon)
    df["games_proj"] = (pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0.0) / 90.0).clip(lower=1.0, upper=float(horizon))

    # --- Position-specific expected points per match ---
    xPts_per_match = []
    for _, r in df.iterrows():
        pos = (r.get("pos") or "").strip()
        appearance_pts = 2.0  # approximate points for a (full) appearance/start

        if pos in ["FWD", "MID"]:
            xp = r["xAttack_per90_adj"] + appearance_pts
        elif pos == "DEF":
            xp = r["xAttack_per90_adj"] + (r["cs_prob"] * 4.0) + appearance_pts
        elif pos in ["GKP", "GK"]:
            xp = (r["cs_prob"] * 4.0) + r["xSaves_per_match"] + appearance_pts
        else:
            # fallback: treat as midfielder-ish
            xp = r["xAttack_per90_adj"] + appearance_pts

        xPts_per_match.append(float(xp))

    df["xPts_per_match"] = xPts_per_match

    # total across horizon: treat xPts_total as expected across projected matches (games_proj)
    df["xPts_total"] = df["xPts_per_match"] * df["games_proj"]

    # alias for backward compatibility
    df["xPts"] = df["xPts_per_match"]

    # price and selection columns
    df["price_m"] = pd.to_numeric(df.get("now_cost", 0), errors="coerce").fillna(0.0) / 10.0
    df["selected_by_percent"] = pd.to_numeric(df.get("selected_by_percent", 0.0), errors="coerce").fillna(0.0)

    # make sure now_cost exists as numeric
    df["now_cost"] = pd.to_numeric(df.get("now_cost", 0), errors="coerce").fillna(0.0)

    # --- choose columns to return (keep expected fields used by app) ---
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
        "xAttack",
        "att_factor",
        "cs_prob",
        "xSaves_per_match",
        "games_proj",
        "xPts_per_match",
        "xPts_total",
        "xPts",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing].copy()


def add_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add value-for-money metrics:
      - xPts_per_m: per-match expected points per million
      - xPts_total_per_m: total expected points over horizon per million
    """
    out = df.copy()

    # determine price in millions (use price_m first, fallback to now_cost/10)
    if "price_m" in out.columns:
        price = pd.to_numeric(out["price_m"], errors="coerce").replace(0, np.nan)
    elif "now_cost" in out.columns:
        price = pd.to_numeric(out["now_cost"], errors="coerce") / 10.0
        price = price.replace(0, np.nan)
    else:
        price = pd.Series(np.nan, index=out.index)

    # per-match value
    out["xPts_per_m"] = out.get("xPts_per_match", 0.0).astype(float) / price
    out["xPts_per_m"] = out["xPts_per_m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # total (horizon) value
    if "xPts_total" in out.columns:
        out["xPts_total_per_m"] = out["xPts_total"].astype(float) / price
        out["xPts_total_per_m"] = out["xPts_total_per_m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        out["xPts_total_per_m"] = 0.0

    return out
