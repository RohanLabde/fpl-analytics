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
    if teams is not None and "id" in teams.columns and "name" in teams.columns:
        team_map = teams.set_index("id")["name"].to_dict()
        df["team_name"] = df["team"].map(team_map)
    else:
        df["team_name"] = df.get("team", np.nan)

    # map element_type id -> short name (singular_name_short)
    if element_types is not None and "id" in element_types.columns and "singular_name_short" in element_types.columns:
        pos_map = element_types.set_index("id")["singular_name_short"].to_dict()
        df["pos"] = df["element_type"].map(pos_map)
    else:
        df["pos"] = df["element_type"]

    # ensure id exists and is numeric
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    df["id"] = df["id"].astype(int)

    return df


def v2_expected_points(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame = None,
    horizon: int = 5,
    form_weight: float = 0.25,
    bonus_weight: float = 0.05,
) -> pd.DataFrame:
    """
    Produce expected points projections with optional blending of:
      - model signal (xG/xA, saves, clean-sheet prob)  -- internal model
      - recent FPL 'form' (official field if present)
      - historical 'bonus' rate (bonus points per match)

    Parameters:
      players, fixtures, teams : dataframes from the FPL API -> bootstrap-static + fixtures
      horizon : number of upcoming matches to project
      form_weight : weight given to the 'form' signal (per-match points)
      bonus_weight : weight given to bonus-per-match signal

    Behavior:
      - Ensures games_proj is clamped to [1, horizon] to avoid tiny-sample inflation.
      - Keeps original model per-match projection as 'xPts_model_per_match'.
      - Produces blended_xPts_per_match and sets xPts_per_match = blended_xPts_per_match for compatibility.
      - Produces xPts_total (blended over games_proj) and blended_xPts_total.
    """

    df = players.copy()

    # ---------------------
    # ensure numeric columns
    # ---------------------
    numeric_cols = [
        "expected_goals_per_90",
        "expected_assists_per_90",
        "saves_per_90",
        "minutes",
        "now_cost",
        "selected_by_percent",
        "bonus",
        "form",  # official API field 'form' exists in bootstrap-static
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # ---------------------
    # base attacking per-90
    # ---------------------
    df["xAttack_per90"] = df["expected_goals_per_90"] + df["expected_assists_per_90"]

    # ---------------------
    # fixture-based multipliers
    # ---------------------
    att_factors = []
    cs_probs = []

    fixtures = fixtures.copy() if fixtures is not None else pd.DataFrame()
    if fixtures is not None and not fixtures.empty:
        if "team_h_difficulty" not in fixtures.columns or "team_a_difficulty" not in fixtures.columns:
            fixtures["team_h_difficulty"] = 3
            fixtures["team_a_difficulty"] = 3
    else:
        fixtures = pd.DataFrame(columns=["team_h", "team_a", "team_h_difficulty", "team_a_difficulty", "kickoff_time"])

    for _, player in df.iterrows():
        team_id = player.get("team", None)
        if pd.isna(team_id):
            att_factors.append(1.0)
            cs_probs.append(0.25)
            continue

        team_fixt = fixtures[
            (fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)
        ].copy()

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

            cs_prob = max(0.05, (5.0 - float(diff)) / 5.0)
            per_fx_cs.append(cs_prob)

            att_factor = 1.0 + (3.0 - float(diff)) * 0.10
            per_fx_att.append(att_factor)

        att_factors.append(float(np.mean(per_fx_att)) if per_fx_att else 1.0)
        cs_probs.append(float(np.mean(per_fx_cs)) if per_fx_cs else 0.25)

    df["att_factor"] = att_factors
    df["cs_prob"] = cs_probs

    # ---------------------
    # saves proxy & adjusted attack
    # ---------------------
    df["xSaves_per_match"] = df["saves_per_90"] * 0.33
    df["xAttack_per90_adj"] = df["xAttack_per90"] * df["att_factor"]
    df["xAttack"] = df["xAttack_per90_adj"]  # alias for UI compatibility

    # ---------------------
    # games projection (minutes -> games), clamp to [1, horizon]
    # ---------------------
    df["games_proj"] = (pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0.0) / 90.0)
    # clamp lower bound to 1 and upper bound to horizon to avoid tiny-sample inflation
    df["games_proj"] = df["games_proj"].clip(lower=1.0, upper=float(horizon))

    # ---------------------
    # position-specific modelled per-match xPts (original model signal)
    # ---------------------
    xPts_model = []
    for _, r in df.iterrows():
        pos = (r.get("pos") or "").strip()
        appearance_pts = 2.0  # assumed for a start

        if pos in ["FWD", "MID"]:
            xp = r["xAttack_per90_adj"] + appearance_pts
        elif pos == "DEF":
            xp = r["xAttack_per90_adj"] + (r["cs_prob"] * 4.0) + appearance_pts
        elif pos in ["GKP", "GK"]:
            xp = (r["cs_prob"] * 4.0) + r["xSaves_per_match"] + appearance_pts
        else:
            xp = r["xAttack_per90_adj"] + appearance_pts

        xPts_model.append(float(xp))

    df["xPts_model_per_match"] = xPts_model
    # model total across projected games (model signal)
    df["xPts_model_total"] = df["xPts_model_per_match"] * df["games_proj"]

    # ---------------------
    # compute form & bonus signals
    # ---------------------
    # 'form' from API is an average points-per-game over recent period; if not present it's 0
    # 'bonus' in API is cumulative bonus points (season). Convert to bonus_per_match estimate.
    # Use minutes->games to approximate matches played historically for bonus normalization.
    df["form"] = pd.to_numeric(df.get("form", 0.0), errors="coerce").fillna(0.0)

    # Estimate historical matches played: minutes / 90, but at least 1 to avoid division by tiny numbers
    hist_games = (pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0.0) / 90.0).replace(0, np.nan)
    hist_games = hist_games.fillna(1.0)  # avoid division by zero for players with 0 minutes
    df["bonus_per_match"] = pd.to_numeric(df.get("bonus", 0.0), errors="coerce").fillna(0.0) / hist_games

    # ---------------------
    # normalize weights (ensure sums <= 1). If sum>1, scale down proportionally.
    # model_weight will be the remainder assigned to the model signal.
    # ---------------------
    fw = float(form_weight)
    bw = float(bonus_weight)
    if fw < 0:
        fw = 0.0
    if bw < 0:
        bw = 0.0

    total_external = fw + bw
    if total_external >= 1.0:
        # scale external weights to sum to 0.9 to keep model signal some weight (or simply scale to 0.99)
        scale = 0.9 / total_external
        fw *= scale
        bw *= scale

    model_weight = max(0.0, 1.0 - (fw + bw))

    # ---------------------
    # blended per-match & total expected points
    # ---------------------
    # blended = model_weight * model_signal + form_weight * form + bonus_weight * bonus_per_match
    df["blended_xPts_per_match"] = (
        model_weight * df["xPts_model_per_match"]
        + fw * df["form"]
        + bw * df["bonus_per_match"]
    )

    # set the primary xPts_per_match used by the app to the blended value (backwards compatibility)
    df["xPts_per_match"] = df["blended_xPts_per_match"].astype(float)

    # total over horizon (use games_proj)
    df["blended_xPts_total"] = df["blended_xPts_per_match"] * df["games_proj"]
    df["xPts_total"] = df["blended_xPts_total"].astype(float)

    # keep backward compatibility alias
    df["xPts"] = df["xPts_per_match"]

    # ---------------------
    # price & selection
    # ---------------------
    df["price_m"] = pd.to_numeric(df.get("now_cost", 0.0), errors="coerce").fillna(0.0) / 10.0
    df["selected_by_percent"] = pd.to_numeric(df.get("selected_by_percent", 0.0), errors="coerce").fillna(0.0)
    df["now_cost"] = pd.to_numeric(df.get("now_cost", 0.0), errors="coerce").fillna(0.0)

    # ---------------------
    # columns to return (include both model & blended signals)
    # ---------------------
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
        # model signals
        "xPts_model_per_match",
        "xPts_model_total",
        # blended signals (primary)
        "blended_xPts_per_match",
        "blended_xPts_total",
        "xPts_per_match",
        "xPts_total",
        "xPts",
        # additional signals for inspection
        "form",
        "bonus_per_match",
    ]

    existing = [c for c in keep_cols if c in df.columns]
    return df[existing].copy()


def add_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add value-for-money metrics:
      - xPts_per_m: blended per-match expected points per million
      - xPts_total_per_m: blended total expected points per million
    """
    out = df.copy()

    if "price_m" in out.columns:
        price = pd.to_numeric(out["price_m"], errors="coerce").replace(0, np.nan)
    elif "now_cost" in out.columns:
        price = pd.to_numeric(out["now_cost"], errors="coerce") / 10.0
        price = price.replace(0, np.nan)
    else:
        price = pd.Series(np.nan, index=out.index)

    # prefer blended signal if present; fall back to xPts_per_match
    per_match_col = "blended_xPts_per_match" if "blended_xPts_per_match" in out.columns else "xPts_per_match"
    total_col = "blended_xPts_total" if "blended_xPts_total" in out.columns else "xPts_total"

    out["xPts_per_m"] = pd.to_numeric(out.get(per_match_col, 0.0), errors="coerce").astype(float) / price
    out["xPts_per_m"] = out["xPts_per_m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["xPts_total_per_m"] = pd.to_numeric(out.get(total_col, 0.0), errors="coerce").astype(float) / price
    out["xPts_total_per_m"] = out["xPts_total_per_m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out
