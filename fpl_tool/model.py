# model.py
import pandas as pd
import numpy as np
from typing import Tuple

def build_player_master(players: pd.DataFrame, teams: pd.DataFrame, element_types: pd.DataFrame) -> pd.DataFrame:
    """Build enriched player DataFrame with team + position labels."""
    df = players.copy()
    team_map = teams.set_index("id")["name"].to_dict()
    pos_map = element_types.set_index("id")["singular_name_short"].to_dict()

    df["team_name"] = df["team"].map(team_map)
    df["pos"] = df["element_type"].map(pos_map)
    return df


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def v2_expected_points(players: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame, horizon: int = 5,
                       prior_minutes: float = 270.0) -> pd.DataFrame:
    """
    Smarter expected points model with shrinkage for per-90 rates and horizon-aware totals.

    Returns DataFrame with additional columns:
      - xAttack_per90 (shrunk)
      - xSaves_per90 (shrunk)
      - games_proj (min(horizon, minutes/90))
      - xPts_per_match  (expected points next match)
      - xPts_total      (expected points across horizon)
      - cs_prob, att_factor (fixture-based)
    """
    df = players.copy()

    # Normalize / ensure numeric columns exist
    df["minutes"] = _safe_numeric(df.get("minutes", pd.Series(dtype=float)))
    # source per-90 features from API (names may differ across feeds — try common ones)
    # fallback to 0 if not present
    df["expected_goals_per_90"] = _safe_numeric(df.get("expected_goals_per_90", pd.Series(dtype=float)))
    df["expected_assists_per_90"] = _safe_numeric(df.get("expected_assists_per_90", pd.Series(dtype=float)))
    df["saves_per_90"] = _safe_numeric(df.get("saves_per_90", pd.Series(dtype=float)))

    # ===== Shrinkage =====
    # Prior = population mean of each per90 metric
    # prior_minutes is k (equivalent minutes of prior) — default ~270 (3 full matches)
    k = float(prior_minutes)

    # Compute global means (use simple mean; could use median)
    g_xg = df["expected_goals_per_90"].mean() if not df["expected_goals_per_90"].empty else 0.0
    g_xa = df["expected_assists_per_90"].mean() if not df["expected_assists_per_90"].empty else 0.0
    g_saves = df["saves_per_90"].mean() if not df["saves_per_90"].empty else 0.0

    # Avoid division by zero later; create minutes series
    mins = df["minutes"].astype(float)

    # Shrink per-90 using minutes as weight
    df["expected_goals_per_90_shrunk"] = (df["expected_goals_per_90"] * mins + g_xg * k) / (mins + k)
    df["expected_assists_per_90_shrunk"] = (df["expected_assists_per_90"] * mins + g_xa * k) / (mins + k)
    df["saves_per_90_shrunk"] = (df["saves_per_90"] * mins + g_saves * k) / (mins + k)

    # If minutes are zero, the formula returns the global mean (which is desirable).

    # ===== Projected games & minutes over horizon =====
    # Rough proxy: games we expect the player to feature in across the horizon
    # Using minutes / 90 as a proxy for past usage, but cap at horizon
    df["games_proj"] = (mins / 90.0).clip(upper=horizon)
    # if a player hasn't played but the team has N fixtures, we still allow up to horizon for totals
    # but games_proj derived by minutes preserves low-minute players from over-inflation
    df["minutes_proj"] = df["games_proj"] * 90.0

    # ===== Fixture-based adjustments (clean-sheet prob and attack factor) =====
    # We'll compute per-team averages over the next `horizon` fixtures.
    # Precompute team fixture slices for speed.
    cs_probs = []
    att_factors = []

    # convert kickoff_time to sortable if necessary (some fixtures may have strings)
    # don't modify original fixtures; assume columns team_h, team_a, team_h_difficulty, team_a_difficulty exist
    for _, player in df.iterrows():
        team_id = player["team"]
        team_fixt = fixtures[
            (fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)
        ].sort_values("kickoff_time").head(horizon)

        fixture_cs = []
        fixture_att = []
        for _, fx in team_fixt.iterrows():
            if fx["team_h"] == team_id:
                diff = fx.get("team_h_difficulty", 3)
            else:
                diff = fx.get("team_a_difficulty", 3)
            # clean-sheet proxy: lower difficulty → higher CS probability
            cs_prob = max(0.05, (5 - diff) / 5.0)  # baseline 0.05
            fixture_cs.append(cs_prob)
            # attack factor: inverse of difficulty (heuristic)
            att_factor = 1.0 + (3.0 - diff) * 0.1
            fixture_att.append(att_factor)

        avg_cs = float(np.mean(fixture_cs)) if fixture_cs else 0.2
        avg_att = float(np.mean(fixture_att)) if fixture_att else 1.0
        cs_probs.append(avg_cs)
        att_factors.append(avg_att)

    df["cs_prob"] = cs_probs
    df["att_factor"] = att_factors

    # ===== Compute attack & saves projections =====
    # xAttack_per90 uses shrunk per90 rates
    df["xAttack_per90"] = df["expected_goals_per_90_shrunk"] + df["expected_assists_per_90_shrunk"]
    # xAttack_per_match (expected goals+assists in a single match) is approx per90*1.0
    df["xAttack_per_match"] = df["xAttack_per90"]  # per-match baseline

    # Adjust for fixture attack factor when computing totals (multiply per_match by att_factor)
    df["xAttack_adj_per_match"] = df["xAttack_per_match"] * df["att_factor"]

    # xSaves_per_match
    df["xSaves_per90"] = df["saves_per_90_shrunk"]
    df["xSaves_per_match"] = df["xSaves_per90"]  # one match baseline

    # ===== Position-specific expected points per match (shrunk, robust) =====
    # We'll use:
    #  - attackers: direct attacking contributions per match + appearance points (2)
    #  - defenders: attacking contribution per match + clean-sheet expected pts + appearance pts
    #  - keepers: clean-sheet expected pts + saves contribution + appearance pts
    # points mapping:
    #   - goal/assist converted into expected FPL points via xAttack (approx 1 goal~4, assist~3) — but here we assume APIs already give xG/xA in FPL points units? 
    #   For simplicity we convert: 1 xG ≈ 4 FPL points, 1 xA ≈ 3 FPL points (you can tune)
    # We'll apply multipliers to convert xAttack to FPL points per match.
    GOAL_TO_POINTS = 4.0
    ASSIST_TO_POINTS = 3.0
    # approximate conversion for per90 metrics -> xPts contribution per match:
    df["xG_points_per_match"] = df["expected_goals_per_90_shrunk"] * GOAL_TO_POINTS
    df["xA_points_per_match"] = df["expected_assists_per_90_shrunk"] * ASSIST_TO_POINTS
    df["xAttack_points_per_match"] = df["xG_points_per_match"] + df["xA_points_per_match"]

    # saves: assume 1 save -> 1 pt in FPL, xSaves_per_match is per-match expected saves
    df["xSaves_points_per_match"] = df["xSaves_per_match"] * 1.0

    # clean sheet value: defenders & GK get 4 points (approx) per clean sheet; multiply by cs_prob
    CS_POINTS = 4.0

    # appearance points: simple heuristic — 2 points per appearance (starter/appearance) scaled by appearance probability
    # appearance_prob_per_match ≈ games_proj/horizon if horizon>0 else 0 (safe)
    df["appearance_prob"] = 0.0
    if horizon > 0:
        df["appearance_prob"] = (df["games_proj"] / float(horizon)).clip(upper=1.0)
    else:
        df["appearance_prob"] = (df["games_proj"]).clip(upper=1.0)

    # compute per-match expected points
    per_match_pts = []
    total_pts = []

    for _, r in df.iterrows():
        base_attack_pts = r["xAttack_points_per_match"] * r["att_factor"]  # adjust by fixture attack factor
        saves_pts = r["xSaves_points_per_match"]
        cs_pts_per_match = r["cs_prob"] * CS_POINTS

        ap = r["appearance_prob"]
        # position-specific
        if r["pos"] in ["FWD", "MID"]:
            # attackers: attack pts scaled by appearance probability + appearance pts
            xpm = base_attack_pts * ap + (2.0 * ap)
            # total across horizon: use games_proj to scale
            xtot = base_attack_pts * r["games_proj"] + (2.0 * r["games_proj"])
        elif r["pos"] == "DEF":
            # defenders: attack + cs + appearance
            xpm = (base_attack_pts * ap) + (cs_pts_per_match * ap) + (2.0 * ap)
            xtot = (base_attack_pts * r["games_proj"]) + (cs_pts_per_match * r["games_proj"]) + (2.0 * r["games_proj"])
        elif r["pos"] == "GKP":
            # keepers: cs + saves + appearance
            xpm = (cs_pts_per_match * ap) + (saves_pts * ap) + (2.0 * ap)
            xtot = (cs_pts_per_match * r["games_proj"]) + (saves_pts * r["games_proj"]) + (2.0 * r["games_proj"])
        else:
            # fallback: appearance points only
            xpm = 2.0 * ap
            xtot = 2.0 * r["games_proj"]

        per_match_pts.append(float(xpm))
        total_pts.append(float(xtot))

    df["xPts_per_match"] = per_match_pts
    df["xPts_total"] = total_pts

    # convenience: value metrics (per million)
    df["now_cost"] = df.get("now_cost", 0)  # tenths of million
    # avoid div by zero
    df["price_m"] = df["now_cost"].replace({0: np.nan}) / 10.0
    df["xPts_per_m"] = df["xPts_total"] / df["price_m"]
    df["xPts_per_m_match"] = df["xPts_per_match"] / df["price_m"]

    # tidy: fill NaNs where division by zero occurred
    df["xPts_per_m"] = df["xPts_per_m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["xPts_per_m_match"] = df["xPts_per_m_match"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Keep helpful columns
    keep_cols = [
        "id", "web_name", "team", "team_name", "pos", "now_cost", "selected_by_percent",
        "minutes", "games_proj", "minutes_proj",
        "expected_goals_per_90_shrunk", "expected_assists_per_90_shrunk", "xAttack_per90",
        "xAttack_per_match", "att_factor", "cs_prob", "xSaves_per_match",
        "appearance_prob", "xPts_per_match", "xPts_total",
        "xPts_per_m", "xPts_per_m_match"
    ]
    # some columns may not exist in every dataset; include intersection
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df.copy()
