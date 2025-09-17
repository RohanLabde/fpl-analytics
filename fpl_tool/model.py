# fpl_tool/model.py
import pandas as pd
import numpy as np

def build_player_master(players: pd.DataFrame, teams: pd.DataFrame, element_types: pd.DataFrame) -> pd.DataFrame:
    """Build enriched player DataFrame with team + position labels."""
    df = players.copy()

    # Map team name and position (short labels: GKP, DEF, MID, FWD)
    team_map = teams.set_index("id")["name"].to_dict()
    pos_map = element_types.set_index("id")["singular_name_short"].to_dict()

    df["team_name"] = df["team"].map(team_map)
    df["pos"] = df["element_type"].map(pos_map)

    return df


def _safe_get_series(df: pd.DataFrame, col: str, default=0.0):
    """Return numeric series if column exists, else a constant series(default)."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    else:
        return pd.Series([default] * len(df), index=df.index)


def v2_expected_points(players: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Robust expected points model with fallbacks.

    Produces:
      - xPts_per_match : expected points in a single full match (avg over next N fixtures)
      - xPts_total     : expected points scaled by player's projected matches (minutes/90 clipped to horizon)
      - xPts           : alias to xPts_total (keeps backward compatibility)
      - att_factor, cs_prob, xSaves (proxies)
    """

    df = players.copy()

    # --- numeric fields (with safe defaults) ---
    minutes = _safe_get_series(df, "minutes", 0.0)
    goals = _safe_get_series(df, "goals_scored", 0.0)
    assists = _safe_get_series(df, "assists", 0.0)
    ict = _safe_get_series(df, "ict_index", 0.0)  # influence/threat proxy if available
    # FPL doesn't expose xG/xA by default — try these column names if present:
    xg90 = _safe_get_series(df, "expected_goals_per_90", 0.0)
    xa90 = _safe_get_series(df, "expected_assists_per_90", 0.0)
    saves90 = _safe_get_series(df, "saves_per_90", 0.0)

    # Projected matches from minutes
    games_proj = (minutes / 90.0).clip(lower=0.0, upper=horizon)

    # If xG/xA exist, use them. Otherwise construct a robust per-90 attacking proxy:
    # fallback_attack_per90 = (goals + assists) per 90 based on minutes, with smoothing + ict scaling
    with np.errstate(divide='ignore', invalid='ignore'):
        observed_rate = (goals + assists) / (minutes / 90.0)
    # replace infinities / NaN with 0
    observed_rate = observed_rate.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Combine: prefer xg90+xa90 if they exist (non-zero), else use smoothed observed_rate scaled by ict
    attack_proxy_per90 = np.where(
        (xg90 + xa90) > 0,
        xg90 + xa90,
        # fallback: use smoothed observed rate (regularize small-minute players)
        (observed_rate * 0.6) + (ict * 0.02)
    )

    # Cap attack proxy to a reasonable maximum (avoid extreme outliers)
    attack_proxy_per90 = np.minimum(attack_proxy_per90, 3.0)  # 3 xG+xA per 90 is extreme upper bound

    # Save proxies into df for debugging / display
    df["xAttack_per90"] = attack_proxy_per90
    df["xSaves_per90"] = saves90

    # prepare fixture columns
    if "kickoff_time" in fixtures.columns:
        try:
            fixtures = fixtures.copy()
            fixtures["kickoff_time"] = pd.to_datetime(fixtures["kickoff_time"], errors="coerce")
        except Exception:
            pass

    xPts_per_match_list = []
    xPts_total_list = []
    att_factor_list = []
    cs_prob_list = []
    xSaves_list = []

    for idx, player in df.iterrows():
        team_id = player.get("team", None)
        # next N fixtures for the team
        team_fixt = fixtures[
            (fixtures.get("team_h") == team_id) | (fixtures.get("team_a") == team_id)
        ].sort_values("kickoff_time").head(horizon)

        per_fixture_pts = []
        per_fixture_att = []
        per_fixture_cs = []
        per_fixture_saves = []

        for _, fx in team_fixt.iterrows():
            # read difficulty (fallback to 3)
            if fx.get("team_h") == team_id:
                diff = fx.get("team_h_difficulty", fx.get("difficulty", 3))
            else:
                diff = fx.get("team_a_difficulty", fx.get("difficulty", 3))

            # clean sheet proxy (simple heuristic; replace with Poisson if you have team xGA)
            cs_prob = max(0.05, (5.0 - float(diff)) / 5.0)

            # attack factor based on difficulty (capped)
            att_factor = 1.0 + (3.0 - float(diff)) * 0.10
            att_factor = float(np.clip(att_factor, 0.6, 1.4))

            # per-match attack and saves
            xAttack_one = float(player["xAttack_per90"]) * att_factor
            xSaves_one = float(player["xSaves_per90"]) * att_factor

            # per-position per-match expected points
            pos = player.get("pos", "")
            if pos in ["FWD", "MID"]:
                pts_one = xAttack_one + 2.0  # include appearance points for full match
            elif pos == "DEF":
                pts_one = xAttack_one + (cs_prob * 4.0) + 2.0
            elif pos == "GKP":
                pts_one = (cs_prob * 4.0) + (xSaves_one * 0.33) + 2.0
            else:
                pts_one = 2.0

            per_fixture_pts.append(pts_one)
            per_fixture_att.append(att_factor)
            per_fixture_cs.append(cs_prob)
            per_fixture_saves.append(xSaves_one)

        # if there are fixtures, average them; otherwise neutral defaults
        if len(per_fixture_pts) > 0:
            x_per_match = float(np.mean(per_fixture_pts))
            avg_att = float(np.mean(per_fixture_att))
            avg_cs = float(np.mean(per_fixture_cs))
            avg_saves = float(np.mean(per_fixture_saves))
        else:
            x_per_match = 0.0
            avg_att = 1.0
            avg_cs = 0.0
            avg_saves = 0.0

        proj_matches = float(games_proj.loc[idx]) if idx in games_proj.index else 0.0
        # total expected across player's projected matches (not simply sum of fixtures)
        x_total_scaled = x_per_match * proj_matches

        xPts_per_match_list.append(x_per_match)
        xPts_total_list.append(x_total_scaled)
        att_factor_list.append(avg_att)
        cs_prob_list.append(avg_cs)
        xSaves_list.append(avg_saves * proj_matches)

    df["xPts_per_match"] = xPts_per_match_list
    df["xPts_total"] = xPts_total_list
    df["att_factor"] = att_factor_list
    df["cs_prob"] = cs_prob_list
    df["xSaves"] = xSaves_list

    # compatibility alias
    df["xPts"] = df["xPts_total"]

    # attach games_proj
    df["games_proj"] = games_proj

    return df


def add_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add value metrics and safe price handling."""
    out = df.copy()
    out["price_m"] = (out.get("now_cost", 0.0) / 10.0).replace(0, np.nan)

    out["xPts_per_m"] = (out.get("xPts", 0.0)).fillna(0.0) / out["price_m"]
    out["xPts_per_m_match"] = (out.get("xPts_per_match", 0.0)).fillna(0.0) / out["price_m"]

    out["xPts_per_m"] = out["xPts_per_m"].fillna(0.0)
    out["xPts_per_m_match"] = out["xPts_per_m_match"].fillna(0.0)

    return out
