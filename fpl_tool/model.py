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


def v2_expected_points(players: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Smarter expected points model using FPL API advanced stats + fixture horizon.

    Returns DataFrame with:
    - xPts_per_match : expected points in a single full match (averaged across next horizon fixtures)
    - xPts_total     : expected total across the horizon, scaled by games_proj (minutes/90 clipped to horizon)
    - xPts          : alias for xPts_total (keeps compatibility)
    - att_factor, cs_prob, xSaves (proxies)
    """

    df = players.copy()

    # Ensure numeric for advanced stats (these keys come from bootstrap-static where available)
    numeric_cols = [
        "minutes", "expected_goals_per_90", "expected_assists_per_90",
        "saves_per_90"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # Projected matches from minutes (rough proxy)
    df["games_proj"] = (df["minutes"] / 90.0).clip(upper=horizon)
    df["minutes_proj"] = df["games_proj"] * 90.0

    # per-90 attacking rates
    df["xAttack_per90"] = df["expected_goals_per_90"] + df["expected_assists_per_90"]
    df["xSaves_per90"] = df["saves_per_90"]

    # containers to fill
    xpts_per_match_list = []
    xpts_total_list = []
    att_factor_list = []
    cs_prob_list = []
    xSaves_list = []

    # Make sure fixture times sorted; if kickoff_time is string ensure pandas datetime for sorting
    if "kickoff_time" in fixtures.columns:
        try:
            fixtures = fixtures.copy()
            fixtures["kickoff_time"] = pd.to_datetime(fixtures["kickoff_time"], errors="coerce")
        except Exception:
            pass

    for _, player in df.iterrows():
        team_id = player.get("team", None)
        # select next `horizon` fixtures for this team
        team_fixt = fixtures[
            (fixtures.get("team_h") == team_id) | (fixtures.get("team_a") == team_id)
        ].sort_values("kickoff_time").head(horizon)

        per_fixture_points = []
        per_fixture_att = []
        per_fixture_cs = []
        per_fixture_saves = []

        for _, fx in team_fixt.iterrows():
            # Get difficulty for the team's opponent side; fallback to generic 'difficulty' if provided
            if fx.get("team_h") == team_id:
                diff = fx.get("team_h_difficulty", fx.get("difficulty", 3))
                home = True
            else:
                diff = fx.get("team_a_difficulty", fx.get("difficulty", 3))
                home = False

            # Simple clean sheet probability proxy (0.05 minimum)
            # (Replace with Poisson-based CS later if you have team expected goals data)
            cs_prob = max(0.05, (5.0 - float(diff)) / 5.0)

            # Attack factor: easier opponents increase attacking returns
            att_factor = 1.0 + (3.0 - float(diff)) * 0.10

            # Attacking contribution in one full match (if the player plays 90')
            xAttack_one = player["xAttack_per90"] * att_factor

            # Saves contribution in one match (GK)
            xSaves_one = player["xSaves_per90"] * att_factor

            # Per-fixture expected points (per full match) by position
            if player.get("pos") in ["FWD", "MID"]:
                pts_one = xAttack_one + 2.0  # include appearance points for a full match
            elif player.get("pos") == "DEF":
                pts_one = xAttack_one + (cs_prob * 4.0) + 2.0
            elif player.get("pos") == "GKP":
                # 1 save point per 3 saves -> scale by 0.33
                pts_one = (cs_prob * 4.0) + (xSaves_one * 0.33) + 2.0
            else:
                pts_one = 2.0

            per_fixture_points.append(float(pts_one))
            per_fixture_att.append(float(att_factor))
            per_fixture_cs.append(float(cs_prob))
            per_fixture_saves.append(float(xSaves_one))

        # if no upcoming fixtures, fall back to neutral values
        if per_fixture_points:
            x_per_match = float(np.mean(per_fixture_points))
            # total if play every upcoming fixture (not scaled by games_proj)
            x_total_if_play_all_fixt = float(np.sum(per_fixture_points))
            avg_att = float(np.mean(per_fixture_att))
            avg_cs = float(np.mean(per_fixture_cs))
            avg_saves = float(np.mean(per_fixture_saves))
        else:
            x_per_match = 0.0
            x_total_if_play_all_fixt = 0.0
            avg_att = 1.0
            avg_cs = 0.0
            avg_saves = 0.0

        # scale to player's projected matches (games_proj is minutes/90 clipped to horizon)
        proj_matches = float(player["games_proj"])
        x_total_scaled = x_per_match * proj_matches

        xpts_per_match_list.append(x_per_match)
        xpts_total_list.append(x_total_scaled)
        att_factor_list.append(avg_att)
        cs_prob_list.append(avg_cs)
        xSaves_list.append(avg_saves * proj_matches)  # saves scaled for projected matches

    df["xPts_per_match"] = xpts_per_match_list
    df["xPts_total"] = xpts_total_list
    df["att_factor"] = att_factor_list
    df["cs_prob"] = cs_prob_list
    df["xSaves"] = xSaves_list

    # For compatibility: xPts remains the same as total across projected matches
    df["xPts"] = df["xPts_total"]

    return df


def add_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add value-for-money metrics (per horizon and per match)."""
    out = df.copy()
    # make safe denominator for now_cost (now_cost is in tenths of millions in FPL API)
    out["price_m"] = (out["now_cost"] / 10.0).replace(0, np.nan)

    # value based on total across projected matches
    out["xPts_per_m"] = out["xPts"].fillna(0.0) / out["price_m"]
    # value based on per-match expected points
    out["xPts_per_m_match"] = out["xPts_per_match"].fillna(0.0) / out["price_m"]

    # restore zeros instead of NaN price results
    out["xPts_per_m"] = out["xPts_per_m"].fillna(0.0)
    out["xPts_per_m_match"] = out["xPts_per_m_match"].fillna(0.0)

    return out
