import pandas as pd
import numpy as np


def build_player_master(players, teams, element_types):
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
    Smarter expected points model using FPL API advanced stats + fixture horizon:
    - FWD & MID: attacking returns (xG + xA) + appearance points
    - DEF: attacking returns + clean sheet probability (fixture horizon) + appearance points
    - GKP: clean sheet probability (fixture horizon) + saves + appearance points
    """

    df = players.copy()

    # Ensure numeric for advanced stats
    numeric_cols = [
        "minutes", "expected_goals_per_90", "expected_assists_per_90",
        "expected_goal_involvements_per_90", "saves_per_90"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # Projected games from minutes (rough proxy)
    df["games_proj"] = (df["minutes"] / 90).clip(upper=horizon)
    df["minutes_proj"] = df["games_proj"] * 90

    # --- Attacking returns (xG + xA scaled by projected games) ---
    df["xAttack"] = (df["expected_goals_per_90"] + df["expected_assists_per_90"]) * df["games_proj"]

    # --- Fixture horizon clean sheet probability ---
    cs_probs = []
    for _, player in df.iterrows():
        team_id = player["team"]

        # Next N fixtures where this team plays
        team_fixt = fixtures[
            (fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)
        ].sort_values("kickoff_time").head(horizon)

        fixture_cs = []
        for _, fx in team_fixt.iterrows():
            if fx["team_h"] == team_id:
                diff = fx["team_h_difficulty"]
            else:
                diff = fx["team_a_difficulty"]

            # Convert FDR difficulty (1–5) into CS probability
            cs_prob = max(0.05, (5 - diff) / 5)  # e.g. diff=2 → 0.6, diff=4 → 0.2
            fixture_cs.append(cs_prob)

        avg_cs = np.mean(fixture_cs) if fixture_cs else 0.2
        cs_probs.append(avg_cs)

    df["cs_prob"] = cs_probs

    # --- Saves proxy for GKs ---
    df["xSaves"] = df["saves_per_90"] * df["games_proj"] * 0.33  # 1 save point per 3 saves

    # --- Position-specific xPts ---
    xpts = []
    for _, row in df.iterrows():
        if row["pos"] in ["FWD", "MID"]:
            xp = row["xAttack"] + (row["games_proj"] * 2)  # 2 pts appearance
        elif row["pos"] == "DEF":
            xp = row["xAttack"] + (row["cs_prob"] * 4 * row["games_proj"]) + (row["games_proj"] * 2)
        elif row["pos"] == "GKP":
            xp = (row["cs_prob"] * 4 * row["games_proj"]) + row["xSaves"] + (row["games_proj"] * 2)
        else:
            xp = row["games_proj"] * 2
        xpts.append(xp)

    df["xPts"] = xpts

    return df


def add_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add value-for-money metrics."""
    df = df.copy()
    df["xPts_per_m"] = df["xPts"] / (df["now_cost"] / 10)
    return df
