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
    Smarter expected points model using FPL API advanced stats:
    - FWD & MID: attacking returns (xG + xA)
    - DEF: attacking + clean sheet probability
    - GKP: clean sheet + saves
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

    # Project minutes (average per match * horizon)
    df["minutes_proj"] = (df["minutes"] / df["minutes"].clip(lower=1).count()) * horizon
    df["games_proj"] = df["minutes_proj"] / 90

    # --- Attacking returns ---
    df["xAttack"] = (df["expected_goals_per_90"] + df["expected_assists_per_90"]) * df["games_proj"]

    # --- Clean sheet proxy ---
    # Approx: based on opponent difficulty (FDR scale 1–5 → convert to CS chance)
    # If avg opponent difficulty ≈ 2 → higher CS chance, if ≈ 5 → low CS chance
    team_fixtures = fixtures.groupby("team_h")[["team_h_difficulty"]].mean().to_dict()["team_h_difficulty"]
    df["avg_fixture_difficulty"] = df["team"].map(team_fixtures).fillna(3)
    df["cs_prob"] = np.clip((5 - df["avg_fixture_difficulty"]) / 5, 0.05, 0.7)  # between 5%–70%

    # --- Saves proxy for GKs ---
    df["xSaves"] = df["saves_per_90"] * df["games_proj"] * 0.33  # 1 save point every 3 saves

    # --- Position-specific xPts ---
    xpts = []
    for _, row in df.iterrows():
        if row["pos"] in ["FWD", "MID"]:
            xp = row["xAttack"] + (row["games_proj"] * 2)  # 2 pts for appearance
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
    """
    Add value-for-money metrics.
    """
    df = df.copy()
    df["xPts_per_m"] = df["xPts"] / (df["now_cost"] / 10)
    return df
