import pandas as pd
import numpy as np


def baseline_expected_points(players: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Basic model: expected points from form + minutes.
    """
    df = players.copy()
    df["xPts"] = (
        df["form"].astype(float) * 0.6 +
        (df["minutes"] / 90).clip(0, horizon) * 0.4
    )
    return df


def v2_expected_points(players: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Improved model:
    - Expected minutes
    - Fixture difficulty (Poisson proxy)
    - Role weighting by position
    """
    df = players.copy()

    # role multipliers
    role_weight = {"GKP": 0.8, "DEF": 1.0, "MID": 1.2, "FWD": 1.4}
    df["role_weight"] = df["pos"].map(role_weight).fillna(1.0)

    xpts = []
    for _, player in df.iterrows():
        team_id = player["team"]   # FIX: was team_id, now corrected
        # Get upcoming fixtures for this player's team
        team_fixt = fixtures[
            (fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)
        ].head(horizon)

        # Fixture difficulty proxy
        if "team_h_difficulty" in fixtures.columns:
            diffs = np.where(
                team_fixt["team_h"] == team_id,
                team_fixt["team_h_difficulty"],
                team_fixt["team_a_difficulty"]
            )
            fixture_factor = np.clip(5 - diffs.mean(), 1, 5) / 5
        else:
            fixture_factor = 1.0

        minutes_factor = min(player["minutes"] / (90 * horizon), 1.0)

        xp = (
            player["form"].astype(float) * 0.5 +
            fixture_factor * 0.3 +
            minutes_factor * 0.2
        ) * player["role_weight"]

        xpts.append(xp)

    df["xPts"] = xpts
    return df


def add_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add value-for-money metrics (xPts per million).
    """
    df = df.copy()
    df["xPts_per_m"] = df["xPts"] / df["price"].replace(0, np.nan)
    return df
