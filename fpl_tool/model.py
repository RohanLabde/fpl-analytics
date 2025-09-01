import pandas as pd
import numpy as np


def baseline_expected_points(pm: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """
    Baseline expected points calculation.
    Just uses form, minutes, and fixture softness as per V1.
    """
    df = pm.copy()
    df["xPts"] = (
        df["form"] * 0.7
        + df["xMins"] * 0.2
        + (1.0 - df["fixture_softness"]) * 0.1
    ) * horizon
    return df


def v2_expected_points(pm: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """
    Smarter expected points calculation (V2).
    - Considers minutes (xMins)
    - Adjusts by clean sheet probability using a Poisson proxy
    - Includes fixture horizon dynamically
    """

    preds = []

    # Detect correct fixture column names
    if "team_h" in fixtures.columns and "team_a" in fixtures.columns:
        home_col, away_col = "team_h", "team_a"
    elif "team_home" in fixtures.columns and "team_away" in fixtures.columns:
        home_col, away_col = "team_home", "team_away"
    else:
        raise KeyError(f"Fixtures missing expected team columns. Found: {fixtures.columns}")

    for _, player in pm.iterrows():
        team_id = player["team_id"]

        # Get fixtures for this team in horizon
        team_fixt = fixtures[
            (fixtures[home_col] == team_id) | (fixtures[away_col] == team_id)
        ].head(horizon)

        if team_fixt.empty:
            xp = 0.0
        else:
            # Minutes proxy
            mins_factor = player.get("xMins", 60) / 90.0

            # Base form proxy
            form_factor = float(player.get("form", 0))

            # Fixture softness adjustment (if available)
            if "fixture_softness" in pm.columns:
                soft_factor = 1.0 - float(player["fixture_softness"])
            else:
                soft_factor = 1.0

            # Poisson clean sheet proxy: assume weaker teams concede ~2 goals, strong ~0.8
            avg_difficulty = team_fixt.get("difficulty", pd.Series([3])).mean()
            cs_prob = max(0.05, 1.2 - (avg_difficulty * 0.2))  # keep between 0.05–1.0

            # Expected points calculation
            xp = (form_factor * 0.6 + mins_factor * 0.3 + cs_prob * 0.1) * len(team_fixt)

        preds.append({"id": player["id"], "xPts": round(xp, 3)})

    return pm.merge(pd.DataFrame(preds), on="id")


def add_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds value metrics like xPts per million.
    """
    df = df.copy()
    df["xPts_per_m"] = df["xPts"] / df["price"]
    return df
