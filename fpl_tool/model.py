import pandas as pd
import numpy as np


def baseline_expected_points(
    players: pd.DataFrame, fixtures: pd.DataFrame, horizon: int = 3
) -> pd.DataFrame:
    """
    Baseline expected points model:
    - Uses form * (minutes / 90)
    - Scales by fixture horizon
    """
    df = players.copy()

    df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0.0)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)

    df["xPts"] = (df["form"] * (df["minutes"] / 90.0)) * (horizon / 3.0)

    return df


def v2_expected_points(
    players: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame, horizon: int = 3
) -> pd.DataFrame:
    """
    Smarter expected points model (V2):
    - Base xPts from form + normalized minutes
    - Adds clean sheet proxy for DEFs & GKs
    - Adds attacking proxy for MIDs & FWDs
    - Scales across fixture horizon
    """

    df = players.copy()

    # Ensure numeric
    for col in ["form", "minutes", "appearances", "clean_sheets", "goals_scored", "assists"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # Avoid divide by zero
    df["appearances"] = df["appearances"].replace(0, 1)

    # Minutes per match
    df["minutes_per_match"] = df["minutes"] / df["appearances"]

    # Base score
    df["xPts"] = df["form"] * (df["minutes_per_match"] / 90.0)

    # Defensive proxy
    defensive_positions = ["GKP", "DEF"]
    df.loc[df["pos"].isin(defensive_positions), "xPts"] += df["clean_sheets"] * 0.2

    # Attacking proxy
    attacking_positions = ["MID", "FWD"]
    df.loc[df["pos"].isin(attacking_positions), "xPts"] += (
        df["goals_scored"] * 0.3 + df["assists"] * 0.2
    )

    # Scale by fixture horizon
    df["xPts"] = df["xPts"] * horizon

    return df
