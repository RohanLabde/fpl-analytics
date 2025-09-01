import pandas as pd
import numpy as np


def baseline_expected_points(pm: pd.DataFrame, teams: pd.DataFrame, soft: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline expected points model:
    Combines form, minutes, and fixture softness into xPts.
    """
    df = pm.copy()

    # Safe fill to avoid NaNs messing things up
    df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)

    # Normalize
    df["form_norm"] = df["form"] / df["form"].max() if df["form"].max() > 0 else 0
    df["minutes_norm"] = df["minutes"] / df["minutes"].max() if df["minutes"].max() > 0 else 0

    # Fixture softness match (defense proxy)
    df = df.merge(soft, on="team", how="left")
    df["soft_norm"] = df["softness"] / df["softness"].max() if df["softness"].max() > 0 else 0

    # Weighted sum
    df["xPts"] = (
        0.7 * df["form_norm"] +
        0.2 * df["minutes_norm"] +
        0.1 * df["soft_norm"]
    ) * 10  # scale up for interpretability

    # Always compute value metric
    df["xPts_per_m"] = df["xPts"] / df["now_cost"].replace(0, np.nan)

    return df


def v2_expected_points(pm: pd.DataFrame, teams: pd.DataFrame, fixtures: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """
    V2 expected points model:
    Uses fixture horizon, Poisson clean sheets, attacking proxy.
    """
    df = pm.copy()

    # Minutes weight
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
    df["minutes_norm"] = df["minutes"] / df["minutes"].max() if df["minutes"].max() > 0 else 0

    # Attacking proxy: goals + assists per 90
    df["attacking_proxy"] = (
        (df["goals_scored"] + df["assists"]) / df["minutes"].replace(0, np.nan) * 90
    ).fillna(0)
    df["attacking_norm"] = df["attacking_proxy"] / df["attacking_proxy"].max() if df["attacking_proxy"].max() > 0 else 0

    # Fixture horizon softness: average opponent difficulty over N matches
    team_softness = []
    for team_id in df["team"].unique():
        team_fixt = fixtures[(fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)].head(horizon)
        softness = team_fixt["difficulty"].mean() if not team_fixt.empty else np.nan
        team_softness.append({"team": team_id, "softness": softness})
    team_softness = pd.DataFrame(team_softness)

    df = df.merge(team_softness, on="team", how="left")
    df["soft_norm"] = df["softness"] / df["softness"].max() if df["softness"].max() > 0 else 0

    # Weighted sum
    df["xPts"] = (
        0.5 * df["minutes_norm"] +
        0.3 * df["attacking_norm"] +
        0.2 * (1 - df["soft_norm"])   # easier fixtures = higher points
    ) * 10

    # Always compute value metric
    df["xPts_per_m"] = df["xPts"] / df["now_cost"].replace(0, np.nan)

    return df
