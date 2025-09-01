import pandas as pd
import numpy as np


def build_player_master(players, teams, element_types):
    """Build player master DataFrame with relevant fields."""
    df = players.copy()

    # Map team and position
    team_map = teams.set_index("id")["name"].to_dict()
    pos_map = element_types.set_index("id")["singular_name_short"].to_dict()

    df["team_name"] = df["team"].map(team_map)
    df["pos"] = df["element_type"].map(pos_map)

    # Keep only relevant columns
    keep_cols = [
        "id",
        "web_name",
        "team",
        "team_name",
        "pos",
        "now_cost",
        "minutes",
        "form",              # ✅ ensure form is kept
        "points_per_game",
        "ep_next",
        "ep_this"
    ]
    return df[keep_cols]


def baseline_expected_points(players: pd.DataFrame) -> pd.DataFrame:
    """Simple expected points based on form + minutes."""
    df = players.copy()

    # Ensure numeric
    df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)

    # Very simple heuristic: form scaled by minutes played
    df["xPts"] = (df["form"] * 0.6) + (df["minutes"] / 1000 * 0.4)
    return df


def v2_expected_points(players: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Smarter expected points model:
    - Uses form
    - Minutes weighting
    - Fixture softness (based on opponent difficulty)
    - Role adjustment (FWD/MID > DEF > GK)
    """

    df = players.copy()

    # Ensure numeric
    df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)

    # Base points: form + minutes
    base_xpts = df["form"] * 0.5 + (df["minutes"] / 1000) * 0.3

    # Fixture softness
    fixture_diffs = []
    for _, player in df.iterrows():
        team_id = player["team"]
        team_fixt = fixtures[
            (fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)
        ].head(horizon)

        if not team_fixt.empty:
            # Opponent strength
            opp_strength = []
            for _, row in team_fixt.iterrows():
                if row["team_h"] == team_id:
                    opp_id = row["team_a"]
                    opp_strength.append(row["team_a_difficulty"])
                else:
                    opp_id = row["team_h"]
                    opp_strength.append(row["team_h_difficulty"])

            avg_diff = np.mean(opp_strength)
            fixture_softness = max(0, (5 - avg_diff) / 5)  # scale 0-1
        else:
            fixture_softness = 0.5

        fixture_diffs.append(fixture_softness)

    df["fixture_softness"] = fixture_diffs

    # Position weights
    pos_weights = {"FWD": 1.2, "MID": 1.1, "DEF": 0.9, "GK": 0.8}
    df["role_weight"] = df["pos"].map(pos_weights).fillna(1.0)

    # Final xPts
    df["xPts"] = (
        base_xpts +
        df["fixture_softness"] * 2
    ) * df["role_weight"]

    return df

import pandas as pd

def add_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds value-related helper columns for player selection.
    - xPts_per_m: Expected points per million (budget efficiency)
    - value: xPts per cost unit
    """
    df = df.copy()
    if "xPts" in df.columns and "now_cost" in df.columns:
        df["xPts_per_m"] = df["xPts"] / (df["now_cost"] / 10)  # cost in £m
        df["value"] = df["xPts"] / df["now_cost"]
    return df
