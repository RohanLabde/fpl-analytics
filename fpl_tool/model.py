# model.py
import pandas as pd
import numpy as np


def build_player_master(players: pd.DataFrame, teams: pd.DataFrame, element_types: pd.DataFrame) -> pd.DataFrame:
    df = players.copy()

    team_map = teams.set_index("id")["name"].to_dict()
    df["team_name"] = df["team"].map(team_map)

    pos_map = element_types.set_index("id")["singular_name_short"].to_dict()
    df["pos"] = df["element_type"].map(pos_map)

    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    df["id"] = df["id"].astype(int)

    return df


def v2_expected_points(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    horizon: int = 5,
    form_weight: float = 0.3,
    bonus_weight: float = 0.2,
    form_window: int = 5,
) -> pd.DataFrame:
    """
    Expected points model with:
    - Underlying stats (xG+xA, CS, saves)
    - Fixture difficulty
    - Form adjustment (slider controlled)
    - Bonus tendency adjustment
    - Minutes security adjustment
    """

    df = players.copy()

    # ---------------- Numeric safety ----------------
    num_cols = [
        "expected_goals_per_90",
        "expected_assists_per_90",
        "saves_per_90",
        "minutes",
        "now_cost",
        "selected_by_percent",
        "points_per_game",
        "form",
        "bonus",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    # ---------------- Base attacking ----------------
    df["xAttack_per90"] = df["expected_goals_per_90"] + df["expected_assists_per_90"]

    # ---------------- Fixture difficulty factors ----------------
    fixtures = fixtures.copy()
    if "team_h_difficulty" not in fixtures.columns:
        fixtures["team_h_difficulty"] = 3
        fixtures["team_a_difficulty"] = 3

    att_factors = []
    cs_probs = []

    for _, r in df.iterrows():
        team_id = r["team"]
        team_fixt = fixtures[(fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)].copy()
        if "kickoff_time" in team_fixt.columns:
            team_fixt = team_fixt.sort_values("kickoff_time")
        team_fixt = team_fixt.head(horizon)

        af = []
        cs = []
        for _, fx in team_fixt.iterrows():
            if fx["team_h"] == team_id:
                diff = fx["team_h_difficulty"]
            else:
                diff = fx["team_a_difficulty"]

            cs_prob = max(0.05, (5 - diff) / 5)
            att = 1 + (3 - diff) * 0.10

            af.append(att)
            cs.append(cs_prob)

        att_factors.append(np.mean(af) if af else 1.0)
        cs_probs.append(np.mean(cs) if cs else 0.25)

    df["att_factor"] = att_factors
    df["cs_prob"] = cs_probs

    # ---------------- Base xPts per match ----------------
    df["xSaves_per_match"] = df["saves_per_90"] * 0.33
    df["xAttack_adj"] = df["xAttack_per90"] * df["att_factor"]

    base_xpts = []
    for _, r in df.iterrows():
        appearance = 2.0
        if r["pos"] in ["MID", "FWD"]:
            xp = r["xAttack_adj"] + appearance
        elif r["pos"] == "DEF":
            xp = r["xAttack_adj"] + (r["cs_prob"] * 4) + appearance
        elif r["pos"] == "GKP":
            xp = (r["cs_prob"] * 4) + r["xSaves_per_match"] + appearance
        else:
            xp = r["xAttack_adj"] + appearance
        base_xpts.append(xp)

    df["xPts_base"] = base_xpts

    # ---------------- FORM FACTOR ----------------
    season_ppg = df["points_per_game"].replace(0, np.nan)
    recent_form = df["form"].replace(0, np.nan)

    form_factor = (recent_form / season_ppg).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    form_factor = form_factor.clip(0.7, 1.3)

    # ---------------- BONUS FACTOR ----------------
    avg_bonus = df["bonus"].mean() if df["bonus"].mean() > 0 else 1
    bonus_factor = (df["bonus"] / avg_bonus).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    bonus_factor = bonus_factor.clip(0.8, 1.2)

    # ---------------- MINUTES SECURITY ----------------
    minutes_factor = (df["minutes"] / 90).clip(0, 1.0)

    # ---------------- FINAL xPts ----------------
    df["xPts_per_match"] = (
        df["xPts_base"]
        * (1 + form_weight * (form_factor - 1))
        * (1 + bonus_weight * (bonus_factor - 1))
        * minutes_factor
    )

    df["xPts_total"] = df["xPts_per_match"] * horizon

    # Expose internals for debugging
    df["form_factor"] = form_factor
    df["bonus_factor"] = bonus_factor
    df["minutes_factor"] = minutes_factor

    # Price
    df["price_m"] = df["now_cost"] / 10

    return df


def add_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    price = out["price_m"].replace(0, np.nan)

    out["xPts_per_m"] = out["xPts_per_match"] / price
    out["xPts_total_per_m"] = out["xPts_total"] / price

    out = out.replace([np.inf, -np.inf], 0).fillna(0)
    return out
