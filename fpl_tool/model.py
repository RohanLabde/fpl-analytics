# model.py
import pandas as pd
import numpy as np


# -----------------------------
# Utilities
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -----------------------------
# Player master
# -----------------------------
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


# -----------------------------
# Main model
# -----------------------------
def v2_expected_points(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    horizon: int = 5,
    form_weight: float = 0.25,
    bonus_weight: float = 0.15,
    minutes_weight: float = 0.30,
):
    """
    Expected points model using:
    - xG / xA
    - Fixture difficulty
    - Clean sheets
    - Saves
    - FPL Form (DIRECT from API)
    - Bonus per 90
    - Minutes reliability curve
    """

    df = players.copy()

    # -----------------------------
    # Ensure numeric
    # -----------------------------
    numeric_cols = [
        "expected_goals_per_90",
        "expected_assists_per_90",
        "saves_per_90",
        "minutes",
        "now_cost",
        "selected_by_percent",
        "form",
        "bonus",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    # -----------------------------
    # Base attacking
    # -----------------------------
    df["xAttack_per90"] = df["expected_goals_per_90"] + df["expected_assists_per_90"]

    # -----------------------------
    # Fixture difficulty multipliers
    # -----------------------------
    fixtures = fixtures.copy()
    if "team_h_difficulty" not in fixtures.columns:
        fixtures["team_h_difficulty"] = 3
        fixtures["team_a_difficulty"] = 3

    att_factors = []
    cs_probs = []

    for _, player in df.iterrows():
        tid = player["team"]

        team_fx = fixtures[(fixtures["team_h"] == tid) | (fixtures["team_a"] == tid)].copy()
        if "kickoff_time" in team_fx.columns:
            team_fx = team_fx.sort_values("kickoff_time")

        team_fx = team_fx.head(horizon)

        per_att = []
        per_cs = []

        for _, fx in team_fx.iterrows():
            if fx["team_h"] == tid:
                diff = fx["team_h_difficulty"]
            else:
                diff = fx["team_a_difficulty"]

            cs_prob = max(0.05, (5 - diff) / 5)
            att_factor = 1.0 + (3 - diff) * 0.1

            per_cs.append(cs_prob)
            per_att.append(att_factor)

        if per_att:
            att_factors.append(np.mean(per_att))
            cs_probs.append(np.mean(per_cs))
        else:
            att_factors.append(1.0)
            cs_probs.append(0.25)

    df["att_factor"] = att_factors
    df["cs_prob"] = cs_probs

    # -----------------------------
    # Saves
    # -----------------------------
    df["xSaves_per_match"] = df["saves_per_90"] * 0.33

    # -----------------------------
    # Base xPts per match (no form/bonus/minutes yet)
    # -----------------------------
    base = []

    for _, r in df.iterrows():
        pos = r["pos"]
        appearance = 2.0
        xA = r["xAttack_per90"] * r["att_factor"]

        if pos in ["FWD", "MID"]:
            xp = xA + appearance
        elif pos == "DEF":
            xp = xA + r["cs_prob"] * 4 + appearance
        elif pos in ["GKP", "GK"]:
            xp = r["cs_prob"] * 4 + r["xSaves_per_match"] + appearance
        else:
            xp = xA + appearance

        base.append(xp)

    df["xPts_base"] = base

    # ==================================================
    # SMART ADJUSTMENTS
    # ==================================================

    # -----------------------------
    # Minutes factor (sigmoid reliability curve)
    # -----------------------------
    df["minutes_factor"] = 0.3 + 0.7 * sigmoid((df["minutes"] - 900) / 400)

    # -----------------------------
    # ✅ FORM FACTOR (DIRECT FROM FPL API)
    # -----------------------------
    league_avg_form = df["form"].mean() if df["form"].mean() > 0 else 1.0

    df["form_factor"] = 1 + form_weight * ((df["form"] - league_avg_form) / league_avg_form)
    df["form_factor"] = df["form_factor"].clip(0.7, 1.3)

    # -----------------------------
    # Bonus factor (per 90)
    # -----------------------------
    bonus_per_90 = np.where(df["minutes"] > 0, df["bonus"] / df["minutes"] * 90, 0)
    league_avg_bonus = np.mean(bonus_per_90) if np.mean(bonus_per_90) > 0 else 0.01

    df["bonus_factor"] = 1 + bonus_weight * ((bonus_per_90 - league_avg_bonus) / league_avg_bonus)
    df["bonus_factor"] = df["bonus_factor"].clip(0.8, 1.2)

    # -----------------------------
    # Final xPts per match
    # -----------------------------
    df["xPts_per_match"] = (
        df["xPts_base"]
        * df["minutes_factor"]
        * df["form_factor"]
        * df["bonus_factor"]
    )

    # -----------------------------
    # Totals
    # -----------------------------
    df["xPts_total"] = df["xPts_per_match"] * float(horizon)

    # -----------------------------
    # Price
    # -----------------------------
    df["price_m"] = df["now_cost"] / 10

    keep = [
        "id", "web_name", "team_name", "pos",
        "price_m", "now_cost", "selected_by_percent", "minutes",
        "xAttack_per90", "att_factor", "cs_prob", "xSaves_per_match",
        "xPts_base",
        "minutes_factor", "form_factor", "bonus_factor",
        "xPts_per_match", "xPts_total"
    ]

    return df[[c for c in keep if c in df.columns]].copy()


# -----------------------------
# Value columns
# -----------------------------
def add_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    price = out["price_m"].replace(0, np.nan)

    out["xPts_per_m"] = (out["xPts_per_match"] / price).replace([np.inf, -np.inf], 0).fillna(0)
    out["xPts_total_per_m"] = (out["xPts_total"] / price).replace([np.inf, -np.inf], 0).fillna(0)

    return out
