# model.py
import pandas as pd
import numpy as np


def build_player_master(players: pd.DataFrame, teams: pd.DataFrame, element_types: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the raw players DataFrame with team name and short position label (GKP/DEF/MID/FWD).
    """
    df = players.copy()

    # map team id -> name
    team_map = teams.set_index("id")["name"].to_dict()
    df["team_name"] = df["team"].map(team_map)

    # map element_type id -> short name
    pos_map = element_types.set_index("id")["singular_name_short"].to_dict()
    df["pos"] = df["element_type"].map(pos_map)

    # ensure id exists
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
):
    """
    Advanced xPts model:

    Base:
      - Attack (xG+xA)
      - Clean sheet prob
      - Saves
      - Appearance pts

    Multipliers:
      - minutes_factor (starts-based nailedness)
      - form_factor (momentum)
      - bonus_factor (bonus magnet)

    Final:
      xPts_per_match = base * minutes_factor * form_adj * bonus_adj
    """

    df = players.copy()

    # -------------------------
    # Ensure numeric columns
    # -------------------------
    numeric_cols = [
        "expected_goals_per_90",
        "expected_assists_per_90",
        "saves_per_90",
        "minutes",
        "now_cost",
        "selected_by_percent",
        "starts",
        "appearances",
        "bonus",
        "form",
        "points",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # -------------------------
    # Base attacking
    # -------------------------
    df["xAttack_per90"] = df["expected_goals_per_90"] + df["expected_assists_per_90"]

    # -------------------------
    # Fixture difficulty model
    # -------------------------
    fixtures = fixtures.copy()
    if "team_h_difficulty" not in fixtures.columns:
        fixtures["team_h_difficulty"] = 3
        fixtures["team_a_difficulty"] = 3

    att_factors = []
    cs_probs = []

    for _, player in df.iterrows():
        team_id = player["team"]

        team_fx = fixtures[
            (fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)
        ].copy()

        if "kickoff_time" in team_fx.columns:
            team_fx = team_fx.sort_values("kickoff_time")

        team_fx = team_fx.head(horizon)

        per_att = []
        per_cs = []

        for _, fx in team_fx.iterrows():
            if fx["team_h"] == team_id:
                diff = fx["team_h_difficulty"]
            else:
                diff = fx["team_a_difficulty"]

            # Clean sheet prob
            cs = max(0.05, (5 - diff) / 5)
            per_cs.append(cs)

            # Attack factor
            att = 1.0 + (3 - diff) * 0.10
            per_att.append(att)

        att_factors.append(np.mean(per_att) if per_att else 1.0)
        cs_probs.append(np.mean(per_cs) if per_cs else 0.25)

    df["att_factor"] = att_factors
    df["cs_prob"] = cs_probs

    # -------------------------
    # Saves
    # -------------------------
    df["xSaves_per_match"] = df["saves_per_90"] * 0.33

    # -------------------------
    # Base per-match xPts (no multipliers yet)
    # -------------------------
    df["xAttack_adj"] = df["xAttack_per90"] * df["att_factor"]

    base_pts = []
    for _, r in df.iterrows():
        pos = r["pos"]
        appearance = 2.0

        if pos in ["MID", "FWD"]:
            xp = r["xAttack_adj"] + appearance
        elif pos == "DEF":
            xp = r["xAttack_adj"] + (r["cs_prob"] * 4.0) + appearance
        elif pos == "GKP":
            xp = (r["cs_prob"] * 4.0) + r["xSaves_per_match"] + appearance
        else:
            xp = r["xAttack_adj"] + appearance

        base_pts.append(xp)

    df["base_xPts_per_match"] = base_pts

    # -------------------------
    # (B) Minutes factor — STARTS BASED
    # -------------------------
    df["starts"] = df["starts"].replace(0, np.nan)
    df["appearances"] = df["appearances"].replace(0, np.nan)

    starts_ratio = (df["starts"] / df["appearances"]).fillna(0.0)

    # Clamp: rotation players punished, nailed players full value
    df["minutes_factor"] = starts_ratio.clip(0.5, 1.0)

    # -------------------------
    # Form factor
    # -------------------------
    df["points_per_app"] = (df["points"] / df["appearances"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["form_per_app"] = df["form"]

    raw_form_factor = df["form_per_app"] / df["points_per_app"].replace(0, np.nan)
    raw_form_factor = raw_form_factor.replace([np.inf, -np.inf], 1.0).fillna(1.0)

    df["form_factor"] = raw_form_factor.clip(0.7, 1.3)

    # -------------------------
    # Bonus factor
    # -------------------------
    df["bonus_per_app"] = (df["bonus"] / df["appearances"]).replace([np.inf, -np.inf], 0).fillna(0)

    league_avg_bonus = df["bonus_per_app"].mean() if df["bonus_per_app"].mean() > 0 else 0.3

    raw_bonus_factor = df["bonus_per_app"] / league_avg_bonus
    raw_bonus_factor = raw_bonus_factor.replace([np.inf, -np.inf], 1.0).fillna(1.0)

    df["bonus_factor"] = raw_bonus_factor.clip(0.8, 1.3)

    # -------------------------
    # Combine everything
    # -------------------------
    df["xPts_per_match"] = (
        df["base_xPts_per_match"]
        * df["minutes_factor"]
        * (1 + form_weight * (df["form_factor"] - 1))
        * (1 + bonus_weight * (df["bonus_factor"] - 1))
    )

    # -------------------------
    # Totals
    # -------------------------
    df["xPts_total"] = df["xPts_per_match"] * float(horizon)
    df["xPts"] = df["xPts_per_match"]

    # -------------------------
    # Price & select %
    # -------------------------
    df["price_m"] = df["now_cost"] / 10.0
    df["selected_by_percent"] = df["selected_by_percent"].fillna(0.0)

    # -------------------------
    # Final columns
    # -------------------------
    keep = [
        "id", "web_name", "team_name", "pos",
        "price_m", "now_cost", "selected_by_percent", "minutes",
        "starts", "appearances",
        "xAttack_per90", "att_factor", "cs_prob", "xSaves_per_match",
        "base_xPts_per_match",
        "minutes_factor", "form_factor", "bonus_factor",
        "xPts_per_match", "xPts_total", "xPts",
    ]

    existing = [c for c in keep if c in df.columns]
    return df[existing].copy()


def add_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    price = out["price_m"].replace(0, np.nan)

    out["xPts_per_m"] = (out["xPts_per_match"] / price).replace([np.inf, -np.inf], 0).fillna(0)
    out["xPts_total_per_m"] = (out["xPts_total"] / price).replace([np.inf, -np.inf], 0).fillna(0)

    return out
