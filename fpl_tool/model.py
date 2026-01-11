import pandas as pd
import numpy as np


# ----------------------------
# Build master player table
# ----------------------------
def build_player_master(players: pd.DataFrame, teams: pd.DataFrame, element_types: pd.DataFrame) -> pd.DataFrame:
    df = players.copy()

    team_map = teams.set_index("id")["name"].to_dict()
    df["team_name"] = df["team"].map(team_map)

    pos_map = element_types.set_index("id")["singular_name_short"].to_dict()
    df["pos"] = df["element_type"].map(pos_map)

    df["id"] = df["id"].astype(int)
    return df


# ----------------------------
# Main xPts model
# ----------------------------
def v2_expected_points(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    horizon: int = 5,
    form_weight: float = 0.25,
    bonus_weight: float = 0.15,
    minutes_weight: float = 0.25,
):
    df = players.copy()

    # ----------------------------
    # Ensure numeric columns
    # ----------------------------
    num_cols = [
        "expected_goals_per_90",
        "expected_assists_per_90",
        "saves_per_90",
        "minutes",
        "now_cost",
        "selected_by_percent",
        "form",
        "bonus",
    ]
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # ----------------------------
    # Base attacking model
    # ----------------------------
    df["xAttack_per90"] = df["expected_goals_per_90"] + df["expected_assists_per_90"]

    # ----------------------------
    # Fixture difficulty factors
    # ----------------------------
    fixtures = fixtures.copy()
    if "team_h_difficulty" not in fixtures.columns:
        fixtures["team_h_difficulty"] = 3
        fixtures["team_a_difficulty"] = 3

    att_factors = []
    cs_probs = []

    for _, p in df.iterrows():
        tid = p["team"]

        team_fx = fixtures[(fixtures["team_h"] == tid) | (fixtures["team_a"] == tid)].copy()
        if "kickoff_time" in team_fx.columns:
            team_fx = team_fx.sort_values("kickoff_time")

        team_fx = team_fx.head(horizon)

        if len(team_fx) == 0:
            att_factors.append(1.0)
            cs_probs.append(0.25)
            continue

        per_att = []
        per_cs = []

        for _, fx in team_fx.iterrows():
            if fx["team_h"] == tid:
                diff = fx["team_h_difficulty"]
            else:
                diff = fx["team_a_difficulty"]

            cs = max(0.05, (5 - diff) / 5)
            att = 1.0 + (3 - diff) * 0.10

            per_cs.append(cs)
            per_att.append(att)

        att_factors.append(np.mean(per_att))
        cs_probs.append(np.mean(per_cs))

    df["att_factor"] = att_factors
    df["cs_prob"] = cs_probs

    # ----------------------------
    # Saves model
    # ----------------------------
    df["xSaves_per_match"] = df["saves_per_90"] * 0.33

    # ----------------------------
    # Base xPts per match
    # ----------------------------
    df["xAttack_adj"] = df["xAttack_per90"] * df["att_factor"]

    xpts = []
    for _, r in df.iterrows():
        pos = r["pos"]
        appearance = 2.0

        if pos in ["MID", "FWD"]:
            xp = r["xAttack_adj"] + appearance
        elif pos == "DEF":
            xp = r["xAttack_adj"] + (r["cs_prob"] * 4) + appearance
        elif pos in ["GKP", "GK"]:
            xp = (r["cs_prob"] * 4) + r["xSaves_per_match"] + appearance
        else:
            xp = r["xAttack_adj"] + appearance

        xpts.append(float(xp))

    df["xPts_base"] = xpts

    # ============================================================
    # =============== SMART ADJUSTMENT LAYERS ====================
    # ============================================================

    # ----------------------------
    # FORM FACTOR (from FPL API)
    # ----------------------------
    df["form"] = df["form"].clip(lower=0)
    league_avg_form = df["form"].replace(0, np.nan).mean()

    form_delta = (df["form"] - league_avg_form) / league_avg_form
    df["form_factor"] = 1 + form_weight * np.tanh(form_delta)
    df["form_factor"] = df["form_factor"].clip(0.85, 1.15)

    # ----------------------------
    # BONUS FACTOR (bonus per 90)
    # ----------------------------
    bonus_per_90 = df["bonus"] / (df["minutes"] / 90).replace(0, np.nan)
    bonus_per_90 = bonus_per_90.fillna(0)

    league_avg_bonus = bonus_per_90.replace(0, np.nan).mean()

    bonus_delta = (bonus_per_90 - league_avg_bonus) / league_avg_bonus
    df["bonus_factor"] = 1 + bonus_weight * np.tanh(bonus_delta)
    df["bonus_factor"] = df["bonus_factor"].clip(0.9, 1.1)

    # ----------------------------
    # MINUTES RELIABILITY FACTOR
    # ----------------------------
    df["minutes_factor"] = 0.5 + 0.5 * np.tanh((df["minutes"] - 900) / 900)
    df["minutes_factor"] = 1 + minutes_weight * (df["minutes_factor"] - 1)
    df["minutes_factor"] = df["minutes_factor"].clip(0.7, 1.05)

    # ----------------------------
    # Final adjusted xPts
    # ----------------------------
    df["xPts_per_match"] = (
        df["xPts_base"]
        * df["form_factor"]
        * df["bonus_factor"]
        * df["minutes_factor"]
    )

    df["xPts_total"] = df["xPts_per_match"] * float(horizon)

    # ----------------------------
    # Value metrics
    # ----------------------------
    df["price_m"] = df["now_cost"] / 10
    df["xPts_per_m"] = df["xPts_per_match"] / df["price_m"].replace(0, np.nan)
    df["xPts_per_m"] = df["xPts_per_m"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # ----------------------------
    # Return final table
    # ----------------------------
    keep = [
        "id", "web_name", "team_name", "pos", "now_cost", "price_m",
        "minutes", "selected_by_percent", "form", "bonus",
        "att_factor", "cs_prob",
        "form_factor", "bonus_factor", "minutes_factor",
        "xPts_base", "xPts_per_match", "xPts_total", "xPts_per_m"
    ]

    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


# ----------------------------
# Value helper
# ----------------------------
def add_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    price = out["price_m"].replace(0, np.nan)

    out["xPts_total_per_m"] = out["xPts_total"] / price
    out["xPts_total_per_m"] = out["xPts_total_per_m"].replace([np.inf, -np.inf], np.nan).fillna(0)

    return out
