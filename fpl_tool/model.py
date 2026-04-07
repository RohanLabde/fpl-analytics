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
# TEAM STRENGTH MODEL
# ----------------------------
def build_team_strength(fixtures: pd.DataFrame, teams: pd.DataFrame):
    fx = fixtures[fixtures["finished"] == True].copy()

    rows = []

    for _, t in teams.iterrows():
        tid = t["id"]

        team_fx = fx[(fx["team_h"] == tid) | (fx["team_a"] == tid)]

        if len(team_fx) == 0:
            rows.append({"team": tid, "att": 1.0, "def": 1.0})
            continue

        GF = 0
        GA = 0

        for _, m in team_fx.iterrows():
            if m["team_h"] == tid:
                gf = m["team_h_score"]
                ga = m["team_a_score"]
            else:
                gf = m["team_a_score"]
                ga = m["team_h_score"]

            GF += gf
            GA += ga

        matches = len(team_fx)

        att = GF / matches
        deff = GA / matches

        rows.append({
            "team": tid,
            "attack_strength": att,
            "defense_weakness": deff
        })

    df = pd.DataFrame(rows)

    # Normalize around league average
    df["attack_strength"] /= df["attack_strength"].mean()
    df["defense_weakness"] /= df["defense_weakness"].mean()

    return df.set_index("team").to_dict("index")


# ----------------------------
# MAIN MODEL (v3)
# ----------------------------
def v3_expected_points(
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

    # Base attacking ability
    df["xAttack_per90"] = df["expected_goals_per_90"] + df["expected_assists_per_90"]

    # ----------------------------
    # Build team strength
    # ----------------------------
    team_strength = build_team_strength(fixtures, teams)

    # ----------------------------
    # Fixture-by-fixture simulation
    # ----------------------------
    total_xpts = []

    for _, p in df.iterrows():
        tid = p["team"]

        team_fx = fixtures[(fixtures["team_h"] == tid) | (fixtures["team_a"] == tid)].copy()

        if "kickoff_time" in team_fx.columns:
            team_fx = team_fx.sort_values("kickoff_time")

        team_fx = team_fx.head(horizon)

        player_total = 0

        for _, fx in team_fx.iterrows():

            # Identify opponent + home/away
            if fx["team_h"] == tid:
                opponent = fx["team_a"]
                home = True
            else:
                opponent = fx["team_h"]
                home = False

            opp = team_strength.get(opponent, {"attack_strength": 1.0, "defense_weakness": 1.0})

            opp_def = opp["defense_weakness"]
            opp_att = opp["attack_strength"]

            # Home/Away adjustment
            home_factor = 1.1 if home else 0.9

            # Attack scaling
            attack_factor = opp_def * home_factor

            # Clean sheet probability (inverse of opponent attack)
            cs_prob = np.exp(-opp_att)

            # Saves (GK only)
            saves = p["saves_per_90"] * 0.33

            appearance = 2.0

            # Position scoring
            if p["pos"] in ["MID", "FWD"]:
                xp = (p["xAttack_per90"] * attack_factor) + appearance

            elif p["pos"] == "DEF":
                xp = (p["xAttack_per90"] * attack_factor) + (cs_prob * 4) + appearance

            elif p["pos"] in ["GKP", "GK"]:
                xp = (cs_prob * 4) + saves + appearance

            else:
                xp = (p["xAttack_per90"] * attack_factor) + appearance

            player_total += xp

        total_xpts.append(player_total)

    df["xPts_total"] = total_xpts
    df["xPts_per_match"] = df["xPts_total"] / horizon

    # ============================================================
    # SMART ADJUSTMENTS (same as before)
    # ============================================================

    # FORM
    df["form"] = df["form"].clip(lower=0)
    league_avg_form = df["form"].replace(0, np.nan).mean()

    form_delta = (df["form"] - league_avg_form) / league_avg_form
    df["form_factor"] = 1 + form_weight * np.tanh(form_delta)
    df["form_factor"] = df["form_factor"].clip(0.85, 1.15)

    # BONUS
    bonus_per_90 = df["bonus"] / (df["minutes"] / 90).replace(0, np.nan)
    bonus_per_90 = bonus_per_90.fillna(0)

    league_avg_bonus = bonus_per_90.replace(0, np.nan).mean()

    bonus_delta = (bonus_per_90 - league_avg_bonus) / league_avg_bonus
    df["bonus_factor"] = 1 + bonus_weight * np.tanh(bonus_delta)
    df["bonus_factor"] = df["bonus_factor"].clip(0.9, 1.1)

    # MINUTES
    df["minutes_factor"] = 0.5 + 0.5 * np.tanh((df["minutes"] - 900) / 900)
    df["minutes_factor"] = 1 + minutes_weight * (df["minutes_factor"] - 1)
    df["minutes_factor"] = df["minutes_factor"].clip(0.7, 1.05)

    # Apply adjustments
    df["xPts_per_match"] *= df["form_factor"] * df["bonus_factor"] * df["minutes_factor"]
    df["xPts_total"] *= df["form_factor"] * df["bonus_factor"] * df["minutes_factor"]

    # ----------------------------
    # Value metrics
    # ----------------------------
    df["price_m"] = df["now_cost"] / 10
    df["xPts_per_m"] = df["xPts_per_match"] / df["price_m"].replace(0, np.nan)
    df["xPts_per_m"] = df["xPts_per_m"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # ----------------------------
    # Final output
    # ----------------------------
    keep = [
        "id", "web_name", "team_name", "pos",
        "price_m", "minutes", "selected_by_percent",
        "form_factor", "bonus_factor", "minutes_factor",
        "xPts_per_match", "xPts_total", "xPts_per_m"
    ]

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
