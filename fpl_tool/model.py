# model.py
import pandas as pd
import numpy as np


def build_player_master(players, teams, element_types):
    df = players.copy()

    team_map = teams.set_index("id")["name"].to_dict()
    pos_map = element_types.set_index("id")["singular_name_short"].to_dict()

    df["team_name"] = df["team"].map(team_map)
    df["pos"] = df["element_type"].map(pos_map)

    df["id"] = df["id"].astype(int)
    return df


# ----------------------------
# TEAM STRENGTH CALCULATION
# ----------------------------
def build_team_strength(fixtures: pd.DataFrame, teams: pd.DataFrame, last_n=None):
    fx = fixtures[fixtures["finished"] == True].copy()

    rows = []

    for _, t in teams.iterrows():
        tid = t["id"]
        name = t["name"]

        team_fx = fx[(fx["team_h"] == tid) | (fx["team_a"] == tid)].copy()
        if "kickoff_time" in team_fx.columns:
            team_fx = team_fx.sort_values("kickoff_time")

        if last_n is not None:
            team_fx = team_fx.tail(last_n)

        played = len(team_fx)
        if played == 0:
            continue

        GF = GA = CS = 0

        for _, m in team_fx.iterrows():
            if m["team_h"] == tid:
                gf = m["team_h_score"]
                ga = m["team_a_score"]
            else:
                gf = m["team_a_score"]
                ga = m["team_h_score"]

            GF += gf
            GA += ga
            if ga == 0:
                CS += 1

        rows.append({
            "team": tid,
            "team_name": name,
            "played": played,
            "GF": GF,
            "GA": GA,
            "GF_per_match": GF / played,
            "GA_per_match": GA / played,
            "CS_rate": CS / played
        })

    df = pd.DataFrame(rows)

    league_avg_GF = df["GF_per_match"].mean()
    league_avg_GA = df["GA_per_match"].mean()

    df["attack_index"] = df["GF_per_match"] / league_avg_GF
    df["defence_index"] = df["GA_per_match"] / league_avg_GA

    return df.set_index("team")


# ----------------------------
# EXPECTED POINTS MODEL V3
# ----------------------------
def v2_expected_points(players, fixtures, teams, horizon=5):
    df = players.copy()

    numeric_cols = [
        "expected_goals_per_90", "expected_assists_per_90",
        "saves_per_90", "minutes", "now_cost", "selected_by_percent"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)

    df["xAttack_per90"] = df["expected_goals_per_90"] + df["expected_assists_per_90"]

    # Build team strength tables
    season_strength = build_team_strength(fixtures, teams, last_n=None)
    form_strength = build_team_strength(fixtures, teams, last_n=5)

    # Merge season + form
    strength = season_strength.copy()
    for col in ["attack_index", "defence_index", "CS_rate"]:
        strength[col] = 0.6 * season_strength[col] + 0.4 * form_strength[col]

    # Compute fixture-adjusted factors
    att_factors = []
    cs_probs = []

    for _, p in df.iterrows():
        team_id = p["team"]

        team_fixt = fixtures[
            ((fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)) &
            (fixtures["finished"] == False)
        ]

        if "kickoff_time" in team_fixt.columns:
            team_fixt = team_fixt.sort_values("kickoff_time")

        team_fixt = team_fixt.head(horizon)

        per_att = []
        per_cs = []

        for _, fx in team_fixt.iterrows():
            if fx["team_h"] == team_id:
                opp = fx["team_a"]
            else:
                opp = fx["team_h"]

            try:
                atk = strength.loc[team_id, "attack_index"]
                defn = strength.loc[team_id, "defence_index"]
                opp_atk = strength.loc[opp, "attack_index"]
                opp_def = strength.loc[opp, "defence_index"]
                cs_rate = strength.loc[team_id, "CS_rate"]
            except:
                continue

            att_factor = atk / opp_def
            cs_prob = cs_rate * (1 / opp_atk)

            per_att.append(att_factor)
            per_cs.append(cs_prob)

        if per_att:
            att_factors.append(float(np.clip(np.mean(per_att), 0.7, 1.6)))
        else:
            att_factors.append(1.0)

        if per_cs:
            cs_probs.append(float(np.clip(np.mean(per_cs), 0.05, 0.7)))
        else:
            cs_probs.append(0.25)

    df["att_factor"] = att_factors
    df["cs_prob"] = cs_probs

    df["xAttack_adj"] = df["xAttack_per90"] * df["att_factor"]
    df["xSaves_per_match"] = df["saves_per_90"] * 0.33

    xPts = []
    for _, r in df.iterrows():
        pos = r["pos"]
        appearance = 2.0

        if pos in ["MID", "FWD"]:
            xp = r["xAttack_adj"] + appearance
        elif pos == "DEF":
            xp = r["xAttack_adj"] + (r["cs_prob"] * 4) + appearance
        else:  # GKP
            xp = (r["cs_prob"] * 4) + r["xSaves_per_match"] + appearance

        xPts.append(float(xp))

    df["xPts_per_match"] = xPts
    df["xPts_total"] = df["xPts_per_match"] * horizon
    df["xPts"] = df["xPts_per_match"]

    df["price_m"] = df["now_cost"] / 10

    keep = [
        "id", "web_name", "team_name", "pos", "price_m", "now_cost",
        "selected_by_percent", "minutes",
        "xAttack_per90", "att_factor", "cs_prob", "xSaves_per_match",
        "xPts_per_match", "xPts_total", "xPts"
    ]
    return df[keep].copy()


def add_value_columns(df):
    out = df.copy()
    price = out["price_m"].replace(0, np.nan)
    out["xPts_per_m"] = out["xPts_per_match"] / price
    out["xPts_total_per_m"] = out["xPts_total"] / price
    out = out.replace([np.inf, -np.inf], 0).fillna(0)
    return out
