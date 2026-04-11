import pandas as pd
import numpy as np


# ----------------------------
# Build player master
# ----------------------------
def build_player_master(players, teams, element_types):
    df = players.copy()

    team_map = teams.set_index("id")["name"].to_dict()
    df["team_name"] = df["team"].map(team_map)

    pos_map = element_types.set_index("id")["singular_name_short"].to_dict()
    df["pos"] = df["element_type"].map(pos_map)

    return df


# ----------------------------
# Team strength
# ----------------------------
def build_team_strength(fixtures, teams):
    fx = fixtures[fixtures["finished"] == True]

    rows = []

    for _, t in teams.iterrows():
        tid = t["id"]

        team_fx = fx[(fx["team_h"] == tid) | (fx["team_a"] == tid)]

        if len(team_fx) == 0:
            rows.append({"team": tid, "attack_strength": 1.0, "defense_weakness": 1.0})
            continue

        GF = GA = 0

        for _, m in team_fx.iterrows():
            if m["team_h"] == tid:
                GF += m["team_h_score"]
                GA += m["team_a_score"]
            else:
                GF += m["team_a_score"]
                GA += m["team_h_score"]

        matches = len(team_fx)

        rows.append({
            "team": tid,
            "attack_strength": GF / matches,
            "defense_weakness": GA / matches
        })

    df = pd.DataFrame(rows)

    df["attack_strength"] /= df["attack_strength"].mean()
    df["defense_weakness"] /= df["defense_weakness"].mean()

    return df.set_index("team").to_dict("index")


# ----------------------------
# Minutes model
# ----------------------------
def estimate_minutes(row):
    mins = row["minutes"]

    if mins > 2000:
        return 85
    elif mins > 1200:
        return 75
    elif mins > 800:
        return 65
    elif mins > 400:
        return 50
    elif mins > 200:
        return 35
    else:
        return 20


# ----------------------------
# MAIN MODEL v5
# ----------------------------
def v5_expected_points(
    players,
    fixtures,
    teams,
    horizon=5,
    form_weight=0.25,
    bonus_weight=0.15,
):

    df = players.copy()

    # Ensure numeric
    cols = [
        "expected_goals_per_90",
        "expected_assists_per_90",
        "saves_per_90",
        "minutes",
        "form",
        "bonus",
        "selected_by_percent",
        "now_cost"
    ]

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["xAttack_per90"] = df["expected_goals_per_90"] + df["expected_assists_per_90"]

    # Team strength
    team_strength = build_team_strength(fixtures, teams)

    # Expected minutes
    df["exp_minutes"] = df.apply(estimate_minutes, axis=1)
    df["mins_factor"] = df["exp_minutes"] / 90

    total_pts = []
    fixtures_count = []

    for _, p in df.iterrows():
        tid = p["team"]

        # FUTURE FIXTURES ONLY
        team_fx = fixtures[
            ((fixtures["team_h"] == tid) | (fixtures["team_a"] == tid)) &
            (fixtures["finished"] == False)
        ].copy()

        if "kickoff_time" in team_fx.columns:
            team_fx = team_fx.sort_values("kickoff_time")

        # GROUP BY GAMEWEEK
        gw_groups = team_fx.groupby("event")

        next_gws = sorted(gw_groups.groups.keys())[:horizon]

        player_total = 0
        total_matches = 0

        for gw in next_gws:
            gw_matches = gw_groups.get_group(gw)
            gw_points = 0

            total_matches += len(gw_matches)

            for _, fx in gw_matches.iterrows():

                if fx["team_h"] == tid:
                    opp = fx["team_a"]
                    home = True
                else:
                    opp = fx["team_h"]
                    home = False

                opp_data = team_strength.get(opp, {"attack_strength": 1, "defense_weakness": 1})

                opp_def = opp_data["defense_weakness"]
                opp_att = opp_data["attack_strength"]

                home_factor = 1.1 if home else 0.9
                attack_factor = opp_def * home_factor

                cs_prob = np.exp(-opp_att)

                attack_pts = p["xAttack_per90"] * p["mins_factor"] * attack_factor
                appearance = 2 * p["mins_factor"]

                if p["pos"] in ["MID", "FWD"]:
                    xp = attack_pts + appearance

                elif p["pos"] == "DEF":
                    xp = attack_pts + (cs_prob * 4 * p["mins_factor"]) + appearance

                elif p["pos"] in ["GKP", "GK"]:
                    saves = p["saves_per_90"] * p["mins_factor"]
                    xp = (cs_prob * 4 * p["mins_factor"]) + saves + appearance

                else:
                    xp = attack_pts + appearance

                gw_points += xp

            player_total += gw_points

        total_pts.append(player_total)
        fixtures_count.append(total_matches)

    df["xPts_total"] = total_pts
    df["xPts_per_match"] = df["xPts_total"] / horizon
    df["fixtures_in_horizon"] = fixtures_count

    # ----------------------------
    # Adjustments
    # ----------------------------
    avg_form = df["form"].replace(0, np.nan).mean()
    df["form_factor"] = 1 + form_weight * np.tanh((df["form"] - avg_form) / avg_form)
    df["form_factor"] = df["form_factor"].clip(0.85, 1.15)

    bonus_per90 = df["bonus"] / (df["minutes"] / 90).replace(0, np.nan)
    avg_bonus = bonus_per90.replace(0, np.nan).mean()

    df["bonus_factor"] = 1 + bonus_weight * np.tanh((bonus_per90 - avg_bonus) / avg_bonus)
    df["bonus_factor"] = df["bonus_factor"].clip(0.9, 1.1)

    df["xPts_total"] *= df["form_factor"] * df["bonus_factor"]
    df["xPts_per_match"] *= df["form_factor"] * df["bonus_factor"]

    # Value
    df["price_m"] = df["now_cost"] / 10
    df["xPts_per_m"] = df["xPts_per_match"] / df["price_m"].replace(0, np.nan)

    return df
