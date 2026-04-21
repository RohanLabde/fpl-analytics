import pandas as pd
import numpy as np


# -----------------------
# BUILD PLAYER MASTER
# -----------------------
def build_player_master(players, teams, element_types):

    df = players.copy()

    team_map = teams.set_index("id")["name"].to_dict()
    pos_map = element_types.set_index("id")["singular_name_short"].to_dict()

    df["team_name"] = df["team"].map(team_map)
    df["pos"] = df["element_type"].map(pos_map)

    df["price_m"] = df["now_cost"] / 10

    # -----------------------
    # RAW POINTS PER MATCH
    # -----------------------
    df["xPts_per_match_raw"] = (
        df["goals_scored"] * 4 +
        df["assists"] * 3 +
        df["clean_sheets"] * 4 +
        df["bonus"] * 1
    ) / (df["minutes"] / 90 + 1e-6)

    # -----------------------
    # BAYESIAN SHRINKAGE (KEY FIX)
    # -----------------------
    global_avg = df["xPts_per_match_raw"].mean()

    df["xPts_per_match"] = (
        (df["xPts_per_match_raw"] * df["minutes"] + global_avg * 900)
        / (df["minutes"] + 900)
    )

    return df


# -----------------------
# FIXTURE INFO (DGW/BGW)
# -----------------------
def get_fixture_info(fixtures, horizon):

    upcoming = fixtures[fixtures["finished"] == False].copy()
    upcoming = upcoming[upcoming["event"].notnull()]
    upcoming = upcoming.sort_values("event")

    # Take next N gameweeks
    events = upcoming["event"].unique()[:horizon]
    upcoming = upcoming[upcoming["event"].isin(events)]

    team_fixture_count = {}
    team_difficulty_sum = {}

    for _, row in upcoming.iterrows():

        for team, diff in [
            (row["team_h"], row["team_h_difficulty"]),
            (row["team_a"], row["team_a_difficulty"])
        ]:

            team_fixture_count[team] = team_fixture_count.get(team, 0) + 1
            team_difficulty_sum[team] = team_difficulty_sum.get(team, 0) + diff

    # Average difficulty
    team_difficulty_avg = {
        team: team_difficulty_sum[team] / team_fixture_count[team]
        for team in team_fixture_count
    }

    return team_fixture_count, team_difficulty_avg


# -----------------------
# MINUTES FACTOR
# -----------------------
def get_minutes_factor(minutes):

    if minutes >= 2500:
        return 1.0
    elif minutes >= 1500:
        return 0.9
    elif minutes >= 800:
        return 0.75
    elif minutes >= 300:
        return 0.6
    else:
        return 0.4


# -----------------------
# DIFFICULTY FACTOR
# -----------------------
def get_difficulty_factor(avg_diff):

    return {
        1: 1.3,
        2: 1.15,
        3: 1.0,
        4: 0.85,
        5: 0.7
    }.get(round(avg_diff), 1.0)


# -----------------------
# V6.1 EXPECTED POINTS
# -----------------------
def v6_expected_points(df, fixtures, teams, horizon=5, form_weight=0.3, bonus_weight=0.2):

    df = df.copy()

    # -----------------------
    # 🔥 HARD FILTER (REMOVE NOISE)
    # -----------------------
    df = df[df["minutes"] >= 150].copy()

    # -----------------------
    # FIXTURE INFO
    # -----------------------
    fixture_count, difficulty_avg = get_fixture_info(fixtures, horizon)

    df["fixture_count"] = df["team"].map(lambda x: fixture_count.get(x, 0))
    df["difficulty_avg"] = df["team"].map(lambda x: difficulty_avg.get(x, 3))

    # -----------------------
    # FACTORS
    # -----------------------
    df["minutes_factor"] = df["minutes"].apply(get_minutes_factor)
    df["difficulty_factor"] = df["difficulty_avg"].apply(get_difficulty_factor)

    # -----------------------
    # FINAL EXPECTED POINTS
    # -----------------------
    df["xPts_total"] = (
        df["xPts_per_match"] *
        df["fixture_count"] *
        df["difficulty_factor"] *
        df["minutes_factor"]
    )

    return df
