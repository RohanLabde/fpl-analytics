
import pandas as pd
import numpy as np

def build_player_master(players, teams, positions):
    if players is None or players.empty:
        return pd.DataFrame()
    teams_small = teams[["id","name","strength_overall_home","strength_overall_away"]].rename(columns={"id":"team"}) if not teams.empty else pd.DataFrame(columns=["team","name","strength_overall_home","strength_overall_away"])
    pos_map = positions[["id","plural_name_short"]].rename(columns={"id":"element_type","plural_name_short":"pos"}) if not positions.empty else pd.DataFrame(columns=["element_type","pos"])
    df = players.merge(teams_small, on="team", how="left").merge(pos_map, on="element_type", how="left")
    df["price"] = df["now_cost"].fillna(0) / 10.0
    # selected_by_percent can be string with %, ensure float
    df["sel"] = pd.to_numeric(df.get("selected_by_percent", 0).astype(str).str.replace("%","", regex=False), errors="coerce").fillna(0.0)
    return df

def fixture_softness(fixtures, teams, horizon=3):
    if fixtures is None or fixtures.empty:
        return {}
    f = fixtures.copy()
    f["event"] = f["event"].fillna(-1).astype(int)
    f = f[f["event"]>0]
    if f.empty:
        return {}
    teams_small = teams[["id","strength_overall_home","strength_overall_away"]].copy()
    soft = {}
    max_gw = int(f["event"].max())
    min_gw = int(f["event"].min())

    def opp_strength(opp_id, is_home):
        row = teams_small[teams_small["id"]==opp_id]
        if row.empty:
            return 3.0
        row = row.iloc[0]
        return float(row["strength_overall_home"] if is_home else row["strength_overall_away"])

    for tid in teams_small["id"]:
        team_fixt = f[(f["team_h"]==tid) | (f["team_a"]==tid)].sort_values("event")
        roll = {}
        for gw in range(min_gw, max_gw+1):
            nxt = team_fixt[team_fixt["event"].between(gw, gw+horizon-1)]
            if nxt.empty:
                roll[gw] = np.nan
            else:
                vals = []
                for _, row in nxt.iterrows():
                    is_home = (row["team_h"]==tid)
                    opp = int(row["team_a"] if is_home else row["team_h"])
                    vals.append(opp_strength(opp, is_home))
                roll[gw] = float(np.mean(vals)) if len(vals)>0 else np.nan
        soft[tid] = roll
    return soft
