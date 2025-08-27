
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

import numpy as np
import pandas as pd

def _recent_team_goals(fixtures: pd.DataFrame, lookback: int = 6):
    """
    From the fixtures table (which includes finished past games), compute
    rolling average Goals For (GF) and Goals Against (GA) for each team
    over the last `lookback` finished matches.
    Returns two DataFrames: gf_avg[team_id], ga_avg[team_id]
    """
    f = fixtures.copy()
    # Keep only finished games that have scores
    f = f[(f["finished"] == True) & f["team_h_score"].notna() & f["team_a_score"].notna()]
    cols = ["event","team_h","team_a","team_h_score","team_a_score"]
    f = f[cols].sort_values("event")

    # Home rows
    home = f[["event","team_h","team_a","team_h_score","team_a_score"]].rename(
        columns={"team_h":"team", "team_a":"opp", "team_h_score":"gf", "team_a_score":"ga"}
    )
    # Away rows
    away = f[["event","team_h","team_a","team_h_score","team_a_score"]].rename(
        columns={"team_a":"team", "team_h":"opp", "team_a_score":"gf", "team_h_score":"ga"}
    )
    allg = pd.concat([home, away], ignore_index=True).sort_values(["team","event"])

    # Rolling averages per team
    allg["gf_roll"] = allg.groupby("team")["gf"].rolling(lookback, min_periods=1).mean().reset_index(0, drop=True)
    allg["ga_roll"] = allg.groupby("team")["ga"].rolling(lookback, min_periods=1).mean().reset_index(0, drop=True)

    gf_avg = allg.groupby("team")["gf_roll"].last()
    ga_avg = allg.groupby("team")["ga_roll"].last()
    return gf_avg, ga_avg

def next_gw_pairs(fixtures: pd.DataFrame, events: pd.DataFrame):
    """
    Return a DataFrame with (team_id, opp_id, is_home) for the *next* GW only.
    Handles single fixtures (no DGW handling in V2 to keep it simple).
    """
    if "is_next" in events.columns and (events["is_next"] == True).any():
        gw = int(events.loc[events["is_next"] == True, "id"].iloc[0])
    else:
        # first unfinished GW, else max id
        if "finished" in events.columns and (~events["finished"]).any():
            gw = int(events.loc[~events["finished"], "id"].min())
        else:
            gw = int(events["id"].max()) if "id" in events.columns else 1

    f = fixtures.copy()
    f = f[(f["event"] == gw) & (f["finished"] == False)]
    if f.empty:
        return pd.DataFrame(columns=["team","opp","is_home"])

    home = f[["team_h","team_a"]].rename(columns={"team_h":"team","team_a":"opp"})
    home["is_home"] = True
    away = f[["team_h","team_a"]].rename(columns={"team_a":"team","team_h":"opp"})
    away["is_home"] = False
    return pd.concat([home, away], ignore_index=True)

def simple_expected_goals(fixtures: pd.DataFrame, events: pd.DataFrame, home_adv: float = 1.10):
    """
    Tiny Poisson prep:
    - Team attacking lambda (λ_for) = recent GF
    - Opponent defensive weakness (λ_opp_def) = opponent recent GA
    - Expected goals for this game = sqrt(λ_for * λ_opp_def) * (home_adv if home else 1)
    Also returns opponent's expected goals against you (for CS probability)
    """
    gf_avg, ga_avg = _recent_team_goals(fixtures, lookback=6)
    pairs = next_gw_pairs(fixtures, events)
    if pairs.empty:
        return {}

    eg = {}
    for _, r in pairs.iterrows():
        team = int(r["team"]); opp = int(r["opp"]); is_home = bool(r["is_home"])
        lam_for_team = float(gf_avg.get(team, 1.2))           # fallback if no data
        lam_opp_def  = float(ga_avg.get(opp, 1.2))
        lam_for = np.sqrt(max(lam_for_team, 0.1) * max(lam_opp_def, 0.1))
        if is_home:
            lam_for *= home_adv

        # opponent expected goals vs this team
        lam_opp_att = float(gf_avg.get(opp, 1.2))
        lam_team_def = float(ga_avg.get(team, 1.2))
        lam_against = np.sqrt(max(lam_opp_att, 0.1) * max(lam_team_def, 0.1))
        if not is_home:
            lam_against *= home_adv  # they have home advantage

        eg[team] = {"opp": opp, "is_home": is_home, "lam_for": lam_for, "lam_against": lam_against}
    return eg

def status_to_start_prob(status: str, chance_next_round):
    """
    Convert FPL status/chance to a simple probability of starting.
    status: 'a' (available), 'd' (doubtful), 'i' (injured), 's' (suspended)
    chance_next_round: may be NaN; if provided, prefer that percent/100.
    """
    try:
        if pd.notna(chance_next_round):
            p = float(chance_next_round) / 100.0
            return max(0.0, min(1.0, p))
    except Exception:
        pass

    s = str(status or "").lower()
    if s == "a": return 0.9
    if s == "d": return 0.5
    if s in ("i","s"): return 0.1
    return 0.7  # unknown
