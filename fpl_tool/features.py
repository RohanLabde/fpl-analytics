import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Existing helpers you already had (we keep/extend them)
# ------------------------------------------------------------
def build_player_master(players, teams, positions):
    """Return a joined player table with readable fields we use everywhere."""
    df = players.copy()

    # Normalize price, pos label, team name
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    df["pos"] = df.get("element_type", 0).map(pos_map)
    df["price"] = pd.to_numeric(df.get("now_cost", 0), errors="coerce").fillna(0.0) / 10.0
    team_map = dict(zip(teams["id"], teams["name"]))
    df["name"] = df["team"].map(team_map)

    # Columns we commonly use in the app
    keep = [
        "id","web_name","team","name","pos","status","chance_of_playing_next_round",
        "minutes","goals_scored","assists","bps","ict_index","selected_by_percent",
        "now_cost"
    ]
    # Allow for columns missing in early runs
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    df["price"] = pd.to_numeric(df.get("now_cost", 0), errors="coerce").fillna(0.0) / 10.0

    return df


def fixture_softness(fixtures, teams, horizon=3):
    """(Simple/legacy) Return a dict team_id -> {gw: softness_score} for next few GWs."""
    # This is only used by V1; leave as-is for back-compat.
    out = {}
    if fixtures is None or fixtures.empty:
        return out
    team_ids = teams["id"].tolist()
    for t in team_ids:
        out[t] = {}
    # very rough: use opponent team id as proxy; smaller is "harder"
    for _, row in fixtures.iterrows():
        if "event" not in row or pd.isna(row["event"]):
            continue
        gw = int(row["event"])
        th, ta = int(row["team_h"]), int(row["team_a"])
        out.setdefault(th, {})[gw] = out.get(th, {}).get(gw, 0) + ta
        out.setdefault(ta, {})[gw] = out.get(ta, {}).get(gw, 0) + th
    return out

# ------------------------------------------------------------
# Minutes & Poisson helpers
# ------------------------------------------------------------
def _recent_team_goals(fixtures: pd.DataFrame, lookback: int = 6):
    """
    From finished fixtures, compute rolling average GF/GA for each team
    (last `lookback` matches). Fallbacks are handled in caller.
    """
    f = fixtures.copy()
    if "finished" in f.columns:
        f = f[(f["finished"] == True) & f["team_h_score"].notna() & f["team_a_score"].notna()]
    else:
        f = f[f["team_h_score"].notna() & f["team_a_score"].notna()]

    if f.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    cols = ["event","team_h","team_a","team_h_score","team_a_score"]
    f = f[cols].sort_values("event")

    home = f[["event","team_h","team_a","team_h_score","team_a_score"]].rename(
        columns={"team_h":"team","team_a":"opp","team_h_score":"gf","team_a_score":"ga"}
    )
    away = f[["event","team_h","team_a","team_h_score","team_a_score"]].rename(
        columns={"team_a":"team","team_h":"opp","team_a_score":"gf","team_h_score":"ga"}
    )
    allg = pd.concat([home, away], ignore_index=True).sort_values(["team","event"])
    allg["gf_roll"] = allg.groupby("team")["gf"].rolling(lookback, min_periods=1)\
        .mean().reset_index(0, drop=True)
    allg["ga_roll"] = allg.groupby("team")["ga"].rolling(lookback, min_periods=1)\
        .mean().reset_index(0, drop=True)

    gf_avg = allg.groupby("team")["gf_roll"].last()
    ga_avg = allg.groupby("team")["ga_roll"].last()
    return gf_avg, ga_avg


def _next_gw(events: pd.DataFrame) -> int:
    if "is_next" in events.columns and (events["is_next"] == True).any():
        return int(events.loc[events["is_next"] == True, "id"].iloc[0])
    if "finished" in events.columns and (~events["finished"]).any():
        return int(events.loc[~events["finished"], "id"].min())
    return int(events["id"].max()) if "id" in events.columns else 1


def status_to_start_prob(status: str, chance_next_round):
    """
    Convert FPL status/chance to probability of starting this match.
    If chance% provided, prefer that.
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
    return 0.7


# ------------------------------------------------------------
# Horizon-aware expected goals
# ------------------------------------------------------------
def simple_expected_goals_horizon(
    fixtures: pd.DataFrame,
    events: pd.DataFrame,
    horizon: int = 1,
    home_adv: float = 1.10,
):
    """
    Return, for each team, the MEAN expected-goals-for and expected-goals-against
    across the next `horizon` fixtures, plus how many fixtures were found.
    Uses recent rolling GF/GA as simple Poisson inputs.
    """
    if fixtures is None or fixtures.empty:
        return {}

    next_gw = _next_gw(events)

    # Upcoming fixtures in [next_gw, next_gw + horizon - 1]
    f = fixtures.copy()
    f = f[(f["event"] >= next_gw) & (f["event"] < next_gw + horizon)]
    if f.empty:
        return {}

    gf_avg, ga_avg = _recent_team_goals(fixtures, lookback=6)

    # Build per-fixture lambdas
    recs = []
    for _, row in f.iterrows():
        gw = int(row["event"])
        th, ta = int(row["team_h"]), int(row["team_a"])

        # home team expected goals
        lam_for_h = np.sqrt(max(gf_avg.get(th, 1.2), 0.1) * max(ga_avg.get(ta, 1.2), 0.1)) * home_adv
        lam_against_h = np.sqrt(max(gf_avg.get(ta, 1.2), 0.1) * max(ga_avg.get(th, 1.2), 0.1))

        # away team expected goals
        lam_for_a = np.sqrt(max(gf_avg.get(ta, 1.2), 0.1) * max(ga_avg.get(th, 1.2), 0.1))
        lam_against_a = np.sqrt(max(gf_avg.get(th, 1.2), 0.1) * max(ga_avg.get(ta, 1.2), 0.1)) * home_adv

        recs.append({"team": th, "lam_for": lam_for_h, "lam_against": lam_against_h})
        recs.append({"team": ta, "lam_for": lam_for_a, "lam_against": lam_against_a})

    df = pd.DataFrame(recs)
    if df.empty:
        return {}

    agg = df.groupby("team").agg(lam_for_mean=("lam_for","mean"),
                                 lam_against_mean=("lam_against","mean"),
                                 n=("lam_for","count")).to_dict(orient="index")
    return agg
