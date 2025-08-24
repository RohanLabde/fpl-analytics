
import numpy as np
import pandas as pd

def _safe_norm(series):
    s = series.copy().astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx - mn < 1e-9:
        return pd.Series([0.5]*len(s), index=s.index)  # neutral
    return (s - mn) / (mx - mn)

def baseline_expected_points(players_df, events, fixtures_softness, horizon=1, alpha=0.7, beta=0.2, gamma=0.1):
    if players_df is None or players_df.empty:
        return pd.DataFrame()
    df = players_df.copy()

    # Determine next GW
    try:
        next_mask = (events["is_next"] == True) if "is_next" in events.columns else None
        if next_mask is not None and next_mask.any():
            next_gw = int(events.loc[next_mask, "id"].iloc[0])
        else:
            if "finished" in events.columns and (~events["finished"]).any():
                next_gw = int(events.loc[~events["finished"], "id"].min())
            else:
                next_gw = int(events["id"].max()) if "id" in events.columns else 1
    except Exception:
        next_gw = 1

    # Features: form, minutes proxy
    df["form"] = pd.to_numeric(df.get("form", 0), errors="coerce").fillna(0.0)
    df["mins"] = pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0.0)
    df["mins_norm"] = _safe_norm(df["mins"])

    # Fixture softness (invert: softer fixture -> higher score)
    fdr_vals = []
    for _, r in df.iterrows():
        team_id = r.get("team", None)
        series = [fixtures_softness.get(team_id, {}).get(next_gw+i, np.nan) for i in range(horizon)]
        m = float(np.nanmean(series)) if len(series)>0 else np.nan
        fdr_vals.append(m if not np.isnan(m) else np.nan)
    df["fdr_raw"] = pd.Series(fdr_vals, index=df.index).fillna(3.0)
    df["fdr_norm_inv"] = 1 - _safe_norm(df["fdr_raw"])  # higher is easier

    # Position names might be NaN; map element_type as fallback
    pos_map = {1:"GKP", 2:"DEF", 3:"MID", 4:"FWD"}
    df["pos"] = df["pos"].fillna(df["element_type"].map(pos_map))

    # xPts
    df["xPts"] = alpha*df["form"] + beta*df["mins_norm"] + gamma*df["fdr_norm_inv"]

    # modest priors by position
    pos_adj = {"GKP":0.15,"DEF":0.10,"MID":0.05,"FWD":0.00}
    df["xPts"] = df.apply(lambda r: r["xPts"] + pos_adj.get(str(r["pos"]),0.0), axis=1)

    # value
    df["price"] = pd.to_numeric(df.get("price", df.get("now_cost", 0)/10.0), errors="coerce").fillna(0.0)
    df["xPts_per_m"] = df["xPts"] / df["price"].replace(0, np.nan)
    return df.sort_values("xPts", ascending=False)
