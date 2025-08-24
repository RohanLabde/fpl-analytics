
import pandas as pd

def pick_squad_greedy(pred_df, budget=100.0):
    """
    Greedy squad builder that avoids external MILP solvers (Render free friendly).
    Constraints: 15 players (2 GKP, 5 DEF, 5 MID, 3 FWD), max 3 per team, budget.
    Heuristic:
      1) Fill required slots per position using best xPts_per_m (value) while obeying team cap and budget.
      2) If budget remains, try to swap within each position to higher xPts.
    """
    df = pred_df.copy()
    df = df.dropna(subset=["pos","price","xPts"]).reset_index(drop=True)

    need = {"GKP":2, "DEF":5, "MID":5, "FWD":3}
    chosen_idx = []
    team_counts = {}
    total_cost = 0.0

    def can_take(row):
        if team_counts.get(row["team"], 0) >= 3: 
            return False
        if total_cost + float(row["price"]) > budget:
            return False
        return True

    # 1) fill by value first
    for pos, cnt_required in need.items():
        cnt = cnt_required
        pool = df[df["pos"]==pos].sort_values(["xPts_per_m","xPts"], ascending=[False, False]).copy()
        for _, r in pool.iterrows():
            if cnt <= 0: break
            if r.name in chosen_idx: 
                continue
            if can_take(r):
                chosen_idx.append(r.name)
                team_counts[r["team"]] = team_counts.get(r["team"], 0) + 1
                total_cost += float(r["price"])
                cnt -= 1
        if cnt > 0:
            # fallback: take cheapest remaining to satisfy counts
            pool = df[df["pos"]==pos].sort_values(["price","xPts"], ascending=[True, False]).copy()
            for _, r in pool.iterrows():
                if cnt <= 0: break
                if r.name in chosen_idx: 
                    continue
                if can_take(r):
                    chosen_idx.append(r.name)
                    team_counts[r["team"]] = team_counts.get(r["team"], 0) + 1
                    total_cost += float(r["price"])
                    cnt -= 1

    squad = df.loc[chosen_idx].copy()

    # 2) attempt simple upgrades within same position (local hill-climb)
    for pos in ["FWD","MID","DEF","GKP"]:
        pool = df[df["pos"]==pos].sort_values("xPts", ascending=False)
        current = squad[squad["pos"]==pos].sort_values("xPts")
        for _, cand in pool.iterrows():
            if current.empty:
                break
            worst = current.head(1).iloc[0]
            if cand.name in squad.index or team_counts.get(cand["team"],0) >= 3:
                continue
            delta_cost = float(cand["price"]) - float(worst["price"])
            if total_cost + delta_cost <= budget and cand["xPts"] > worst["xPts"]:
                # swap
                base = squad.drop(index=worst.name)
                add  = cand.to_frame().T
                squad = pd.concat([base, add], axis=0)
                team_counts[worst["team"]] = max(0, team_counts.get(worst["team"],1) - 1)
                team_counts[cand["team"]] = team_counts.get(cand["team"],0) + 1
                total_cost += delta_cost
                current = squad[squad["pos"]==pos].sort_values("xPts")

    squad = squad.sort_values(["pos","xPts"], ascending=[True, False]).reset_index(drop=True)
    return squad, float(squad["xPts"].sum()), float(squad["price"].sum())

# --- MY SQUAD ANALYZER HELPERS ---

def best_starting_xi(pred_df, squad_ids):
    """
    Pick a legal XI:
      - Exactly 1 GK
      - DEF >= 3
      - MID >= 2
      - FWD >= 1
      - Total = 11
    Greedy: satisfy minimums by xPts, then fill remaining slots by highest xPts.
    Returns: xi_df (11 rows), bench_df (4 rows)
    """
    df = pred_df.set_index("id", drop=False).loc[squad_ids].copy()
    # Safety: some player frames might not have "id" as index
    if "pos" not in df.columns or "xPts" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    # Split by position
    gk = df[df["pos"] == "GKP"].sort_values("xPts", ascending=False)
    de = df[df["pos"] == "DEF"].sort_values("xPts", ascending=False)
    mi = df[df["pos"] == "MID"].sort_values("xPts", ascending=False)
    fw = df[df["pos"] == "FWD"].sort_values("xPts", ascending=False)

    # Must-haves
    pick = []
    pick += list(gk.head(1)["id"])           # 1 GK
    pick += list(de.head(3)["id"])           # 3 DEF
    pick += list(mi.head(2)["id"])           # 2 MID
    pick += list(fw.head(1)["id"])           # 1 FWD

    # Remove duplicates in case of short pools
    pick = list(dict.fromkeys([p for p in pick if p in df["id"].values]))

    # Fill remaining 11 - len(pick) with next-best xPts from remaining outfielders
    remaining_needed = 11 - len(pick)
    if remaining_needed > 0:
        remaining = df[~df["id"].isin(pick)]
        # Outfield only for these extra spots
        remaining = remaining[remaining["pos"].isin(["DEF","MID","FWD"])].sort_values("xPts", ascending=False)
        pick += list(remaining.head(remaining_needed)["id"])

    xi = df[df["id"].isin(pick)].sort_values("xPts", ascending=False)
    bench = df[~df["id"].isin(pick)].sort_values("xPts", ascending=False)
    return xi, bench

def suggest_captain(xi_df):
    """
    Captain = highest xPts in XI; Vice = second-highest.
    """
    if xi_df is None or xi_df.empty:
        return None, None
    top2 = xi_df.sort_values("xPts", ascending=False).head(2)
    cap = top2.iloc[0] if len(top2) >= 1 else None
    vc  = top2.iloc[1] if len(top2) >= 2 else None
    return cap, vc

def best_single_transfer(pred_df, squad_ids, bank):
    """
    Evaluate all 1-player replacements (same position) that obey:
      - Budget: new_cost <= current_cost + bank
      - Max 3 per team
    Returns the best move by delta_xPts (XI before vs XI after), with details.
    """
    df = pred_df.set_index("id", drop=False).copy()
    squad = df.loc[squad_ids].copy()
    current_cost = float(squad["price"].sum())
    team_counts = squad["team"].value_counts().to_dict()
    budget_cap = current_cost + float(bank)

    # Baseline XI points
    base_xi, _ = best_starting_xi(pred_df, squad_ids)
    if base_xi.empty:
        return None
    base_pts = float(base_xi["xPts"].sum())

    best = None  # (delta, out_id, in_id, new_pts, new_cost)

    for _, out_row in squad.iterrows():
        pos = out_row["pos"]
        # Candidates of same position not already in squad
        cands = df[(df["pos"] == pos) & (~df["id"].isin(squad_ids))].copy()
        # Try good options only to keep it quick
        cands = cands.sort_values(["xPts","xPts_per_m"], ascending=[False, False]).head(150)

        for _, cand in cands.iterrows():
            # Team cap check
            out_team = int(out_row["team"])
            in_team = int(cand["team"])
            # simulate team counts
            team_ok = True
            if in_team != out_team:
                cnt = team_counts.get(in_team, 0) + 1
                if cnt > 3:
                    team_ok = False
            if not team_ok:
                continue

            # Budget check
            new_cost = current_cost - float(out_row["price"]) + float(cand["price"])
            if new_cost > budget_cap:
                continue

            new_squad_ids = [pid for pid in squad_ids if pid != int(out_row["id"])] + [int(cand["id"])]
            new_xi, _ = best_starting_xi(pred_df, new_squad_ids)
            if new_xi.empty:
                continue
            new_pts = float(new_xi["xPts"].sum())
            delta = new_pts - base_pts
            if (best is None) or (delta > best[0]):
                best = (delta, int(out_row["id"]), int(cand["id"]), new_pts, new_cost)

    if not best or best[0] <= 0:
        return None
    return {
        "delta": round(best[0], 3),
        "out_id": best[1],
        "in_id": best[2],
        "new_pts": round(best[3], 3),
        "new_cost": round(best[4], 1),
        "base_pts": round(base_pts, 3),
    }

def best_double_transfer(pred_df, squad_ids, bank, top_pool_per_pos=30):
    """
    Simple 2-transfer search (kept small for free hosting):
      - Build small candidate pools per position (top by xPts).
      - Try pairs of positions (including same pos twice).
      - Respect budget + team caps; return best delta.
    """
    import itertools

    df = pred_df.set_index("id", drop=False).copy()
    squad = df.loc[squad_ids].copy()
    current_cost = float(squad["price"].sum())
    budget_cap = current_cost + float(bank)

    base_xi, _ = best_starting_xi(pred_df, squad_ids)
    if base_xi.empty:
        return None
    base_pts = float(base_xi["xPts"].sum())

    pools = {}
    for pos in ["GKP","DEF","MID","FWD"]:
        pools[pos] = df[(df["pos"] == pos) & (~df["id"].isin(squad_ids))].sort_values(
            ["xPts","xPts_per_m"], ascending=[False, False]
        ).head(top_pool_per_pos)

    best = None  # (delta, (out1,out2), (in1,in2), new_pts, new_cost)

    # choose two outgoing players (ordered combination)
    out_pairs = list(itertools.combinations(list(squad["id"].values), 2))
    for out1, out2 in out_pairs:
        row1, row2 = df.loc[out1], df.loc[out2]
        for in1 in pools.get(row1["pos"], []).itertuples(index=False):
            for in2 in pools.get(row2["pos"], []).itertuples(index=False):
                new_cost = current_cost - float(row1.price) - float(row2.price) + float(in1.price) + float(in2.price)
                if new_cost > budget_cap:
                    continue
                # team caps
                temp_ids = [pid for pid in squad_ids if pid not in (out1, out2)] + [int(in1.id), int(in2.id)]
                tmp_df = df.loc[temp_ids]
                team_ok = (tmp_df.groupby("team").size() <= 3).all()
                if not team_ok:
                    continue
                new_xi, _ = best_starting_xi(pred_df, temp_ids)
                if new_xi.empty:
                    continue
                new_pts = float(new_xi["xPts"].sum())
                delta = new_pts - base_pts
                if (best is None) or (delta > best[0]):
                    best = (delta, (int(out1), int(out2)), (int(in1.id), int(in2.id)), new_pts, round(new_cost,1))

    if not best or best[0] <= 0:
        return None
    return {
        "delta": round(best[0], 3),
        "outs": best[1],
        "ins": best[2],
        "new_pts": round(best[3], 3),
        "new_cost": best[4],
        "base_pts": round(base_pts, 3),
    }

