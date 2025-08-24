
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
