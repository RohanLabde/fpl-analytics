import streamlit as st
import pandas as pd

# --- absolute imports (Render-safe) ---
from fpl_tool.data import load_all, next_deadline_ist
from fpl_tool.features import build_player_master, fixture_softness
from fpl_tool.model import baseline_expected_points
from fpl_tool.optimizer import (
    pick_squad_greedy,
    best_starting_xi,
    suggest_captain,
    best_single_transfer,
    best_double_transfer,
)

st.set_page_config(page_title="FPL Analytics – Render Free", layout="wide")

st.title("⚽ FPL Analytics – Fast Decisions (Free)")
st.caption("Data: Official Fantasy Premier League API. Uses form, minutes, and fixture softness.")

# -----------------------
# Load + prep data
# -----------------------
with st.spinner("Loading latest FPL data..."):
    data = load_all()

players = data["players"]
teams = data["teams"]
events = data["events"]
fixtures = data["fixtures"]

deadline = next_deadline_ist(events)
if deadline:
    st.info(f"⏳ Next deadline (IST): **{deadline.strftime('%a, %d %b %Y %H:%M')}**")

pm = build_player_master(players, teams, data["positions"])
soft = fixture_softness(fixtures, teams, horizon=3)

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Model Settings")
horizon = st.sidebar.slider("Fixture horizon (matches)", 1, 5, 3)
alpha = st.sidebar.slider("Weight: Form", 0.0, 1.5, 0.7, 0.05)
beta  = st.sidebar.slider("Weight: Minutes", 0.0, 1.0, 0.2, 0.05)
gamma = st.sidebar.slider("Weight: Fixture softness", 0.0, 1.0, 0.1, 0.05)
budget = st.sidebar.slider("Budget", 90.0, 100.0, 100.0, 0.5)

pred = baseline_expected_points(pm, events, soft, horizon=horizon, alpha=alpha, beta=beta, gamma=gamma)

# -----------------------
# Captaincy / Value tables
# -----------------------
st.subheader("🎯 Captaincy picks (Top 15 by xPts)")
if not pred.empty:
    cols = ["web_name","pos","name","price","sel","xPts","xPts_per_m"]
    st.dataframe(pred[cols].head(15).reset_index(drop=True))
else:
    st.warning("No prediction data available yet. Please try refreshing.")

st.subheader("💼 Value picks (Top 15 by xPts per million)")
if not pred.empty:
    cols = ["web_name","pos","name","price","sel","xPts","xPts_per_m"]
    st.dataframe(pred[cols].sort_values("xPts_per_m", ascending=False).head(15).reset_index(drop=True))

# -----------------------
# 15-man greedy optimizer
# -----------------------
st.subheader("🧠 Squad Optimizer (15-man, greedy)")
if st.button("Build 15-man Squad"):
    if pred.empty:
        st.error("Predictions are empty; cannot build squad.")
    else:
        squad, xp, cost = pick_squad_greedy(pred, budget=budget)
        st.success(f"Selected 15 → Estimated xPts = {xp:.2f}, Cost = {cost:.1f}")
        st.dataframe(squad[["web_name","pos","name","price","xPts","xPts_per_m"]])

# -----------------------
# Analyze My 15-man Squad
# -----------------------
st.subheader("🧩 Analyze My 15-man Squad")
st.caption("Pick your current 15 players, set your bank (FPL money in the bank), and we’ll pick a best XI, captain/vice, and suggest transfers.")

if not pred.empty:
    # Build label map keyed by player ID (avoids collisions). Use iterrows() to access by column names.
    def label_for_row(row: pd.Series) -> str:
        return f"{row['web_name']} ({row['name']}, {row['pos']}, {row['price']:.1f})"

    opts = pred[["id","web_name","name","pos","price"]].copy()
    label_map = {int(row["id"]): label_for_row(row) for _, row in opts.iterrows()}  # <-- use row['id']

    # Multiselect over IDs; show labels via format_func
    squad_ids = st.multiselect(
        "Select exactly 15 players from your current squad",
        options=list(label_map.keys()),
        format_func=lambda pid: label_map.get(int(pid), str(pid)),
        max_selections=15
    )

    # Show current squad cost (only when exactly 15 are picked)
    current_cost = float(pred.set_index("id").loc[squad_ids]["price"].sum()) if len(squad_ids) == 15 else 0.0
    if current_cost and len(squad_ids) == 15:
        st.caption(f"Approx current squad cost: **{current_cost:.1f}**")

    bank = st.number_input(
        "Bank (money in the bank)",
        min_value=0.0, max_value=20.0, value=0.0, step=0.1,
        help="Enter your available funds. We assume selling price ≈ current price for now."
    )

    if st.button("Analyze my squad"):
        if len(squad_ids) != 15:
            st.error("Please select exactly 15 players.")
        else:
            xi, bench = best_starting_xi(pred, squad_ids)
            if xi.empty:
                st.error("Could not build a valid XI. Ensure you have at least 1 GKP, 3 DEF, 2 MID, 1 FWD.")
            else:
                cap, vc = suggest_captain(xi)
                st.success(f"✅ Best XI total xPts: {xi['xPts'].sum():.2f}")
                st.write("**Best XI (sorted by xPts):**")
                st.dataframe(xi[["web_name","pos","name","price","xPts"]].reset_index(drop=True))
                st.write("**Bench (by xPts):**")
                st.dataframe(bench[["web_name","pos","name","price","xPts"]].reset_index(drop=True))

                if cap is not None:
                    st.info(f"🧢 **Captain:** {cap['web_name']}  |  🎖️ **Vice:** {vc['web_name'] if vc is not None else '—'}")

                st.markdown("---")
                st.markdown("### 🔁 Transfer suggestions")

                one = best_single_transfer(pred, squad_ids, bank)
                if one:
                    out_p = pred.set_index("id").loc[one["out_id"]]
                    in_p  = pred.set_index("id").loc[one["in_id"]]
                    st.write(
                        f"**Best 1 transfer**:  {out_p['web_name']} ➜ {in_p['web_name']}  "
                        f"|  ΔxPts: **+{one['delta']:.2f}**  | New XI xPts: {one['new_pts']:.2f}"
                    )
                    st.caption(
                        f"Cost: from {out_p['price']:.1f} to {in_p['price']:.1f} "
                        f"(new squad cost ≈ {one['new_cost']:.1f}, bank used ≤ {bank:.1f})"
                    )
                else:
                    st.warning("No positive 1-transfer upgrade found within your bank and team limits.")

                try_two = st.checkbox("Also try 2-transfers (slower)", value=False)
                if try_two:
                    two = best_double_transfer(pred, squad_ids, bank, top_pool_per_pos=25)
                    if two:
                        out1, out2 = two["outs"]; in1, in2 = two["ins"]
                        o1 = pred.set_index("id").loc[out1]; o2 = pred.set_index("id").loc[out2]
                        i1 = pred.set_index("id").loc[in1]; i2 = pred.set_index("id").loc[in2]
                        st.write(
                            f"**Best 2 transfers**:  {o1['web_name']} & {o2['web_name']} ➜ "
                            f"{i1['web_name']} & {i2['web_name']}  |  ΔxPts: **+{two['delta']:.2f}**  "
                            f"| New XI xPts: {two['new_pts']:.2f}"
                        )
                        st.caption(f"New squad cost ≈ {two['new_cost']:.1f} (bank used ≤ {bank:.1f})")
                    else:
                        st.warning("No positive 2-transfer upgrade found within your bank/team caps (with the small search pool).")
else:
    st.info("Predictions table is empty. Reload the app.")
