# app.py
import streamlit as st
import pandas as pd
import requests
import itertools

from fpl_tool.model import build_player_master, v2_expected_points, add_value_columns

st.set_page_config(page_title="FPL Analytics – Expected Points (per match / total)", layout="wide")
st.title("⚽ FPL Analytics – Expected Points (per match & horizon totals)")
st.caption("Model: v2 (xG/xA per90, fixture adjustment, clean-sheet proxy, saves proxy). Use min-minutes filter to avoid tiny-sample artifacts.")

# --- Data loaders ---
@st.cache_data(ttl=3600)
def load_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    data = r.json()
    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    element_types = pd.DataFrame(data["element_types"])
    return players, teams, element_types


@st.cache_data(ttl=3600)
def load_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    r = requests.get(url)
    return pd.DataFrame(r.json())


players, teams, element_types = load_fpl_data()
fixtures = load_fixtures()

# build player master and run model
pm = build_player_master(players, teams, element_types)

# --- sidebar controls ---
st.sidebar.header("Model & display settings")
horizon = st.sidebar.slider("Fixture horizon (matches)", 1, 10, 5)
rank_metric = st.sidebar.selectbox("Rank by (per match or total)", ["xPts_per_match", "xPts_total"])
min_minutes = st.sidebar.slider("Min historical minutes for leaderboards (0 = no filter)", 0, 2000, 90)
top_n = st.sidebar.number_input("Top N per position", min_value=1, max_value=20, value=10)

# run model
pred = v2_expected_points(pm, fixtures, teams, horizon=horizon)
pred = add_value_columns(pred)

# standard columns: ensure exist
if "price_m" not in pred.columns and "now_cost" in pred.columns:
    pred["price_m"] = pred["now_cost"].astype(float) / 10.0
if "selected_by_percent" not in pred.columns:
    pred["selected_by_percent"] = 0.0
pred["sel_by_%"] = pred["selected_by_percent"].map(lambda x: f"{float(x):.1f}%")

# optional leaderboard filter to exclude tiny-sample players
leaderboard_df = pred.copy()
if min_minutes > 0 and "minutes" in leaderboard_df.columns:
    leaderboard_df = leaderboard_df[leaderboard_df["minutes"].fillna(0) >= min_minutes]


# formatting helper
def fmt_df_for_display(df, cols):
    d = df.copy()
    # price formatting
    if "price_m" in d.columns:
        d["price_m"] = d["price_m"].round(1)
    # round common numeric fields
    for c in ["xAttack_per90", "xAttack_per90_adj", "att_factor", "cs_prob", "xSaves_per_match", "xPts_per_match", "xPts_total", "xPts_per_m"]:
        if c in d.columns:
            d[c] = d[c].round(3)
    final_cols = [c for c in cols if c in d.columns]
    return d[final_cols]


# helper: top by position
def top_by_position(df, score_col, top_n=10, gk_n=3):
    pos_map = {"GKP": gk_n, "DEF": top_n, "MID": top_n, "FWD": top_n}
    out = {}
    for pos, n in pos_map.items():
        out[pos] = df[df["pos"] == pos].sort_values(score_col, ascending=False).head(n)
    return out


# --- Captaincy picks (leaderboards) ---
st.subheader(f"🎯 Captaincy picks (Top by {rank_metric} per position)")
captains = top_by_position(leaderboard_df, rank_metric, top_n=top_n, gk_n=3)
for pos, tbl in captains.items():
    st.markdown(f"**Top {len(tbl)} {pos}s by {rank_metric}**")
    if pos == "GKP":
        cols = ["web_name", "team_name", "pos", "price_m", "sel_by_%", "cs_prob", "xSaves_per_match", rank_metric, "xPts_total"]
    elif pos == "DEF":
        cols = ["web_name", "team_name", "pos", "price_m", "sel_by_%", "xAttack_per90", "att_factor", "cs_prob", rank_metric, "xPts_total"]
    else:
        cols = ["web_name", "team_name", "pos", "price_m", "sel_by_%", "xAttack_per90", "att_factor", rank_metric, "xPts_total"]
    st.dataframe(fmt_df_for_display(tbl, cols).reset_index(drop=True))


# --- Value picks ---
st.subheader("💼 Value picks (Top by xPts_per_m)")
value_score = "xPts_per_m"
values = top_by_position(leaderboard_df, value_score, top_n=top_n, gk_n=3)
for pos, tbl in values.items():
    st.markdown(f"**Top {len(tbl)} {pos}s by {value_score}**")
    if pos == "GKP":
        cols = ["web_name", "team_name", "pos", "price_m", "sel_by_%", "cs_prob", "xSaves_per_match", value_score]
    elif pos == "DEF":
        cols = ["web_name", "team_name", "pos", "price_m", "sel_by_%", "xAttack_per90", "cs_prob", value_score]
    else:
        cols = ["web_name", "team_name", "pos", "price_m", "sel_by_%", "xAttack_per90", value_score]
    st.dataframe(fmt_df_for_display(tbl, cols).reset_index(drop=True))


# --- Analyze my 15-man squad ---
st.subheader("🧩 Analyze My 15-man Squad")
player_options = {}
for r in pred.itertuples(index=False):
    pid = int(r.id)
    name = getattr(r, "web_name", "")
    team = getattr(r, "team_name", "")
    pos = getattr(r, "pos", "")
    price = getattr(r, "price_m", 0.0)
    sel = getattr(r, "selected_by_percent", 0.0)
    player_options[pid] = f"{name} ({team}, {pos}, {price:.1f}m, {sel:.1f}%)"

squad_ids = st.multiselect("Select your 15 players", options=list(player_options.keys()), format_func=lambda x: player_options[x])
bank = st.number_input("Bank (money in the bank, £m)", min_value=0.0, step=0.1)

# helper: valid formations (outfield splits)
VALID_FORMATIONS = {(3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 3, 3), (5, 3, 2), (5, 4, 1)}


def best_xi_optimal(squad_df: pd.DataFrame, score_col="xPts_per_match") -> pd.DataFrame:
    """
    Exhaustive search across 11-player combos (15 choose 11 = 1365) to pick optimal XI
    that satisfies exactly 1 GK and a valid outfield split.
    """
    if squad_df.shape[0] < 11:
        return squad_df.sort_values(score_col, ascending=False).head(11)

    ids = squad_df["id"].tolist()
    meta = squad_df.set_index("id")
    best_score = -1e9
    best_ids = None

    for combo in itertools.combinations(ids, 11):
        combo_df = meta.loc[list(combo)]
        # 1 GK required
        if (combo_df["pos"] == "GKP").sum() != 1:
            continue
        defs = (combo_df["pos"] == "DEF").sum()
        mids = (combo_df["pos"] == "MID").sum()
        fwds = (combo_df["pos"] == "FWD").sum()
        if (defs + mids + fwds) != 10:
            continue
        if (defs, mids, fwds) not in VALID_FORMATIONS:
            continue
        score = combo_df[score_col].astype(float).sum()
        if score > best_score:
            best_score = score
            best_ids = list(combo)

    if best_ids is None:
        # fallback greedy
        gk = squad_df[squad_df["pos"] == "GKP"].sort_values(score_col, ascending=False).head(1)
        out = squad_df[squad_df["pos"] != "GKP"].sort_values(score_col, ascending=False).head(10)
        return pd.concat([gk, out])
    else:
        return meta.loc[best_ids].reset_index(drop=False)


if len(squad_ids) == 15:
    squad_df = pred[pred["id"].isin(squad_ids)].copy()

    # compute best XI (optimal across valid formations) using chosen rank metric
    best_xi = best_xi_optimal(squad_df, score_col=rank_metric)

    st.markdown("### ✅ Best XI (optimal across valid formations):")
    show_cols = ["web_name", "pos", "team_name", "price_m", "sel_by_%", "xPts_per_match", "xPts_total"]
    st.dataframe(fmt_df_for_display(best_xi.sort_values(rank_metric, ascending=False), show_cols).reset_index(drop=True))

    # captain / vice
    ranked = best_xi.sort_values(rank_metric, ascending=False)
    captain = ranked.iloc[0]["web_name"]
    vice = ranked.iloc[1]["web_name"] if ranked.shape[0] > 1 else ""
    st.success(f"⭐ Recommended Captain: **{captain}** | Vice Captain: **{vice}**")

    # subs
    subs = squad_df[~squad_df["id"].isin(best_xi["id"])].sort_values(rank_metric, ascending=False)
    st.markdown("### 🪑 Subs (bench, sorted by chosen metric):")
    st.dataframe(fmt_df_for_display(subs, show_cols).reset_index(drop=True))

    # transfer suggester (single out -> in)
    st.markdown("---")
    st.subheader("🔁 Suggested transfers (single-out replacement)")
    current_total = best_xi["xPts_total"].sum() if "xPts_total" in best_xi.columns else best_xi[rank_metric].sum()
    candidates = []
    for out_id in squad_ids:
        out_player = pred[pred["id"] == out_id].iloc[0]
        # budget in tenths of millions (API uses tenths)
        budget_available = bank * 10 + out_player.get("now_cost", 0)
        # same position, not in squad, within budget
        pool = pred[(pred["pos"] == out_player["pos"]) & (~pred["id"].isin(squad_ids)) & (pred["now_cost"] <= budget_available)]
        if pool.empty:
            continue
        best_in = pool.sort_values("xPts_total", ascending=False).iloc[0]
        new_squad_ids = [pid for pid in squad_ids if pid != out_id] + [best_in["id"]]
        new_squad_df = pred[pred["id"].isin(new_squad_ids)].copy()
        new_best = best_xi_optimal(new_squad_df, score_col="xPts_total")
        new_total = new_best["xPts_total"].sum()
        gain = new_total - current_total
        if gain > 0:
            candidates.append((gain, out_player, best_in, new_total))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

    if candidates:
        st.markdown("#### 💡 Top 2 suggested transfers:")
        for gain, out_p, in_p, new_total in candidates[:2]:
            st.success(f"**{out_p['web_name']} ➝ {in_p['web_name']}**  (+{gain:.2f} xPts_total)")

        st.markdown("#### 🎯 Choose a suggested OUT (dropdown)")
        choice = st.selectbox("Choose an OUT from suggestions", [c[1]["web_name"] for c in candidates])
        if choice:
            selected = next(c for c in candidates if c[1]["web_name"] == choice)
            gain, out_p, in_p, new_total = selected
            st.info(f"Best replacement for **{out_p['web_name']}** ➝ **{in_p['web_name']}** (+{gain:.2f} xPts_total). New XI total = {new_total:.2f}")
    else:
        st.info("No beneficial single-player replacement found within constraints.")
else:
    st.info("Please select exactly 15 players to analyze transfers.")
