import pandas as pd
import requests
import streamlit as st
from itertools import combinations

from fpl_tool.model import build_player_master, v5_expected_points


# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data(ttl=3600)
def load_data():
    base = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    fixtures = pd.DataFrame(requests.get("https://fantasy.premierleague.com/api/fixtures/").json())

    return (
        pd.DataFrame(base["elements"]),
        pd.DataFrame(base["teams"]),
        pd.DataFrame(base["element_types"]),
        fixtures
    )


# -----------------------
# MODEL
# -----------------------
@st.cache_data(ttl=3600)
def run_model(players, teams, et, fixtures, horizon, fw, bw):
    pm = build_player_master(players, teams, et)

    pred = v5_expected_points(
        pm.copy(),
        fixtures.copy(),
        teams.copy(),
        horizon=horizon,
        form_weight=fw,
        bonus_weight=bw
    )

    pred["selected_by_percent"] = pd.to_numeric(pred["selected_by_percent"], errors="coerce").fillna(0)

    return pred[[
        "web_name", "team_name", "pos",
        "price_m", "minutes",
        "xPts_total", "xPts_per_match",
        "selected_by_percent"
    ]].copy()


# -----------------------
# OPTIMIZERS
# -----------------------
def optimize_xi(squad):
    best_score, best_team = -999, None

    gk = squad[squad.pos=="GKP"]
    d = squad[squad.pos=="DEF"]
    m = squad[squad.pos=="MID"]
    f = squad[squad.pos=="FWD"]

    formations = [(3,4,3),(3,5,2),(4,4,2),(4,3,3),(5,3,2)]

    for D,M,F in formations:
        if len(d)<D or len(m)<M or len(f)<F: continue

        for g in combinations(gk.index,1):
            for d_ in combinations(d.index,D):
                for m_ in combinations(m.index,M):
                    for f_ in combinations(f.index,F):

                        team = squad.loc[list(g)+list(d_)+list(m_)+list(f_)]
                        score = team.xPts_per_match.sum()

                        if score > best_score:
                            best_score = score
                            best_team = team

    return best_team, best_score


def pick_captains(xi):
    xi = xi.sort_values("xPts_per_match", ascending=False)
    return xi.iloc[0], xi.iloc[1]


def suggest_transfers(squad_names, pred):
    squad = pred[pred.web_name.isin(squad_names)]
    pool = pred[~pred.web_name.isin(squad_names)]

    weakest = squad.sort_values("xPts_total").head(2)
    pool = pool.sort_values("xPts_total", ascending=False).head(30)

    suggestions = []

    for _, w in weakest.iterrows():
        replacements = pool[pool.pos==w.pos].head(5)

        for _, r in replacements.iterrows():
            gain = r.xPts_total - w.xPts_total
            if gain > 0:
                suggestions.append({
                    "OUT": w.web_name,
                    "IN": r.web_name,
                    "GAIN": round(gain,2)
                })

    return pd.DataFrame(suggestions).sort_values("GAIN", ascending=False)


# -----------------------
# UI
# -----------------------
st.set_page_config(layout="wide")
st.title("⚽ FPL AI – Smart Assistant")

players, teams, et, fixtures = load_data()

h = st.sidebar.slider("Horizon",1,10,5)
fw = st.sidebar.slider("Form Weight",0.0,1.0,0.3)
bw = st.sidebar.slider("Bonus Weight",0.0,0.5,0.2)

pred = run_model(players, teams, et, fixtures, h, fw, bw)

# -----------------------
# SESSION STATE (MANUAL SQUAD)
# -----------------------
if "user_squad" not in st.session_state:
    st.session_state.user_squad = []


# -----------------------
# GLOBAL SQUAD SELECTOR
# -----------------------
st.subheader("🧱 Select Your Squad")

player_names = sorted(pred["web_name"].unique())

selected_players = st.multiselect(
    "Select 15 players",
    options=player_names,
    default=st.session_state.user_squad
)

st.session_state.user_squad = selected_players

st.caption(f"Selected: {len(selected_players)} / 15")


# -----------------------
# TABS
# -----------------------
tabs = st.tabs(["Dashboard","Explorer","Decisions","Transfers","XI Optimizer"])


# -----------------------
# DASHBOARD
# -----------------------
with tabs[0]:
    st.header("📊 Dashboard")

    df = pred.copy()

    # 🔥 Top players overall
    st.subheader("🔥 Top Players (Overall)")

    st.dataframe(
        df.sort_values("xPts_total", ascending=False)
        .head(10)[["web_name", "team_name", "pos", "xPts_total"]]
    )

    # 🔥 Best per position
    st.subheader("📌 Best Players by Position")

    for pos in ["GKP", "DEF", "MID", "FWD"]:
        st.markdown(f"### {pos}")

        st.dataframe(
            df[df["pos"] == pos]
            .sort_values("xPts_total", ascending=False)
            .head(5)[["web_name", "team_name", "xPts_total"]]
        )

# -----------------------
# EXPLORER
# -----------------------
with tabs[1]:
    for pos in ["GKP","DEF","MID","FWD"]:
        st.subheader(pos)
        st.dataframe(
            pred[pred.pos==pos]
            .sort_values("xPts_total",ascending=False)
            .head(10)
        )


# -----------------------
# DECISIONS
# -----------------------
with tabs[2]:
    st.header("🧠 Smart Decisions")

    df = pred[pred["minutes"] > 300]

    # ✅ Captain Picks (MID + FWD ONLY)
    st.subheader("👑 Captain Picks (MID/FWD)")

    captains = df[df["pos"].isin(["MID", "FWD"])] \
        .sort_values("xPts_per_match", ascending=False) \
        .head(5)

    st.dataframe(captains)

    # ✅ Differentials (exclude GK)
    st.subheader("💎 Differentials (<10% owned)")

    differentials = df[
        (df["selected_by_percent"] < 10) &
        (df["pos"] != "GKP")
    ].sort_values("xPts_per_match", ascending=False).head(5)

    st.dataframe(differentials)

    # ✅ Safe Picks
    st.subheader("🛡 Safe Picks (>20% owned)")

    safe = df[df["selected_by_percent"] > 20] \
        .sort_values("xPts_per_match", ascending=False).head(5)

    st.dataframe(safe)

    # ✅ Goalkeepers (separate)
    st.subheader("🧤 Best Goalkeepers")

    gk = df[df["pos"] == "GKP"] \
        .sort_values("xPts_per_match", ascending=False).head(5)

    st.dataframe(gk)

# -----------------------
# TRANSFERS
# -----------------------
with tabs[3]:
    st.header("🔄 Transfer Suggestions")

    if len(st.session_state.user_squad) != 15:
        st.warning("Select exactly 15 players above")
    else:
        result = suggest_transfers(st.session_state.user_squad, pred)

        if result.empty:
            st.warning("No suggestions found")
        else:
            st.dataframe(result)


# -----------------------
# XI OPTIMIZER
# -----------------------
with tabs[4]:
    st.header("⚽ Best Starting XI")

    if len(st.session_state.user_squad) != 15:
        st.warning("Select exactly 15 players above")
    else:
        squad_df = pred[pred.web_name.isin(st.session_state.user_squad)]

        xi, score = optimize_xi(squad_df)
        cap, vc = pick_captains(xi)

        st.success(f"Best XI Score: {round(score,2)}")

        st.dataframe(xi[["web_name","pos","team_name","xPts_per_match"]])

        st.write(f"👑 Captain: {cap.web_name}")
        st.write(f"🥈 Vice Captain: {vc.web_name}")
