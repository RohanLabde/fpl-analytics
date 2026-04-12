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
def run_model(players, teams, element_types, fixtures, horizon, fw, bw):
    pm = build_player_master(players, teams, element_types)

    pred = v5_expected_points(
        pm.copy(),
        fixtures.copy(),
        teams.copy(),
        horizon=horizon,
        form_weight=fw,
        bonus_weight=bw
    )

    # CLEAN DATA
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
def build_optimal_squad(pred, budget=100):
    df = pred.sort_values("xPts_total", ascending=False)

    squad, cost = [], 0
    team_count = {}
    limits = {"GKP":2,"DEF":5,"MID":5,"FWD":3}
    counts = {"GKP":0,"DEF":0,"MID":0,"FWD":0}

    for _, p in df.iterrows():
        if counts[p.pos] >= limits[p.pos]: continue
        if team_count.get(p.team_name,0) >= 3: continue
        if cost + p.price_m > budget: continue

        squad.append(p)
        counts[p.pos]+=1
        team_count[p.team_name]=team_count.get(p.team_name,0)+1
        cost += p.price_m

        if sum(counts.values())==15: break

    return pd.DataFrame(squad), cost


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
st.title("⚽ FPL AI – Full System")

players, teams, et, fixtures = load_data()

h = st.sidebar.slider("Horizon",1,10,5)
fw = st.sidebar.slider("Form Weight",0.0,1.0,0.3)
bw = st.sidebar.slider("Bonus Weight",0.0,0.5,0.2)

pred = run_model(players, teams, et, fixtures, h, fw, bw)

# SESSION STATE
if "squad" not in st.session_state:
    st.session_state.squad = None

tabs = st.tabs(["Dashboard","Explorer","Decisions","Transfers","Squad Builder","XI Optimizer"])


# -----------------------
# DASHBOARD
# -----------------------
with tabs[0]:
    st.write("Model ready. Use other tabs.")


# -----------------------
# EXPLORER
# -----------------------
with tabs[1]:
    for pos in ["GKP","DEF","MID","FWD"]:
        st.subheader(pos)
        st.dataframe(pred[pred.pos==pos].sort_values("xPts_total",ascending=False).head(10))


# -----------------------
# DECISIONS
# -----------------------
with tabs[2]:
    df = pred[pred.minutes>300]

    st.subheader("Captain Picks")
    st.dataframe(df.sort_values("xPts_per_match",ascending=False).head(5))

    st.subheader("Differentials")
    st.dataframe(df[df.selected_by_percent<10].head(5))


# -----------------------
# SQUAD BUILDER
# -----------------------
with tabs[4]:
    budget = st.slider("Budget",80,120,100)

    if st.button("Build Squad"):
        squad, cost = build_optimal_squad(pred, budget)

        if len(squad)==15:
            st.session_state.squad = squad
            st.success(f"Squad built (£{round(cost,1)}m)")
            st.dataframe(squad)
        else:
            st.error("Failed to build squad")


# -----------------------
# TRANSFERS
# -----------------------
with tabs[3]:
    if st.session_state.squad is None:
        st.warning("Build squad first")
    else:
        names = st.session_state.squad.web_name.tolist()
        st.dataframe(suggest_transfers(names, pred))


# -----------------------
# XI OPTIMIZER
# -----------------------
with tabs[5]:
    if st.session_state.squad is None:
        st.warning("Build squad first")
    else:
        xi, score = optimize_xi(st.session_state.squad)
        cap, vc = pick_captains(xi)

        st.success(f"Best XI Score: {round(score,2)}")
        st.dataframe(xi)

        st.write("Captain:", cap.web_name)
        st.write("Vice Captain:", vc.web_name)
