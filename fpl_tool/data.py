
import time, requests, pandas as pd
from datetime import datetime
from dateutil import tz

FPL_BASE = "https://fantasy.premierleague.com/api"

def _get(url, retries=3, sleep=0.8, timeout=25):
    last_exc = None
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.ok:
                return r.json()
        except Exception as e:
            last_exc = e
        time.sleep(sleep)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to GET {url}")

def fetch_bootstrap():
    return _get(f"{FPL_BASE}/bootstrap-static/")

def fetch_fixtures():
    return _get(f"{FPL_BASE}/fixtures/")

def fetch_player_history(player_id: int):
    return _get(f"{FPL_BASE}/element-summary/{player_id}/")

def to_df_bootstrap(bs):
    players = pd.DataFrame(bs.get("elements", []))
    teams = pd.DataFrame(bs.get("teams", []))
    events = pd.DataFrame(bs.get("events", []))
    positions = pd.DataFrame(bs.get("element_types", []))
    return players, teams, events, positions

def next_deadline_ist(events_df):
    if events_df is None or events_df.empty:
        return None
    events_df = events_df.copy()
    if "deadline_time" not in events_df.columns:
        return None
    events_df["deadline_time"] = pd.to_datetime(events_df["deadline_time"], utc=True, errors="coerce")
    upcoming = events_df[events_df["is_next"] == True] if "is_next" in events_df.columns else pd.DataFrame()
    if upcoming is None or upcoming.empty:
        upcoming = events_df[events_df["finished"] == False] if "finished" in events_df.columns else pd.DataFrame()
    if upcoming is None or upcoming.empty:
        return None
    ist = tz.gettz("Asia/Kolkata")
    return upcoming.iloc[0]["deadline_time"].astimezone(ist)

def load_all():
    bs = fetch_bootstrap()
    fixtures = pd.DataFrame(fetch_fixtures())
    players, teams, events, positions = to_df_bootstrap(bs)
    return {"players": players, "teams": teams, "events": events, "positions": positions, "fixtures": fixtures}
