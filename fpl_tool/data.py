import requests
import pandas as pd
from datetime import datetime, timedelta

FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES = "https://fantasy.premierleague.com/api/fixtures/"

def load_all():
    """
    Load all necessary data from the official FPL API.
    Returns dictionary with keys: elements, teams, events, element_types, fixtures.
    """
    # bootstrap-static
    resp = requests.get(FPL_BOOTSTRAP)
    resp.raise_for_status()
    data = resp.json()

    # fixtures
    fix = requests.get(FPL_FIXTURES)
    fix.raise_for_status()
    fixtures = fix.json()

    # normalize structure: keep official keys
    return {
        "elements": pd.DataFrame(data["elements"]),          # players
        "teams": pd.DataFrame(data["teams"]),                # teams
        "events": pd.DataFrame(data["events"]),              # gameweeks
        "element_types": pd.DataFrame(data["element_types"]),# positions (GKP, DEF, MID, FWD)
        "fixtures": pd.DataFrame(fixtures)                   # fixture data
    }


def next_deadline_ist(events: pd.DataFrame):
    """
    Get next deadline (IST timezone).
    """
    upcoming = events[events["is_next"] == True]
    if upcoming.empty:
        return None
    deadline_str = upcoming.iloc[0]["deadline_time"]
    deadline_dt = datetime.fromisoformat(deadline_str.replace("Z", "+00:00")) + timedelta(hours=5, minutes=30)
    return deadline_dt
