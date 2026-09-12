"""Understat fetch and parse. Two JSON endpoints, verified 2026-09-11 (PLAN.md section 0)."""

import logging
import time

import pandas as pd
import requests
import requests_cache

from rapm import RAW

log = logging.getLogger(__name__)

BASE = "https://understat.com/main"
HEADERS = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}
LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]
YEARS = range(2014, 2026)
OUT = RAW / "understat"

# ponytail: cache forever; delete cache.sqlite to refetch
OUT.mkdir(parents=True, exist_ok=True)
session = requests_cache.CachedSession(str(OUT / "cache.sqlite"), expire_after=None)


def get(url, tries=8):
    """A six-hour unattended scrape hits the occasional connection reset; without a
    retry here, one blip kills every league-season still to come (as happened on the
    first overnight run). Cached responses replay for free, so a generous budget costs
    nothing on the common path."""
    for i in range(tries):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            break
        except requests.exceptions.RequestException:
            if i == tries - 1:
                raise
            time.sleep(min(2**i, 30))
    if not getattr(r, "from_cache", False):
        time.sleep(1)  # one live request per second
    return r.json()


def _numeric(df):
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError):
            pass
    return df


def league_matches(league, year):
    d = get(f"{BASE}/getLeagueData/{league}/{year}")["dates"]
    m = pd.json_normalize(d)
    m = m.rename(
        columns={
            "id": "match_id", "h.id": "home_id", "h.title": "home", "a.id": "away_id",
            "a.title": "away", "goals.h": "goals_home", "goals.a": "goals_away",
            "xG.h": "xg_home", "xG.a": "xg_away", "isResult": "is_result",
        }
    )
    m = m[["match_id", "datetime", "is_result", "home_id", "home", "away_id", "away",
           "goals_home", "goals_away", "xg_home", "xg_away"]]
    m["league"], m["season"] = league, year
    return _numeric(m)


def match_data(match_id):
    d = get(f"{BASE}/getMatchData/{match_id}")
    rows = []
    for side in "ha":
        roster = d["rosters"][side]
        if not isinstance(roster, dict):
            # Understat sends [] instead of {} for some malformed/void fixtures; without
            # this check the whole league-season crashed on one such match (Bundesliga
            # 2024 lost entirely to it on the first overnight run).
            log.warning("match %s side %s has no roster data: %r", match_id, side, roster)
            continue
        rows += roster.values()
    rosters = pd.DataFrame(rows)
    shots = pd.DataFrame(d["shots"]["h"] + d["shots"]["a"])
    rosters["match_id"] = match_id
    if not shots.empty:
        shots["match_id"] = match_id
    return _numeric(rosters), _numeric(shots)


def scrape(league, year):
    out = OUT / league / str(year)
    if (out / "shots.parquet").exists():
        return
    matches = league_matches(league, year)
    played = matches[matches.is_result]
    rosters, shots = [], []
    for i, mid in enumerate(played.match_id):
        r, s = match_data(mid)
        if not r.empty:
            starters = r[r.position != "Sub"].groupby("h_a").size()
            if not (starters == 11).all():
                log.warning("match %s starters per side: %s", mid, starters.to_dict())
        rosters.append(r)
        shots.append(s)
        if i % 50 == 0:
            log.info("%s %s: %d/%d", league, year, i, len(played))
    out.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(out / "matches.parquet", index=False)
    pd.concat(rosters).to_parquet(out / "rosters.parquet", index=False)
    pd.concat(shots).to_parquet(out / "shots.parquet", index=False)
    log.info("%s %s done: %d matches", league, year, len(played))
