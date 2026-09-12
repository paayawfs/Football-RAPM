"""StatsBomb open data fetch. Full-league seasons only (PLAN.md section 0)."""

import json
import logging
import time
import warnings

import pandas as pd
import requests
import requests_cache
from statsbombpy import sb

from rapm import RAW

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="credentials were not supplied")

# (competition_id, season_id): label
SEASONS = {
    (2, 27): "EPL_2015", (11, 27): "LaLiga_2015", (12, 27): "SerieA_2015", (7, 27): "Ligue1_2015",
    (37, 4): "WSL_2018", (37, 42): "WSL_2019", (37, 90): "WSL_2020", (37, 281): "WSL_2023",
    (49, 107): "NWSL_2023", (135, 281): "FrauenBL_2023", (182, 281): "LigaF_2023",
    (131, 281): "SerieAW_2023", (1238, 108): "ISL_2021",
}
OUT = RAW / "statsbomb"

# statsbombpy calls plain requests.get() internally with no session and no cache.
# raw.githubusercontent.com's shared backend pool returns transient 503s under sustained
# sequential load, and a crash partway through fetch() otherwise discards every match
# already downloaded. install_cache() patches requests globally, so a rerun after a
# crash replays finished matches from disk instantly instead of re-hitting the network.
OUT.mkdir(parents=True, exist_ok=True)
requests_cache.install_cache(str(OUT / "http_cache.sqlite"), expire_after=None)


def _retry(fn, tries=10, **kw):
    """Backstop for the rare persistent outage that outlasts a few retries; cached
    successes make a generous budget here free on any subsequent run."""
    time.sleep(0.2)
    for i in range(tries):
        try:
            return fn(**kw)
        except requests.HTTPError:
            if i == tries - 1:
                raise
            time.sleep(min(2**i, 30))


def _tidy_object_columns(df):
    """Parquet needs one arrow type per column. Two StatsBomb quirks break that:
    event qualifier columns mix dicts and None, and co-manager matches give id
    columns (home_manager_id, home_manager_country_id, ...) a comma-joined string
    like '4711, 3626' instead of an int. Store dicts as JSON; any other mixed-type
    column (id columns are identifiers, never used arithmetically) becomes strings."""
    for c in df.columns:
        col = df[c]
        types = {type(v) for v in col if not (v is None or (isinstance(v, float) and pd.isna(v)))}
        if types == {dict}:
            df[c] = col.map(lambda v: json.dumps(v) if isinstance(v, dict) else v)
        elif len(types) > 1:
            df[c] = col.map(lambda v: v if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    return df


def lineups(match_id):
    rows, spells, cards = [], [], []
    for team, df in _retry(sb.lineups, match_id=match_id).items():
        for _, p in df.iterrows():
            base = {"match_id": match_id, "team": team, "player_id": p.player_id,
                    "player_name": p.player_name, "jersey_number": p.jersey_number}
            rows.append(base)
            spells += [{**base, **s} for s in p.positions]
            cards += [{**base, **c} for c in p.cards]
    return pd.DataFrame(rows), pd.DataFrame(spells), pd.DataFrame(cards)


def fetch(cid, sid):
    out = OUT / SEASONS[(cid, sid)]
    if (out / "events.parquet").exists():
        return
    matches = _retry(sb.matches, competition_id=cid, season_id=sid)
    players, spells, cards, events = [], [], [], []
    for i, mid in enumerate(matches.match_id):
        p, s, c = lineups(mid)
        players.append(p), spells.append(s), cards.append(c)
        events.append(_retry(sb.events, match_id=mid))
        if i % 25 == 0:
            log.info("%s: %d/%d", out.name, i, len(matches))
    out.mkdir(parents=True, exist_ok=True)
    _tidy_object_columns(matches).to_parquet(out / "matches.parquet", index=False)
    pd.concat(players).to_parquet(out / "lineups.parquet", index=False)
    pd.concat(spells).to_parquet(out / "positions.parquet", index=False)
    pd.concat(cards).to_parquet(out / "cards.parquet", index=False)
    _tidy_object_columns(pd.concat(events, ignore_index=True)).to_parquet(out / "events.parquet", index=False)
    log.info("%s done: %d matches", out.name, len(matches))
