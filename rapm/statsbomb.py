"""StatsBomb open data fetch. Full-league seasons only (PLAN.md section 0)."""

import json
import logging
import time
import warnings

import pandas as pd
import requests
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


def _retry(fn, tries=6, **kw):
    """raw.githubusercontent.com returns transient 503s; back off and retry."""
    for i in range(tries):
        try:
            return fn(**kw)
        except requests.HTTPError:
            if i == tries - 1:
                raise
            time.sleep(2**i)


def _stringify_dicts(df):
    """Parquet cannot hold columns that mix dicts and None across matches; store dicts as JSON."""
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, dict)).any():
            df[c] = df[c].map(lambda v: json.dumps(v) if isinstance(v, dict) else v)
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
    matches = sb.matches(competition_id=cid, season_id=sid)
    players, spells, cards, events = [], [], [], []
    for i, mid in enumerate(matches.match_id):
        p, s, c = lineups(mid)
        players.append(p), spells.append(s), cards.append(c)
        events.append(_retry(sb.events, match_id=mid))
        if i % 25 == 0:
            log.info("%s: %d/%d", out.name, i, len(matches))
    out.mkdir(parents=True, exist_ok=True)
    _stringify_dicts(matches).to_parquet(out / "matches.parquet", index=False)
    pd.concat(players).to_parquet(out / "lineups.parquet", index=False)
    pd.concat(spells).to_parquet(out / "positions.parquet", index=False)
    pd.concat(cards).to_parquet(out / "cards.parquet", index=False)
    _stringify_dicts(pd.concat(events, ignore_index=True)).to_parquet(out / "events.parquet", index=False)
    log.info("%s done: %d matches", out.name, len(matches))
