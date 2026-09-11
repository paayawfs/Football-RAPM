"""Usage: python scripts/01_fetch_statsbomb.py [COMPETITION_ID SEASON_ID]. No args fetches everything."""

import logging
import sys

from rapm import statsbomb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

if len(sys.argv) == 3:
    statsbomb.fetch(int(sys.argv[1]), int(sys.argv[2]))
else:
    for cid, sid in statsbomb.SEASONS:
        statsbomb.fetch(cid, sid)
