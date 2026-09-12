"""Usage: python scripts/01_fetch_statsbomb.py [COMPETITION_ID SEASON_ID]. No args fetches everything.

Each event table runs several hundred MB in memory before writing, and holding that for
season after season in one long-lived process pushed total memory high enough that the
whole batch got OS-killed partway through. One subprocess per season returns all of that
memory when it exits, capping peak usage regardless of how many seasons remain.
"""

import logging
import subprocess
import sys

from rapm import statsbomb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

if len(sys.argv) == 3:
    statsbomb.fetch(int(sys.argv[1]), int(sys.argv[2]))
else:
    for cid, sid in statsbomb.SEASONS:
        r = subprocess.run([sys.executable, __file__, str(cid), str(sid)], check=False)
        if r.returncode != 0:
            log.error("%s %s failed (exit %d), skipping", cid, sid, r.returncode)
