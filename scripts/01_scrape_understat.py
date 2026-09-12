"""Usage: python scripts/01_scrape_understat.py [LEAGUE YEAR]. No args scrapes everything."""

import logging
import sys

from rapm import understat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

if len(sys.argv) == 3:
    understat.scrape(sys.argv[1], int(sys.argv[2]))
else:
    for league in understat.LEAGUES:
        for year in understat.YEARS:
            try:
                understat.scrape(league, year)
            except Exception:
                log.exception("%s %s failed, skipping", league, year)
