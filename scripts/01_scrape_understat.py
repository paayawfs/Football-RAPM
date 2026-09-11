"""Usage: python scripts/01_scrape_understat.py [LEAGUE YEAR]. No args scrapes everything."""

import logging
import sys

from rapm import understat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

if len(sys.argv) == 3:
    understat.scrape(sys.argv[1], int(sys.argv[2]))
else:
    for league in understat.LEAGUES:
        for year in understat.YEARS:
            understat.scrape(league, year)
