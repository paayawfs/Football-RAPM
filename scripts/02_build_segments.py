"""Build data/processed/segments.parquet from every scraped Understat league-season."""

import logging

import pandas as pd

from rapm import PROCESSED, RAW
from rapm.segments import build_league_season

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

out = []
for base in sorted((RAW / "understat").glob("*/*")):
    if not (base / "matches.parquet").exists():
        continue
    try:
        seg = build_league_season(base)
    except Exception:
        log.exception("%s/%s failed, skipping", base.parent.name, base.name)
        continue
    out.append(seg)
    log.info("%s/%s: %d segments", base.parent.name, base.name, len(seg))

segments = pd.concat(out, ignore_index=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
segments.to_parquet(PROCESSED / "segments.parquet", index=False)
log.info("done: %d segments across %d matches", len(segments), segments.match_id.nunique())
