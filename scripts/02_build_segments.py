"""Build data/processed/segments.parquet from every scraped Understat league-season."""

import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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

# pyarrow's type inference for the home/away player-id list columns is not reliable
# at this row count -- observed silently widening list<int64> to list<double> once
# every league-season is concatenated together, though not on any single block or a
# two-block test. Force the correct type explicitly rather than depend on inference.
table = pa.Table.from_pandas(segments, preserve_index=False)
for col in ["home_players", "away_players"]:
    i = table.schema.get_field_index(col)
    table = table.set_column(i, col, table.column(col).cast(pa.list_(pa.int64())))

PROCESSED.mkdir(parents=True, exist_ok=True)
pq.write_table(table, PROCESSED / "segments.parquet")
log.info("done: %d segments across %d matches", len(segments), segments.match_id.nunique())
