"""Section 5.1 (descriptive) and 5.2 (identifiability) metrics for every league-season.

Writes data/processed/table1_descriptive.parquet and table2_identifiability.parquet.
"""

import json
import logging

import pandas as pd

from rapm import PROCESSED, RAW
from rapm.descriptive import descriptive_stats
from rapm.identify import identifiability_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

seg_all = pd.read_parquet(PROCESSED / "segments.parquet")
n_blocks = seg_all[["league", "season"]].drop_duplicates().shape[0]

table1, table2 = [], []
for (league, season), seg in seg_all.groupby(["league", "season"]):
    era = seg.era.mode().iloc[0]  # dominant rule; 2019-20 restart seasons straddle both
    try:
        rosters = pd.read_parquet(RAW / "understat" / league / str(int(season)) / "rosters.parquet")
        rosters = rosters[rosters.match_id.isin(seg.match_id)]
        d = descriptive_stats(seg, rosters)
    except Exception:
        log.exception("%s %s: descriptive_stats failed, skipping", league, season)
    else:
        d["subs_dist"] = json.dumps(d["subs_dist"])
        table1.append({"league": league, "season": season, "era": era, **d})

    try:
        m = identifiability_metrics(seg)
    except Exception:
        log.exception("%s %s: identifiability_metrics failed, skipping", league, season)
    else:
        table2.append(dict(
            league=league, season=season, era=era, n_players=m.n_players,
            condition_number=m.condition_number, condition_number_trimmed=m.condition_number_trimmed,
            effective_rank=m.effective_rank, variance_share_bottom_10pct=m.variance_share_bottom_10pct,
            lam=m.lam, edf=m.edf,
            id_share_median=m.identification_share.median(),
            id_share_q1=m.identification_share.quantile(0.25),
            id_share_q3=m.identification_share.quantile(0.75),
            id_share_gt_half=(m.identification_share > 0.5).mean(),
        ))
    log.info("%s %s done", league, season)

pd.DataFrame(table1).to_parquet(PROCESSED / "table1_descriptive.parquet", index=False)
pd.DataFrame(table2).to_parquet(PROCESSED / "table2_identifiability.parquet", index=False)
log.info("done: %d/%d league-seasons in table1, %d/%d in table2",
          len(table1), n_blocks, len(table2), n_blocks)
