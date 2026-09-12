"""PLAN.md section 5.4, restart experiment: for each of the four leagues that
restarted in 2020, compare the final N pre-lockdown matches with the N restart
matches -- same squads, same season, the cleanest within-season switch from three
to five subs. Lambda is shared (fit once on the pooled pre+restart segments for
that league) so a lambda difference between the two windows can't drive the
comparison, matching section 5.3's stated principle for any era comparison.
"""

import logging

import pandas as pd

from rapm import PROCESSED
from rapm.compare import METRICS, independent_bootstrap_diff, metrics_row
from rapm.design import build_net_design
from rapm.identify import identifiability_metrics
from rapm.ridge import select_lambda
from rapm.segments import RESTART_DATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

RESTART_LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A"]  # Ligue_1 2019-20 was abandoned
N_DRAWS = 200

seg_all = pd.read_parquet(PROCESSED / "segments.parquet")

rows = []
for league in RESTART_LEAGUES:
    cutoff = RESTART_DATE[league]
    block = seg_all[(seg_all.league == league) & (seg_all.season == 2019)]
    match_dates = block.groupby("match_id").date.first().str[:10].sort_values()
    restart_ids = match_dates[match_dates >= cutoff].index
    pre_ids_sorted = match_dates[match_dates < cutoff].index  # already sorted ascending
    n = len(restart_ids)
    pre_ids = pre_ids_sorted[-n:]  # the final n pre-lockdown matches

    seg_pre = block[block.match_id.isin(pre_ids)]
    seg_post = block[block.match_id.isin(restart_ids)]
    log.info("%s: %d pre-lockdown matches vs %d restart matches", league, len(pre_ids), n)

    dm = build_net_design(pd.concat([seg_pre, seg_post], ignore_index=True))
    lam, _, _, _, _ = select_lambda(dm.X, dm.y, dm.w, dm.match_id, dm.penalized)

    m_pre = identifiability_metrics(seg_pre, lam=lam)
    m_post = identifiability_metrics(seg_post, lam=lam)
    row_pre, row_post = metrics_row(m_pre), metrics_row(m_post)

    diff = independent_bootstrap_diff(seg_post, seg_pre, lam, n_draws=N_DRAWS)  # post - pre
    for metric in METRICS:
        rows.append(dict(
            league=league, n_matches=n, metric=metric, lam=lam,
            pre_lockdown=row_pre[metric], restart=row_post[metric],
            diff_median=diff[metric].median(),
            diff_p05=diff[metric].quantile(0.05), diff_p95=diff[metric].quantile(0.95),
        ))
    log.info("%s: done", league)

result = pd.DataFrame(rows)
result.to_parquet(PROCESSED / "table3_restart.parquet", index=False)
log.info("done: %d rows", len(result))
