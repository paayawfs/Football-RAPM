"""PLAN.md section 5.3: the event-study data underlying "Figure 3" -- each
league-season's identifiability metrics plotted against years since that league's
own adoption of five substitutions, with a match-level bootstrap interval per
point instead of an asymptotic-SE regression (five leagues, twelve seasons is too
thin for the latter). The Premier League's reversal (five only for the 2020
restart, back to three for 2020-21/2021-22, five again from 2022-23) is kept as
its own series rather than aligned into the other four leagues' one-way switch.

This is descriptive, not causal -- section 5.4's restart/truncation experiments
carry the causal weight; this only supplies the plotted comparison next to them.
"""

import logging

import pandas as pd

from rapm import PROCESSED
from rapm.compare import METRICS, bootstrap_metrics, metrics_row
from rapm.design import build_net_design
from rapm.identify import identifiability_metrics
from rapm.ridge import select_lambda

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

N_DRAWS = 50  # PLAN.md doesn't fix a count for 5.3 (unlike 5.4's explicit 500); this
              # keeps the full-panel run tractable alongside the 5.4 experiments.

seg_all = pd.read_parquet(PROCESSED / "segments.parquet")
t2 = pd.read_parquet(PROCESSED / "table2_identifiability.parquet")

adoption_season = t2[t2.era == "five"].groupby("league").season.min()

rows = []
for (league, season), seg in seg_all.groupby(["league", "season"]):
    dm = build_net_design(seg)
    lam, _, _, _, _ = select_lambda(dm.X, dm.y, dm.w, dm.match_id, dm.penalized)
    m = identifiability_metrics(seg, lam=lam)
    point = metrics_row(m)

    boot = bootstrap_metrics(seg, lam, n_draws=N_DRAWS)
    years_since = season - adoption_season.get(league, season)

    for metric in METRICS:
        rows.append(dict(
            league=league, season=season, years_since_adoption=int(years_since),
            is_premier_league=(league == "EPL"), metric=metric,
            point=point[metric], boot_median=boot[metric].median(),
            boot_p05=boot[metric].quantile(0.05), boot_p95=boot[metric].quantile(0.95),
        ))
    log.info("%s %s (years_since_adoption=%d) done", league, season, years_since)

result = pd.DataFrame(rows)
result.to_parquet(PROCESSED / "table_event_study.parquet", index=False)
log.info("done: %d rows", len(result))
