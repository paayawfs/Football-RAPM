"""PLAN.md section 5.4, counterfactual truncation: the primary mechanism test for
H1. For each sampled five-sub league-season, delete each team's 4th/5th
substitutions (the entrant never appears; whoever they'd have replaced plays to
90), rebuild segments on the identical matches, and compare identifiability
metrics against the actual design at a fixed, shared lambda -- a match-level
paired bootstrap (500 draws) gives the confidence interval on each metric's
difference.

This runs on one representative recent five-sub season per league, not all 28
five-sub league-seasons -- see PLAN.md for why, and for the full sweep as a
natural next step.
"""

import logging

import pandas as pd

from rapm import PROCESSED, RAW
from rapm.compare import METRICS, paired_bootstrap_diff
from rapm.design import build_net_design
from rapm.identify import identifiability_metrics
from rapm.ridge import select_lambda
from rapm.segments import QUARANTINE, build_from_frames
from rapm.truncate import truncate_league_season

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SAMPLE = [("EPL", 2024), ("La_liga", 2024), ("Bundesliga", 2024), ("Serie_A", 2024), ("Ligue_1", 2024)]
N_DRAWS_PRIMARY = 500
N_DRAWS_VARIANT = 100


def _load(league, season):
    base = RAW / "understat" / league / str(season)
    matches = pd.read_parquet(base / "matches.parquet")
    matches = matches[matches.is_result & ~matches.match_id.isin(QUARANTINE)]
    rosters = pd.read_parquet(base / "rosters.parquet")
    shots = pd.read_parquet(base / "shots.parquet")
    return matches, rosters, shots


rows = []
for league, season in SAMPLE:
    matches, rosters, shots = _load(league, season)
    seg_actual = build_from_frames(matches, rosters, shots)
    dm = build_net_design(seg_actual)
    lam, _, _, _, _ = select_lambda(dm.X, dm.y, dm.w, dm.match_id, dm.penalized)
    m_actual = identifiability_metrics(seg_actual, lam=lam)
    log.info("%s %s: actual cond_trim=%.1f eff_rank=%.1f id_share_med=%.4f (lam=%.1f)",
             league, season, m_actual.condition_number_trimmed, m_actual.effective_rank,
             m_actual.identification_share.median(), lam)

    for variant, n_draws in [("first", N_DRAWS_PRIMARY), ("last", N_DRAWS_VARIANT)]:
        trunc_rosters = truncate_league_season(rosters, keep=3, variant=variant)
        seg_trunc = build_from_frames(matches, trunc_rosters, shots)
        m_trunc = identifiability_metrics(seg_trunc, lam=lam)
        log.info("%s %s [%s]: truncated cond_trim=%.1f eff_rank=%.1f id_share_med=%.4f",
                 league, season, variant, m_trunc.condition_number_trimmed,
                 m_trunc.effective_rank, m_trunc.identification_share.median())

        diff = paired_bootstrap_diff(seg_actual, seg_trunc, lam, n_draws=n_draws)
        for metric in METRICS:
            rows.append(dict(
                league=league, season=season, variant=variant, metric=metric,
                actual=getattr(m_actual, metric), truncated=getattr(m_trunc, metric),
                diff_median=diff[metric].median(),
                diff_p05=diff[metric].quantile(0.05), diff_p95=diff[metric].quantile(0.95),
                n_draws=n_draws,
            ))
        log.info("%s %s [%s]: bootstrap done (%d draws)", league, season, variant, n_draws)

result = pd.DataFrame(rows)
result.to_parquet(PROCESSED / "table4_truncation.parquet", index=False)
log.info("done: %d rows", len(result))
