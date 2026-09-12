"""Shared match-level bootstrap machinery for sections 5.3 and 5.4: comparing
identifiability metrics between two designs (pre/post era, actual/truncated,
pre-lockdown/restart) with a match-level block bootstrap rather than treating
segments as independent observations.

Every comparison here holds `lam` fixed across both arms and across every bootstrap
draw -- picked once from a real fit, never re-selected by CV inside a draw. Two
reasons: re-tuning lambda inside every one of a few hundred bootstrap iterations is
neither standard practice nor affordable at this scale, and PLAN.md 5.4 is explicit
that a truncated design's own outcome is meaningless (the player set differs), so
nothing here may depend on a truncated-design CV fit in the first place.
"""

import numpy as np
import pandas as pd

from rapm.identify import IdentifiabilityMetrics, identifiability_metrics

METRICS = ["condition_number_trimmed", "effective_rank", "variance_share_bottom_10pct",
           "edf", "id_share_median", "id_share_gt_half"]


def metrics_row(m: IdentifiabilityMetrics):
    return dict(
        condition_number_trimmed=m.condition_number_trimmed, effective_rank=m.effective_rank,
        variance_share_bottom_10pct=m.variance_share_bottom_10pct, edf=m.edf,
        id_share_median=m.identification_share.median(),
        id_share_gt_half=(m.identification_share > 0.5).mean(),
    )


def _draw_counts(matches, rng):
    """How many times each match id is drawn in one bootstrap resample."""
    draw = rng.choice(matches, size=len(matches), replace=True)
    return pd.Series(draw).value_counts()


def _apply_resample(seg, counts):
    """Repeat each match's full block of segment rows as many times as it was drawn."""
    by_match = {mid: grp for mid, grp in seg.groupby("match_id")}
    return pd.concat([by_match[mid] for mid, c in counts.items() for _ in range(c)], ignore_index=True)


def _resample_matches(seg, rng):
    """Match-level block bootstrap: resample match ids with replacement, repeating
    each drawn match's full block of segment rows as many times as it was drawn."""
    return _apply_resample(seg, _draw_counts(seg.match_id.unique(), rng))


def bootstrap_metrics(seg, lam, n_draws=100, min_minutes=450, seed=0):
    """n_draws refits of identifiability_metrics on match-level bootstrap resamples
    of seg, at a fixed lam. Returns a DataFrame, one row per draw."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_draws):
        seg_boot = _resample_matches(seg, rng)
        m = identifiability_metrics(seg_boot, min_minutes=min_minutes, lam=lam)
        rows.append(metrics_row(m))
    return pd.DataFrame(rows)


def summarize(boot_df):
    """Median and 5th/95th percentile for every metric column."""
    return pd.DataFrame({
        "median": boot_df.median(), "p05": boot_df.quantile(0.05), "p95": boot_df.quantile(0.95),
    })


def independent_bootstrap_diff(seg_a, seg_b, lam, n_draws=100, min_minutes=450, seed=0):
    """For two comparisons drawn from *different* match sets (e.g. the restart
    experiment's pre-lockdown vs. restart matches, or two league-seasons): bootstrap
    each side independently, then difference the two draw sequences index-wise. Valid
    because the two sequences are independent by construction (separate RNG streams
    below); do not use this when the two sides share the same matches -- see
    paired_bootstrap_diff for that case, which removes shared sampling noise instead
    of adding two independent noise sources together."""
    boot_a = bootstrap_metrics(seg_a, lam, n_draws=n_draws, min_minutes=min_minutes, seed=seed)
    boot_b = bootstrap_metrics(seg_b, lam, n_draws=n_draws, min_minutes=min_minutes, seed=seed + 1)
    return boot_a - boot_b


def paired_bootstrap_diff(seg_a, seg_b, lam, n_draws=100, min_minutes=450, seed=0):
    """The bootstrap that matters when both designs share the *same* matches (actual
    vs. truncated): each draw resamples match ids once and refits both designs on
    that same draw, so the reported interval is on the paired difference, not two
    independently noisy marginals subtracted after the fact."""
    rng = np.random.default_rng(seed)
    matches = seg_a.match_id.unique()
    rows = []
    for _ in range(n_draws):
        counts = _draw_counts(matches, rng)
        m_a = identifiability_metrics(_apply_resample(seg_a, counts), min_minutes=min_minutes, lam=lam)
        m_b = identifiability_metrics(_apply_resample(seg_b, counts), min_minutes=min_minutes, lam=lam)
        row_a, row_b = metrics_row(m_a), metrics_row(m_b)
        rows.append({k: row_a[k] - row_b[k] for k in METRICS})
    return pd.DataFrame(rows)
