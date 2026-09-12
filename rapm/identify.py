"""Section 5.2: design-matrix identifiability metrics for one league-season block.

Everything here operates on the player block after partialling out the unpenalised
controls *and* the team-season columns (PLAN.md section 5's preamble): this
restricts identification to genuine within-team lineup variation -- the thing five
substitutions is hypothesised to increase -- rather than cross-team schedule
variation, which the team-season fixed effects already absorb.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from rapm.design import build_net_design
from rapm.ridge import select_lambda


@dataclass
class IdentifiabilityMetrics:
    n_players: int                    # players with >= min_minutes in this block
    condition_number: float
    condition_number_trimmed: float   # e_1 / e_{0.9p}
    effective_rank: float
    variance_share_bottom_10pct: float
    lam: float                        # CV-selected lambda (from the full model)
    edf: float                        # effective degrees of freedom at lam
    identification_share: pd.Series   # per player, indexed by player_id
    contrast_precision: pd.DataFrame  # top-200 teammate pairs by shared minutes


def player_minutes(seg):
    """Total minutes per player_id in this block, home or away."""
    exploded = pd.concat([
        seg[["dur"]].assign(player=seg.home_players),
        seg[["dur"]].assign(player=seg.away_players),
    ]).explode("player")
    return exploded.groupby("player").dur.sum()


def _project_out(X_players, C, w):
    """Weighted projection of X_players off the control matrix C (PLAN.md section 5's
    X~ formula), via row-scaled least squares rather than an explicit (C'WC)^-1 --
    robust to a rank-deficient C (e.g. a team-season column with a genuinely all-zero
    unpenalised control alongside it) without needing special-case handling."""
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(C * sw[:, None], X_players * sw[:, None], rcond=None)
    return X_players - C @ beta


def _shared_minutes_pairs(seg, eligible, top_n=200):
    """Total shared on-pitch minutes for every teammate pair, restricted to
    eligible (a set of str(player_id)), across the whole block. Returns the top_n
    pairs by shared minutes, with ids as strings throughout -- player ids may be
    Understat ints in real data or arbitrary hashables in a synthetic test, and
    design.py's "player_<id>" column names are strings either way, so string is
    the one representation both sides can compare on without assuming a type."""
    shared = {}
    for lst, dur in zip(pd.concat([seg.home_players, seg.away_players]),
                         pd.concat([seg.dur, seg.dur])):
        present = [str(p) for p in lst if str(p) in eligible]
        for i, j in combinations(sorted(present), 2):
            shared[(i, j)] = shared.get((i, j), 0.0) + dur
    top = sorted(shared.items(), key=lambda kv: -kv[1])[:top_n]
    return pd.DataFrame([(i, j, m) for (i, j), m in top],
                         columns=["player_i", "player_j", "shared_minutes"])


def identifiability_metrics(seg, min_minutes=450, lam=None):
    dm = build_net_design(seg)
    minutes = player_minutes(seg)
    eligible = {str(p) for p in minutes[minutes >= min_minutes].index}

    player_col_idx = [i for i, c in enumerate(dm.columns) if c.startswith("player_")]
    player_ids = [dm.columns[i][len("player_"):] for i in player_col_idx]
    keep = [i for i, pid in zip(player_col_idx, player_ids) if pid in eligible]
    kept_ids = [dm.columns[i][len("player_"):] for i in keep]

    X_players = dm.X[:, keep].toarray()
    control_idx = [i for i, pen in enumerate(dm.penalized) if not pen]  # unpenalised controls
    team_idx = [i for i, c in enumerate(dm.columns) if c.startswith("team_")]
    C = dm.X[:, control_idx + team_idx].toarray()

    X_tilde = _project_out(X_players, C, dm.w)
    Xw = X_tilde * np.sqrt(dm.w)[:, None]
    M = (Xw.T @ Xw) / dm.w.sum()  # normalised per minute, for shape comparisons

    e = np.linalg.eigvalsh(M)[::-1]
    e = np.clip(e, 0, None)
    p = len(e)
    cond = e[0] / e[-1] if e[-1] > 0 else np.inf
    trim_idx = int(np.floor(0.9 * (p - 1)))
    cond_trim = e[0] / e[trim_idx] if e[trim_idx] > 0 else np.inf
    share = e / e.sum() if e.sum() > 0 else np.zeros_like(e)
    nz = share > 0
    eff_rank = float(np.exp(-np.sum(share[nz] * np.log(share[nz]))))
    n_bottom = max(1, int(np.floor(0.1 * p)))
    var_bottom = float(e[-n_bottom:].sum() / e.sum()) if e.sum() > 0 else np.nan

    if lam is None:
        lam, _, _, _, _ = select_lambda(dm.X, dm.y, dm.w, dm.match_id, dm.penalized)

    # EDF / identification share / contrast precision use the *raw* (unnormalised)
    # M, since they must be on the same scale as lam (which was chosen against the
    # raw model, not the per-minute-normalised one used for the shape metrics above).
    M_raw = Xw.T @ Xw
    e_raw = np.linalg.eigvalsh(M_raw)[::-1]
    edf = float(np.sum(np.clip(e_raw, 0, None) / (np.clip(e_raw, 0, None) + lam)))

    A = np.linalg.solve(M_raw + lam * np.eye(p), M_raw)   # (M+lam I)^-1 M
    id_share = pd.Series(np.diag(A), index=kept_ids)

    pairs = _shared_minutes_pairs(seg, set(kept_ids))
    # A = (M+lamI)^-1 M already; solving once more against (M+lamI) gives the full
    # sandwich (M+lamI)^-1 M (M+lamI)^-1 -- valid because M and (M+lamI)^-1 share
    # eigenvectors, so they commute.
    Ainv_M_Ainv = np.linalg.solve(M_raw + lam * np.eye(p), A)
    idx = {pid: k for k, pid in enumerate(kept_ids)}
    ses = []
    for i, j in zip(pairs.player_i, pairs.player_j):
        c = np.zeros(p); c[idx[i]] = 1.0; c[idx[j]] = -1.0
        ses.append(float(c @ Ainv_M_Ainv @ c))
    pairs["contrast_variance"] = ses  # multiply by sigma^2 and take sqrt outside, once sigma is known

    return IdentifiabilityMetrics(
        n_players=p, condition_number=float(cond), condition_number_trimmed=float(cond_trim),
        effective_rank=eff_rank, variance_share_bottom_10pct=var_bottom,
        lam=float(lam), edf=edf, identification_share=id_share, contrast_precision=pairs,
    )
