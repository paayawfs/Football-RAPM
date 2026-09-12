"""Weighted ridge with a partial penalty (PLAN.md section 4.5).

minimise sum_s w_s (y_s - x_s b)^2 + lambda * sum_{j penalised} b_j^2

`fit_dense` forms M = X^T W X explicitly and solves the normal equations -- fine for
a single league-season block (~500-600 columns), and M is needed anyway for the
identifiability metrics in phase 3. `fit_sparse` never forms M; it solves the
mathematically equivalent augmented least-squares problem with `lsqr`, which is
what the pooled multi-season panel (tens of thousands of columns) requires.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from sklearn.model_selection import GroupKFold


@dataclass
class RidgeFit:
    b: np.ndarray
    lam: float
    M: np.ndarray | None    # X^T W X; only set by fit_dense (needed for phase-3 metrics)
    sigma2: float           # weighted mean squared residual
    tau2: float             # sigma2 / lam -- prior variance under the ridge-as-MAP view


def _row_scale(X, w):
    sw = np.sqrt(w)
    if sp.issparse(X):
        return X.multiply(sw[:, None]).tocsr(), sw
    return X * sw[:, None], sw


def _drop_degenerate(Xw, penalized):
    """M+D can only be singular along a direction supported entirely on unpenalised
    coordinates: D adds lam to every penalised diagonal entry, which alone rules out
    a null vector with any penalised-coordinate support (M+D is PSD, so v'(M+D)v=0
    forces v'Dv=0, i.e. v is zero everywhere D is positive). So it is enough to check
    the unpenalised columns for exact linear dependence -- not just the all-zero case
    (a bucket a small block never visits) but real collinearity between two nonzero
    controls, e.g. ghost_games and the block's own single intercept becoming identical
    for a league-season that falls entirely inside the no-crowds window. Detected by
    a plain Gram-Schmidt pass, cheap since the unpenalised block is a handful of
    columns; a dependent column's effect can't be estimated from this block and is 0
    by convention, so it is dropped from the solve rather than left to fail it."""
    unpen_idx = np.flatnonzero(~penalized)
    unpen = Xw[:, unpen_idx]
    unpen = unpen.toarray() if sp.issparse(unpen) else unpen  # a handful of columns, cheap either way
    keep = np.ones(Xw.shape[1], dtype=bool)
    basis = np.zeros((Xw.shape[0], 0))
    for k, j in enumerate(unpen_idx):
        v = unpen[:, k]
        resid = v - basis @ (basis.T @ v) if basis.shape[1] else v
        norm = np.linalg.norm(resid)
        if norm <= 1e-8 * max(np.linalg.norm(v), 1.0):
            keep[j] = False
        else:
            basis = np.hstack([basis, (resid / norm)[:, None]])
    return keep if not keep.all() else None


def fit_dense(X, y, w, penalized, lam):
    Xw, sw = _row_scale(X, w)
    Xw_dense = Xw.toarray() if sp.issparse(Xw) else Xw
    keep = _drop_degenerate(Xw_dense, penalized)
    Xk = Xw_dense if keep is None else Xw_dense[:, keep]
    pen_k = penalized if keep is None else penalized[keep]
    z = sw * y
    M = Xk.T @ Xk
    rhs = Xk.T @ z
    D = np.diag(pen_k.astype(float)) * lam
    b_k = np.linalg.solve(M + D, rhs)
    b = b_k if keep is None else _scatter(b_k, keep, len(penalized))
    resid = y - (X @ b)
    sigma2 = float(np.sum(w * resid**2) / np.sum(w))
    return RidgeFit(b=b, lam=lam, M=M, sigma2=sigma2, tau2=sigma2 / lam if lam > 0 else np.inf)


def _scatter(values, keep, n):
    b = np.zeros(n)
    b[keep] = values
    return b


def fit_sparse(X, y, w, penalized, lam):
    """Augmented least squares: stack sqrt(lam)*I on the penalised columns as extra
    rows with a zero target. This is an exact solve of the ridge normal equations,
    not an approximation, and never materialises X^T X."""
    Xw, sw = _row_scale(X, w)
    keep = _drop_degenerate(Xw, penalized)
    Xk = Xw if keep is None else Xw[:, keep]
    pen_k = penalized if keep is None else penalized[keep]
    z = sw * y
    p = Xk.shape[1]
    pen_idx = np.flatnonzero(pen_k)
    aug_rows = sp.csr_matrix(
        (np.full(len(pen_idx), np.sqrt(lam)), (np.arange(len(pen_idx)), pen_idx)),
        shape=(len(pen_idx), p),
    )
    A = sp.vstack([Xk, aug_rows], format="csr")
    rhs = np.concatenate([z, np.zeros(len(pen_idx))])
    result = lsqr(A, rhs, atol=1e-10, btol=1e-10, iter_lim=5000)
    b_k = result[0]
    b = b_k if keep is None else _scatter(b_k, keep, len(penalized))
    resid = y - (X @ b)
    sigma2 = float(np.sum(w * resid**2) / np.sum(w))
    return RidgeFit(b=b, lam=lam, M=None, sigma2=sigma2, tau2=sigma2 / lam if lam > 0 else np.inf)


def select_lambda(X, y, w, match_id, penalized, fit_fn=fit_dense, grid=None, n_splits=5):
    """GroupKFold by match_id, weighted-MSE grid search. Returns (lambda_min,
    lambda_1se, grid, mean_mse, se_mse) so the curve can be plotted later
    (PLAN.md figure 2) without refitting."""
    if grid is None:
        grid = np.logspace(-1, 7, 25)
    gkf = GroupKFold(n_splits=n_splits)
    fold_mse = np.zeros((n_splits, len(grid)))
    for k, (train, test) in enumerate(gkf.split(X, y, groups=match_id)):
        for j, lam in enumerate(grid):
            fit = fit_fn(X[train], y[train], w[train], penalized, lam)
            pred = X[test] @ fit.b
            resid = y[test] - pred
            fold_mse[k, j] = np.sum(w[test] * resid**2) / np.sum(w[test])
    mean_mse = fold_mse.mean(axis=0)
    se_mse = fold_mse.std(axis=0, ddof=1) / np.sqrt(n_splits)
    i_min = int(np.argmin(mean_mse))
    threshold = mean_mse[i_min] + se_mse[i_min]
    # largest lambda (strongest shrinkage) whose mean CV MSE is still within 1 SE of the min
    candidates = np.flatnonzero(mean_mse <= threshold)
    i_1se = int(candidates[np.argmax(grid[candidates])])
    return grid[i_min], grid[i_1se], grid, mean_mse, se_mse
