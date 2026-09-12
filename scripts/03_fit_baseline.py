"""Fit Model 1 (net, xG-differential RAPM) on one league-season as the phase-2
verification check (PLAN.md section 4.3, 4.5). Usage: python scripts/03_fit_baseline.py [LEAGUE SEASON]
"""

import sys

import numpy as np
import pandas as pd

from rapm import PROCESSED
from rapm.design import build_net_design
from rapm.ridge import fit_dense, select_lambda

league, season = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) == 3 else ("EPL", 2022)

seg = pd.read_parquet(PROCESSED / "segments.parquet")
seg = seg[(seg.league == league) & (seg.season == season)]
print(f"{league} {season}: {len(seg)} segments, {seg.match_id.nunique()} matches")

dm = build_net_design(seg)
print(f"design matrix: {dm.X.shape[0]} rows x {dm.X.shape[1]} columns "
      f"({dm.penalized.sum()} penalised, {(~dm.penalized).sum()} controls)")

lam_min, lam_1se, grid, mse, se = select_lambda(dm.X, dm.y, dm.w, dm.match_id, dm.penalized)
print(f"lambda_min={lam_min:.2f}  lambda_1se={lam_1se:.2f}  "
      f"(grid {grid[0]:.2f}-{grid[-1]:.2f}, CV MSE at min={mse.min():.4f})")

fit = fit_dense(dm.X, dm.y, dm.w, dm.penalized, lam=lam_min)
resid = dm.y - dm.X @ fit.b
ss_res = np.sum(dm.w * resid**2)
baseline_resid = dm.y - np.average(dm.y, weights=dm.w)
ss_tot = np.sum(dm.w * baseline_resid**2)
print(f"sigma2={fit.sigma2:.4f}  tau2={fit.tau2:.4f}  weighted R2 vs mean-only={1 - ss_res/ss_tot:.4f}")

player_idx = np.array([i for i, c in enumerate(dm.columns) if c.startswith("player_")])
player_cols = np.array([dm.columns[i] for i in player_idx])
b_players = fit.b[player_idx]
order = np.argsort(-b_players)
print("\ntop 10 players (xG diff per 90, ridge coefficient):")
for i in order[:10]:
    print(f"  {player_cols[i]:>14s}  {b_players[i]:+.3f}")
print("bottom 10:")
for i in order[-10:]:
    print(f"  {player_cols[i]:>14s}  {b_players[i]:+.3f}")

M = fit.M
cond = np.linalg.cond(M[np.ix_(dm.penalized, dm.penalized)])
print(f"\ncondition number of the penalised (player+team) block of M: {cond:.3e}")
