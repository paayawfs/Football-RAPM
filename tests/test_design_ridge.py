"""Design matrix + ridge solver, checked against a synthetic panel with known truth
(PLAN.md section 4.3-4.5). Recovering the simulated player effects at a plausible
correlation is the same logic as the real simulation-recovery check in section 5.5,
just small enough to run as a unit test.
"""

import numpy as np
import pandas as pd

from rapm.design import build_net_design, build_od_design
from rapm.ridge import fit_dense, fit_sparse, select_lambda


def _synthetic_panel(rng, n=400, ghost=False):
    teams = ["T1", "T2", "T3", "T4"]
    players = {t: [f"{t}_p{i}" for i in range(14)] for t in teams}
    true_effect = {p: rng.normal(0, 0.5) for t in teams for p in players[t]}
    rows = []
    for i in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        hp = list(rng.choice(players[h], 11, replace=False))
        ap = list(rng.choice(players[a], 11, replace=False))
        dur, start = int(rng.integers(5, 20)), int(rng.integers(0, 70))
        true_y = sum(true_effect[p] for p in hp) - sum(true_effect[p] for p in ap)
        xg_diff = (true_y + rng.normal(0, 1.0)) * dur / 90
        date = "2020-08-01 12:00:00" if ghost else "2022-01-01 12:00:00"
        rows.append(dict(
            match_id=i // 3, league="TEST", season=2022, era="five", date=date,
            home_id=h, away_id=a, seg_idx=i % 3, start=start, end=start + dur, dur=dur,
            home_players=hp, away_players=ap, n_home=11, n_away=11,
            score_state=int(rng.integers(-2, 3)),
            xg_home=max(xg_diff, 0) + rng.random() * 0.1, xg_away=max(-xg_diff, 0) + rng.random() * 0.1,
            npxg_home=0.0, npxg_away=0.0, goals_home=0, goals_away=0,
        ))
    return pd.DataFrame(rows), true_effect


def test_net_design_shape():
    rng = np.random.default_rng(0)
    seg, _ = _synthetic_panel(rng)
    dm = build_net_design(seg)
    # 4 teams x 14 players + 4 team-season columns = 60 penalised; fixed control count
    assert dm.penalized.sum() == 60
    assert (~dm.penalized).sum() == 12  # 1 intercept + 4 score + 5 minute + man_adv + ghost
    assert dm.X.shape == (400, 72)


def test_dense_and_sparse_ridge_agree():
    rng = np.random.default_rng(1)
    seg, _ = _synthetic_panel(rng)
    dm = build_net_design(seg)
    fit_d = fit_dense(dm.X, dm.y, dm.w, dm.penalized, lam=10.0)
    fit_s = fit_sparse(dm.X, dm.y, dm.w, dm.penalized, lam=10.0)
    assert np.max(np.abs(fit_d.b - fit_s.b)) < 1e-6


def test_degenerate_control_column_does_not_crash():
    """A block with zero ghost-games segments makes that control column identically
    zero; fit_dense/fit_sparse must handle it (coefficient 0) rather than raise."""
    rng = np.random.default_rng(2)
    seg, _ = _synthetic_panel(rng, ghost=False)
    dm = build_net_design(seg)
    fit = fit_dense(dm.X, dm.y, dm.w, dm.penalized, lam=10.0)
    assert fit.b[dm.columns.index("ghost_games")] == 0.0
    assert np.all(np.isfinite(fit.b))


def test_simulation_recovery():
    """Recovers simulated player effects at a plausible correlation -- the same
    logic PLAN.md section 5.5 runs at full scale, small enough here to assert on."""
    rng = np.random.default_rng(3)
    seg, true_effect = _synthetic_panel(rng)
    dm = build_net_design(seg)
    lam_min, _, _, _, _ = select_lambda(dm.X, dm.y, dm.w, dm.match_id, dm.penalized)
    fit = fit_dense(dm.X, dm.y, dm.w, dm.penalized, lam=lam_min)
    player_cols = [c for c in dm.columns if c.startswith("player_")]
    est = dict(zip(player_cols, fit.b[:len(player_cols)]))
    truth = [true_effect[c[len("player_"):]] for c in player_cols]
    recovered = [est[c] for c in player_cols]
    assert np.corrcoef(truth, recovered)[0, 1] > 0.7


def test_od_design_matches_net_row_scale():
    rng = np.random.default_rng(4)
    seg, _ = _synthetic_panel(rng)
    dm = build_od_design(seg)
    assert dm.X.shape[0] == 2 * len(seg)
    fit_d = fit_dense(dm.X, dm.y, dm.w, dm.penalized, lam=10.0)
    fit_s = fit_sparse(dm.X, dm.y, dm.w, dm.penalized, lam=10.0)
    assert np.max(np.abs(fit_d.b - fit_s.b)) < 1e-6


if __name__ == "__main__":
    test_net_design_shape()
    test_dense_and_sparse_ridge_agree()
    test_degenerate_control_column_does_not_crash()
    test_simulation_recovery()
    test_od_design_matches_net_row_scale()
    print("ok")
