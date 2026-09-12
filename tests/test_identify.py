"""Section 5.2 identifiability metrics, checked on a small synthetic panel."""

import numpy as np
import pandas as pd

from rapm.identify import identifiability_metrics, player_minutes


def _panel(rng, n_matches=200):
    """Two teams, each an 11-player core that mostly plays plus a bench that fills
    in when a core player rests. Real rotation (nobody is *always* on) so player
    minutes vary continuously rather than splitting into an always/never edge case."""
    teams = ["T1", "T2"]
    core = {t: [f"{t}_c{i}" for i in range(11)] for t in teams}
    bench = {t: [f"{t}_b{i}" for i in range(20)] for t in teams}

    def lineup(team):
        n_rest = rng.integers(0, 3)
        rest = set(rng.choice(11, n_rest, replace=False)) if n_rest else set()
        subs_in = list(rng.choice(bench[team], n_rest, replace=False)) if n_rest else []
        return [p for k, p in enumerate(core[team]) if k not in rest] + subs_in

    rows = []
    for i in range(n_matches):
        h, a = teams
        hp, ap = lineup(h), lineup(a)
        rows.append(dict(
            match_id=i, league="TEST", season=2022, era="five", date="2022-01-01 12:00:00",
            home_id=h, away_id=a, seg_idx=0, start=0, end=90, dur=90,
            home_players=hp, away_players=ap, n_home=11, n_away=11, score_state=0,
            xg_home=float(rng.random()), xg_away=float(rng.random()), npxg_home=0.0, npxg_away=0.0,
            goals_home=0, goals_away=0,
        ))
    return pd.DataFrame(rows)


def test_player_minutes_selects_full_time_players():
    rng = np.random.default_rng(0)
    seg = _panel(rng, n_matches=50)
    minutes = player_minutes(seg)
    # a core player rests sometimes but still plays the large majority of matches
    assert minutes["T1_c0"] > 40 * 90
    # a bench player, spread thinly across a 20-player bench, plays only a handful
    assert minutes["T1_b0"] < 10 * 90


def test_identifiability_metrics_shapes_and_ranges():
    rng = np.random.default_rng(1)
    seg = _panel(rng)
    min_minutes = 450
    expected_n = int((player_minutes(seg) >= min_minutes).sum())

    m = identifiability_metrics(seg, min_minutes=min_minutes, lam=1000.0)

    assert m.n_players == expected_n
    assert m.condition_number >= m.condition_number_trimmed >= 1.0
    assert 0 < m.effective_rank <= m.n_players
    assert 0 <= m.variance_share_bottom_10pct <= 1
    assert 0 <= m.edf <= m.n_players
    # identification share is a diagonal of (M+lamI)^-1 M for a psd M: bounded in [0, 1]
    assert m.identification_share.between(-1e-8, 1 + 1e-8).all()
    assert len(m.contrast_precision) > 0
    assert (m.contrast_precision.shared_minutes > 0).all()
    assert (m.contrast_precision.contrast_variance >= -1e-8).all()  # a variance, never negative


def test_never_rested_player_has_near_zero_identification():
    """A player ever-present for their team is, after partialling out that team's own
    fixed effect, indistinguishable from the team itself -- the theoretical floor this
    whole design is meant to expose, not a bug in the projection."""
    rng = np.random.default_rng(2)
    n = 100
    rows = []
    for i in range(n):
        rows.append(dict(
            match_id=i, league="TEST", season=2022, era="five", date="2022-01-01 12:00:00",
            home_id="T1", away_id="T2", seg_idx=0, start=0, end=90, dur=90,
            home_players=[f"T1_p{k}" for k in range(11)],  # identical lineup every match
            away_players=[f"T2_p{k}" for k in range(9)] + list(rng.choice([f"T2_b{k}" for k in range(5)], 2, replace=False)),
            n_home=11, n_away=11, score_state=0,
            xg_home=float(rng.random()), xg_away=float(rng.random()), npxg_home=0.0, npxg_away=0.0,
            goals_home=0, goals_away=0,
        ))
    seg = pd.DataFrame(rows)
    m = identifiability_metrics(seg, min_minutes=450, lam=1000.0)
    assert m.identification_share["T1_p0"] < 1e-6


if __name__ == "__main__":
    test_player_minutes_selects_full_time_players()
    test_identifiability_metrics_shapes_and_ranges()
    test_never_rested_player_has_near_zero_identification()
    print("ok")
