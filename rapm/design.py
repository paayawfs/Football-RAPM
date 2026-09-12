"""Build the RAPM design matrix from lineup segments (PLAN.md sections 4.3-4.4).

Player and team-season columns are penalised (ridge-shrunk); everything else --
league-season intercept, score-state, minute-bucket, man-advantage, ghost-games --
is an unpenalised control. Output is always a sparse matrix: a single league-season
block is small enough for this to be free, and the pooled multi-season panel (tens
of thousands of player columns) needs it to be usable at all, so one code path
covers both instead of a dense path that only works at small scale.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

# PLAN.md's own stated default: a single global window, not refined per league
# (no attendance data on hand to do that refinement).
GHOST_START, GHOST_END = "2020-05-16", "2021-05-23"
MINUTE_EDGES = [0, 15, 30, 45, 60, 75, 90]  # 6 buckets; bucket 0 ([0,15)) is the reference


@dataclass
class DesignMatrix:
    X: sp.csr_matrix
    y: np.ndarray         # xG-differential outcome
    y_goals: np.ndarray   # goal-differential outcome (robustness variant)
    w: np.ndarray         # segment duration weights
    match_id: np.ndarray  # for GroupKFold
    columns: list          # column name per column of X, same order
    penalized: np.ndarray  # bool mask, True for player/team-season columns


def _minute_bucket_dummies(start):
    bucket = np.clip(np.searchsorted(MINUTE_EDGES, start, side="right") - 1, 0, 5)
    return [(f"minute_{MINUTE_EDGES[b]}_{MINUTE_EDGES[b + 1]}", (bucket == b).astype(float))
            for b in range(1, 6)]  # bucket 0 is the reference, no column for it


def _score_bucket_dummies(score_state):
    s = np.asarray(score_state)
    return [
        ("score_le_-2", (s <= -2).astype(float)),
        ("score_-1", (s == -1).astype(float)),
        ("score_+1", (s == 1).astype(float)),
        ("score_ge_+2", (s >= 2).astype(float)),
    ]


def _controls(league, season, score_state, start, date, n_att, n_def, home_attacking=None):
    """Unpenalised control columns shared by both models. `home_attacking` is None for
    the net model (already home-minus-away) and a bool array for the O/D model."""
    league_season = pd.Series(league).astype(str) + "_" + pd.Series(season).astype(str)
    cols = [(f"intercept_{key}", (league_season.values == key).astype(float))
            for key in sorted(league_season.unique())]
    cols += _score_bucket_dummies(score_state)
    cols += _minute_bucket_dummies(start)
    cols.append(("man_advantage", (np.asarray(n_att) - np.asarray(n_def)).astype(float)))
    date = np.asarray(date)
    cols.append(("ghost_games", ((date >= GHOST_START) & (date <= GHOST_END)).astype(float)))
    if home_attacking is not None:
        cols.append(("home_attacking", np.asarray(home_attacking).astype(float)))
    X = sp.csr_matrix(np.column_stack([c[1] for c in cols]))
    names = [c[0] for c in cols]
    return X, names, np.zeros(len(names), dtype=bool)  # controls are never penalised


def _stack(blocks):
    """blocks: list of (X, names, penalized_bool_array) with the same row count."""
    X = sp.hstack([b[0] for b in blocks], format="csr")
    names = [n for b in blocks for n in b[1]]
    penalized = np.concatenate([b[2] for b in blocks])
    return X, names, penalized


def _team_key(team_id, league, season):
    return pd.Series(team_id).astype(str) + "_" + pd.Series(league).astype(str) + "_" + pd.Series(season).astype(str)


def _onesided_player_block(player_lists, n_rows, prefix, sign=1.0):
    players = sorted({p for lst in player_lists for p in lst})
    idx = {p: j for j, p in enumerate(players)}
    rows, cols = [], []
    for i, lst in enumerate(player_lists):
        for p in lst:
            rows.append(i)
            cols.append(idx[p])
    X = sp.csr_matrix((np.full(len(rows), sign), (rows, cols)), shape=(n_rows, len(players)))
    return X, [f"{prefix}_{p}" for p in players]


def _onesided_team_block(team_id, league, season, prefix, n_rows, sign=1.0):
    key = _team_key(team_id, league, season)
    teams = sorted(key.unique())
    idx = {t: j for j, t in enumerate(teams)}
    X = sp.csr_matrix((np.full(n_rows, sign), (np.arange(n_rows), key.map(idx).values)),
                       shape=(n_rows, len(teams)))
    return X, [f"{prefix}_{t}" for t in teams]


def build_net_design(seg):
    """PLAN.md section 4.3: one row per segment, net (home-minus-away) outcome."""
    seg = seg.reset_index(drop=True)
    n = len(seg)

    home_p, home_names = _onesided_player_block(seg.home_players.tolist(), n, "player", sign=1.0)
    away_p, away_names = _onesided_player_block(seg.away_players.tolist(), n, "player", sign=-1.0)
    # players appearing on both sides across the block need one shared column each,
    # not two -- merge by name via a second hstack+groupby-sum pass.
    player_X, player_names = _merge_signed_duplicates(home_p, home_names, away_p, away_names)

    home_t, home_t_names = _onesided_team_block(seg.home_id, seg.league, seg.season, "team", n, sign=1.0)
    away_t, away_t_names = _onesided_team_block(seg.away_id, seg.league, seg.season, "team", n, sign=-1.0)
    team_X, team_names = _merge_signed_duplicates(home_t, home_t_names, away_t, away_t_names)

    ctrl_X, ctrl_names, ctrl_pen = _controls(
        seg.league, seg.season, seg.score_state.values, seg.start.values, seg.date.str[:10].values,
        seg.n_home.values, seg.n_away.values,
    )

    X, columns, penalized = _stack([
        (player_X, player_names, np.ones(player_X.shape[1], dtype=bool)),
        (team_X, team_names, np.ones(team_X.shape[1], dtype=bool)),
        (ctrl_X, ctrl_names, ctrl_pen),
    ])

    y = 90 * (seg.xg_home - seg.xg_away) / seg.dur
    y_goals = 90 * (seg.goals_home - seg.goals_away) / seg.dur
    return DesignMatrix(X=X, y=y.values, y_goals=y_goals.values, w=seg.dur.values.astype(float),
                         match_id=seg.match_id.values, columns=columns, penalized=penalized)


def _merge_signed_duplicates(X1, names1, X2, names2):
    """A player/team on both sides of a block (e.g. home in one match, away in
    another) gets two separate one-hot columns above; collapse same-named columns
    into one signed column by summing them."""
    X = sp.hstack([X1, X2], format="csr")
    names = names1 + names2
    unique = sorted(set(names))
    idx = {name: j for j, name in enumerate(unique)}
    col_map = sp.csr_matrix(
        (np.ones(len(names)), ([idx[n] for n in names], range(len(names)))),
        shape=(len(unique), len(names)),
    )
    return (col_map @ X.T).T.tocsr(), unique


def build_od_design(seg):
    """PLAN.md section 4.4: two rows per segment (one per attacking side)."""
    seg = seg.reset_index(drop=True)
    n = len(seg)
    n2 = 2 * n

    att_players, def_players = [], []
    for row in seg.itertuples():
        att_players += [row.home_players, row.away_players]
        def_players += [row.away_players, row.home_players]
    o_X, o_names = _onesided_player_block(att_players, n2, "O")
    d_X, d_names = _onesided_player_block(def_players, n2, "D")

    att_team_id = np.empty(n2, dtype=object)
    def_team_id = np.empty(n2, dtype=object)
    att_team_id[0::2], att_team_id[1::2] = seg.home_id.values, seg.away_id.values
    def_team_id[0::2], def_team_id[1::2] = seg.away_id.values, seg.home_id.values
    league2 = np.repeat(seg.league.values, 2)
    season2 = np.repeat(seg.season.values, 2)
    o_team_X, o_team_names = _onesided_team_block(att_team_id, league2, season2, "OT", n2)
    d_team_X, d_team_names = _onesided_team_block(def_team_id, league2, season2, "DT", n2)

    n_att = np.empty(n2); n_def = np.empty(n2)
    n_att[0::2], n_att[1::2] = seg.n_home.values, seg.n_away.values
    n_def[0::2], n_def[1::2] = seg.n_away.values, seg.n_home.values
    home_attacking = np.tile([True, False], n)
    score2 = np.where(home_attacking, np.repeat(seg.score_state.values, 2),
                       -np.repeat(seg.score_state.values, 2))
    start2 = np.repeat(seg.start.values, 2)
    date2 = np.repeat(seg.date.str[:10].values, 2)

    ctrl_X, ctrl_names, ctrl_pen = _controls(
        league2, season2, score2, start2, date2, n_att, n_def, home_attacking=home_attacking,
    )

    X, columns, penalized = _stack([
        (o_X, o_names, np.ones(o_X.shape[1], dtype=bool)),
        (d_X, d_names, np.ones(d_X.shape[1], dtype=bool)),
        (o_team_X, o_team_names, np.ones(o_team_X.shape[1], dtype=bool)),
        (d_team_X, d_team_names, np.ones(d_team_X.shape[1], dtype=bool)),
        (ctrl_X, ctrl_names, ctrl_pen),
    ])

    dur2 = np.repeat(seg.dur.values, 2).astype(float)
    xg_for = np.empty(n2); xg_for[0::2], xg_for[1::2] = seg.xg_home.values, seg.xg_away.values
    goals_for = np.empty(n2); goals_for[0::2], goals_for[1::2] = seg.goals_home.values, seg.goals_away.values
    y = 90 * xg_for / dur2
    y_goals = 90 * goals_for / dur2
    match_id2 = np.repeat(seg.match_id.values, 2)
    return DesignMatrix(X=X, y=y, y_goals=y_goals, w=dur2, match_id=match_id2,
                         columns=columns, penalized=penalized)
