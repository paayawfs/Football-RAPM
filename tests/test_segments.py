"""One hand-built match exercising every convention in PLAN.md section 4.2:
substitution, a substitute-replaces-substitute chain, an unused substitute, a red
card, a goal, an own goal (side-flip and xG exclusion), a penalty (in xG, out of
non-penalty xG), score-state timing (m+1 rule), boundary inclusivity at a segment
edge, and a stoppage-time shot landing in the final segment.

Layout (home H1..H14, away A1..A11):
  H1,H2,H4-H10 on the whole match. H3 on the whole match, scores an own goal at 50.
  H11 starts, subbed off at 60 -> H12, subbed off at 75 -> H13 (chain). H14 unused.
  A1-A11 start; A5 red-carded at 30, never replaced.
  Goals: away goal at minute 20 (open play), home own goal at minute 50 (benefits
  away), away penalty at minute 80. A stoppage-time shot at minute 91.
Expected breakpoints: {0,21,30,51,60,75,81,90} -> 7 segments.
"""

import pandas as pd

from rapm.segments import build_segments, era


def _row(id_, player, position, time, h_a, roster_out=0, red_card=0):
    return dict(id=id_, player_id=id_, player=player, position=position, time=time,
                h_a=h_a, roster_out=roster_out, red_card=red_card)


def _match():
    m = pd.DataFrame([dict(match_id=1, league="EPL", season=2022,
                            datetime="2022-10-01 15:00:00")])
    return next(m.itertuples())


def _roster():
    rows = [
        _row(1, "H1", "DC", 90, "h"), _row(2, "H2", "DC", 90, "h"),
        _row(3, "H3", "DC", 90, "h"),  # own-goal scorer, plays on
        _row(4, "H4", "MC", 90, "h"), _row(5, "H5", "MC", 90, "h"),
        _row(6, "H6", "MC", 90, "h"), _row(7, "H7", "MC", 90, "h"),
        _row(8, "H8", "FW", 90, "h"), _row(9, "H9", "FW", 90, "h"),
        _row(10, "H10", "FW", 90, "h"),
        _row(11, "H11", "FW", 60, "h"),                 # subbed off at 60
        _row(12, "H12", "Sub", 15, "h", roster_out=11),  # on 60-75, subbed off
        _row(13, "H13", "Sub", 15, "h", roster_out=12),  # on 75-90
        _row(14, "H14", "Sub", 0, "h", roster_out=0),    # unused, must be dropped
        _row(21, "A1", "DC", 90, "a"), _row(22, "A2", "DC", 90, "a"),
        _row(23, "A3", "MC", 90, "a"), _row(24, "A4", "MC", 90, "a"),
        _row(25, "A5", "MC", 30, "a", red_card=1),  # sent off at 30, never replaced
        _row(26, "A6", "FW", 90, "a"), _row(27, "A7", "FW", 90, "a"),
        _row(28, "A8", "DC", 90, "a"), _row(29, "A9", "DC", 90, "a"),
        _row(30, "A10", "MC", 90, "a"), _row(31, "A11", "FW", 90, "a"),
    ]
    return pd.DataFrame(rows)


def _shots():
    rows = [
        dict(minute=20, result="Goal", h_a="a", xG=0.15, situation="OpenPlay"),
        dict(minute=50, result="OwnGoal", h_a="h", xG=0.0, situation="OpenPlay"),
        dict(minute=80, result="Goal", h_a="a", xG=0.79, situation="Penalty"),
        dict(minute=91, result="SavedShot", h_a="h", xG=0.05, situation="OpenPlay"),
    ]
    return pd.DataFrame(rows)


def test_segments():
    seg = build_segments(_match(), _roster(), _shots())

    # convention 1: segment i spans [b_i, b_{i+1}); breakpoints from subs/reds/goals+1
    assert seg.start.tolist() == [0, 21, 30, 51, 60, 75, 81]
    assert seg.end.tolist() == [21, 30, 51, 60, 75, 81, 90]
    assert (seg.end - seg.start == seg.dur).all()

    # 11 players a side, every segment (one-for-one subs, a straight man-down red card)
    assert (seg.n_home == 11).all()
    assert (seg.n_away.iloc[:2] == 11).all()   # A5 still on through end of seg[1] (=30)
    assert (seg.n_away.iloc[2:] == 10).all()   # gone from seg[2] (start=30) onward

    # convention 4: on-pitch test is entry<=start and exit>=end (boundary inclusive) ->
    # H11 (exit=60) is on for the segment ending at 60, off for the one starting at 60
    assert 11 in seg.iloc[2].home_players   # seg [30,51): H11 still on
    assert 11 not in seg.iloc[4].home_players  # seg [60,75): H11 off, H12 on
    assert 12 in seg.iloc[4].home_players
    assert 12 not in seg.iloc[5].home_players  # seg [75,81): H12 off (chain), H13 on
    assert 13 in seg.iloc[5].home_players

    # unused sub H14 never appears
    assert all(14 not in ps for ps in seg.home_players)

    # convention 6: score_state = home - away among goals with m+1 <= segment start
    assert seg.score_state.tolist() == [0, -1, -1, -2, -2, -2, -3]

    # convention 7: own goal counts for the benefiting side (away) in goals_*, but its
    # (zero) xG is excluded from the xG outcome regardless
    assert seg.iloc[2].goals_away == 1 and seg.iloc[2].goals_home == 0  # own goal's segment
    assert seg.goals_away.sum() == 3 and seg.goals_home.sum() == 0  # goal + own goal + penalty
    assert seg.iloc[2].xg_home == 0.0  # the own goal's own (zero) xG isn't added to xg_home

    # convention 8: penalty counts in xg but not npxg
    pen_seg = seg.iloc[5]
    assert pen_seg.xg_away == 0.79
    assert pen_seg.npxg_away == 0.0

    # convention 3: stoppage-time shot (minute 91) lands in the final segment
    last = seg.iloc[-1]
    assert last.start == 81 and last.end == 90
    assert last.xg_home == 0.05


def test_era():
    assert era("EPL", 2018, "2018-10-01") == "three"
    assert era("EPL", 2022, "2022-10-01") == "five"
    assert era("EPL", 2019, "2019-12-01") == "three"   # pre-restart
    assert era("EPL", 2019, "2020-06-20") == "five"    # post-restart
    assert era("Ligue_1", 2019, "2020-06-20") == "three"  # season abandoned, no restart
    assert era("EPL", 2020, "2020-10-01") == "three"   # PL reverted
    assert era("La_liga", 2020, "2020-10-01") == "five"  # everyone else stayed at five
    assert era("EPL", 2021, "2021-10-01") == "three"
    assert era("La_liga", 2021, "2021-10-01") == "five"


if __name__ == "__main__":
    test_segments()
    test_era()
    print("ok")
