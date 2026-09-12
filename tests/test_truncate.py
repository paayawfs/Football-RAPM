"""Counterfactual substitution truncation (PLAN.md section 5.4), checked against a
hand-built roster with an isolated single-sub slot, a double-sub chain, and a
red-carded substitute, verified via player_intervals on the truncated output
(not just the raw time/roster_out fields) so the chain actually resolves cleanly.
"""

import pandas as pd

from rapm.segments import player_intervals
from rapm.truncate import truncate_rosters


def _row(id_, position, time, roster_out=0, red_card=0, h_a="h"):
    return dict(id=id_, player_id=id_, player=f"p{id_}", position=position, time=time,
                roster_out=roster_out, red_card=red_card, h_a=h_a)


def _roster():
    rows = [
        # slot A: starter -> sub1 (kept-1st) -> sub2 (kept-4th, chained)
        _row(1, "DC", 40),                     # H1, starter
        _row(2, "Sub", 30, roster_out=1),       # enters 40, exits 70
        _row(3, "Sub", 20, roster_out=2),       # enters 70, exits 90 (event #4 by entry time)
        # slot B: starter -> sub (kept-2nd)
        _row(4, "MC", 50),                      # H2
        _row(5, "Sub", 40, roster_out=4),       # enters 50, exits 90
        # slot C: starter -> sub (kept-3rd)
        _row(6, "MC", 60),                      # H3
        _row(7, "Sub", 30, roster_out=6),       # enters 60, exits 90
        # slot D: starter -> sub (event #5, dropped under "first")
        _row(8, "FW", 80),                      # H4
        _row(9, "Sub", 10, roster_out=8),        # enters 80, exits 90
    ] + [_row(100 + i, "DC", 90, h_a="a") for i in range(11)]  # untouched away side
    return pd.DataFrame(rows)


def test_first_variant_drops_the_two_latest_events():
    out = truncate_rosters(_roster(), keep=3, variant="first")
    ids = set(out.id)
    assert 3 not in ids and 9 not in ids          # events #4 (id 3) and #5 (id 9) gone
    assert {1, 2, 4, 5, 6, 7, 8} <= ids            # everyone else survives

    iv = player_intervals(out)
    by_id = iv.set_index("id")
    assert by_id.loc[2, "exit"] == 90              # id2 now plays till 90 instead of 70
    assert by_id.loc[8, "exit"] == 90              # id8 (starter) now plays till 90 instead of 80
    # slots B and C are entirely untouched (their own subs, events #2 and #3, are kept):
    # the starter still exits when their real substitute enters, and that substitute
    # still plays out the rest of the match exactly as it actually happened.
    assert by_id.loc[4, "exit"] == 50 and by_id.loc[5, "entry"] == 50 and by_id.loc[5, "exit"] == 90
    assert by_id.loc[6, "exit"] == 60 and by_id.loc[7, "entry"] == 60 and by_id.loc[7, "exit"] == 90


def test_last_variant_drops_the_two_earliest_events():
    out = truncate_rosters(_roster(), keep=3, variant="last")
    ids = set(out.id)
    assert 2 not in ids and 5 not in ids          # events #1 (id2) and #2 (id5) gone
    assert {1, 3, 4, 6, 7, 8, 9} <= ids

    iv = player_intervals(out)
    by_id = iv.set_index("id")
    # id1 (starter, slot A) now plays until id3 (the first surviving sub in that
    # chain) arrives, instead of being subbed off at 40
    assert by_id.loc[1, "exit"] == 70
    assert by_id.loc[3, "entry"] == 70 and by_id.loc[3, "exit"] == 90
    # id4 (starter, slot B) has no surviving successor at all (id5 was its only
    # sub and it's dropped) -- plays the full match instead
    assert by_id.loc[4, "exit"] == 90
    # untouched slots (their own sub was kept, being event #3 or later)
    assert by_id.loc[6, "time"] == 60 and by_id.loc[7, "entry"] == 60


def test_never_extends_past_a_real_red_card():
    """id2 enters at 40 and is sent off at minute 65 (time=25, red_card=1) instead
    of playing until 70. That still leaves id3 entering at 65, after id7 (60) and
    before id9 (80), so the same events (#4 = id3, #5 = id9) are dropped as in the
    unmodified roster -- only id2's own fate changes. Under "first" truncation,
    dropping id3 would ordinarily extend its predecessor id2 to play till 90, but
    id2 was actually dismissed at 65 and cannot keep playing past that."""
    roster = _roster()
    roster.loc[roster.id == 2, ["time", "red_card"]] = [25, 1]

    out = truncate_rosters(roster, keep=3, variant="first")
    kept = out.set_index("id")
    assert 3 not in kept.index and 9 not in kept.index  # same two events dropped as before
    assert kept.loc[2, "time"] == 25          # id2 left unchanged, not extended to 90


if __name__ == "__main__":
    test_first_variant_drops_the_two_latest_events()
    test_last_variant_drops_the_two_earliest_events()
    test_never_extends_past_a_real_red_card()
    print("ok")
