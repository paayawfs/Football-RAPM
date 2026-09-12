"""Counterfactual substitution truncation (PLAN.md section 5.4), the primary
mechanism test for H1: rebuild a five-sub match's roster as if only `keep`
substitutions per team had been allowed, and compare identifiability metrics on
the identical matches with only the design differing.

Chains are per-slot chronologically ordered (starter -> sub1 -> sub2 -> ...), and
because the drop rule always operates on a team's substitutions ranked by real
entry time, the dropped set is always a clean prefix or suffix of every single
slot's own chain -- never a gap in the middle of one slot while another slot is
untouched. That is what makes "walk back to the nearest surviving ancestor" well
defined below.
"""

import pandas as pd

from rapm.segments import player_intervals


def _next_in_chain(iv, rid):
    """The row (if any) whose roster_out points to rid -- who replaced rid."""
    nxt = iv[iv.roster_out == rid]
    return nxt.index[0] if len(nxt) else None


def truncate_rosters(roster, keep=3, variant="first"):
    """One match's roster (both teams) -> a new roster with each team's
    substitution count capped at `keep`.

    variant="first": keep the first `keep` substitutions chronologically; later
        ones never happen (entrant removed, whoever they'd have replaced instead
        plays to 90).
    variant="last": keep the last `keep`; earlier ones never happen (entrant
        removed, the slot's original occupant plays on until the first surviving
        substitution in that slot arrives, instead of being subbed early).

    A red-carded player is never extended past their real dismissal minute --
    if truncation would otherwise ask a sent-off player to keep playing (an
    already-rare case: a substitute who was later shown red *and* whose own
    subsequent substitution falls beyond the truncation cutoff), their original
    time is kept and the truncated minutes are simply lost rather than assigning
    an impossible lineup.
    """
    iv = player_intervals(roster).set_index("id", drop=False)
    out = roster.set_index("id", drop=False).copy()

    for team in iv.h_a.unique():
        subs = iv[(iv.h_a == team) & (iv.roster_out != 0)].sort_values("entry")
        if len(subs) <= keep:
            continue
        drop = list(subs.index[keep:]) if variant == "first" else list(subs.index[:-keep])
        drop_set = set(drop)
        out = out.drop(index=drop, errors="ignore")

        for rid in drop:
            anc = iv.loc[rid, "roster_out"]
            while anc in drop_set:
                anc = iv.loc[anc, "roster_out"]
            if iv.loc[anc, "red_card"] == 1:
                continue  # never extend a player past their real dismissal

            if variant == "first":
                new_time = 90 - iv.loc[anc, "entry"]
            else:
                successor = _next_in_chain(iv, rid)
                if successor is not None and successor in drop_set:
                    continue  # a later row in this same dropped run resolves it
                end = iv.loc[successor, "entry"] if successor is not None else 90
                new_time = end - iv.loc[anc, "entry"]
                if successor is not None:
                    # the surviving row right after a dropped prefix still points
                    # (via roster_out) at the now-removed row it originally replaced;
                    # redirect it to the ancestor or player_intervals' chain walk
                    # breaks on a missing id.
                    out.loc[successor, "roster_out"] = anc
            out.loc[anc, "time"] = new_time

    return out.reset_index(drop=True)


def truncate_league_season(rosters, keep=3, variant="first"):
    """Apply truncate_rosters() match-by-match across a league-season's rosters."""
    out = [truncate_rosters(roster, keep=keep, variant=variant)
           for _, roster in rosters.groupby("match_id")]
    return pd.concat(out, ignore_index=True) if out else rosters.iloc[0:0]
