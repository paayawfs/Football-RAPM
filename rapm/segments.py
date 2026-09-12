"""Build lineup segments from Understat rosters and shots (PLAN.md section 4).

A segment is a stretch of a match where the set of players on the pitch for both
teams is constant. Breaks occur at substitutions, red cards, and goals (PLAN.md
section 4.2). Conventions below are numbered to match the plan and asserted by
tests/test_segments.py.
"""

import numpy as np
import pandas as pd

QUARANTINE = {
    # roster malformed (both sides []), no shot data at all
    27930, 4238,
    # score does not reconcile from shots; shot data has a genuine gap or duplicate
    5274, 5615, 29482, 5999, 5959, 5894,
}

# Five-substitution adoption per league-season (Understat "season" = start year).
# 2019 straddles the mid-season restart; every other season is a flat rule.
RESTART_DATE = {
    "EPL": "2020-06-17", "La_liga": "2020-06-11",
    "Serie_A": "2020-06-20", "Bundesliga": "2020-05-16",
}  # Ligue_1 2019 was abandoned before any restart: stays three-sub throughout.
FIVE_SUB_FROM_2020 = {"La_liga", "Serie_A", "Bundesliga", "Ligue_1"}  # EPL reverted to three


def era(league, season, date):
    """Return 'three' or 'five' subs, the rule in effect for one match."""
    if season <= 2018:
        return "three"
    if season >= 2022:
        return "five"
    if season == 2019:
        cutoff = RESTART_DATE.get(league)
        return "five" if cutoff and str(date) >= cutoff else "three"
    # season in (2020, 2021): Premier League reverted to three, everyone else stayed five
    return "five" if league in FIVE_SUB_FROM_2020 else "three"


def player_intervals(roster):
    """One match's roster rows -> the same rows with entry/exit minute columns.
    Starters enter at 0; a substitute enters when the player they replaced exits,
    resolved through chains of substitute-replaces-substitute. Unused substitutes
    (never entered) are dropped."""
    roster = roster[~((roster.position == "Sub") & (roster.roster_out == 0))].copy()
    by_id = roster.set_index("id")
    entry = {}

    def entry_of(rid):
        if rid not in entry:
            row = by_id.loc[rid]
            prev = row.roster_out
            entry[rid] = 0 if prev == 0 else entry_of(prev) + by_id.loc[prev, "time"]
        return entry[rid]

    for rid in by_id.index:
        entry_of(rid)
    roster = roster.copy()
    roster["entry"] = roster["id"].map(entry)
    roster["exit"] = roster["entry"] + roster["time"]
    return roster


def build_segments(match, roster, shots):
    """match: one row from matches.parquet. roster/shots: that match's rows.
    Returns the segment table for this match (PLAN.md section 4.2/4.3)."""
    r = player_intervals(roster)

    goal_shots = shots[shots.result.isin(["Goal", "OwnGoal"])].copy()
    # An own goal's h_a is the scorer's (conceding) side; it counts for the other side.
    goal_shots["scoring_side"] = np.where(
        goal_shots.result == "OwnGoal", goal_shots.h_a.map({"h": "a", "a": "h"}), goal_shots.h_a
    )

    breaks = {0, 90} | set(r.exit) | {m + 1 for m in goal_shots.minute}
    bounds = sorted(b for b in breaks if 0 <= b <= 90)

    rows = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        if end <= start:
            continue
        on = r[(r.entry <= start) & (r.exit >= end)]
        home_players = sorted(on[on.h_a == "h"].player_id.tolist())
        away_players = sorted(on[on.h_a == "a"].player_id.tolist())

        last = i == len(bounds) - 2
        in_window = (shots.minute >= start) & (shots.minute < end) if not last else shots.minute >= start
        seg_shots = shots[in_window & (shots.result != "OwnGoal")]  # own-goal xG excluded (always 0, but by convention)
        xg = seg_shots.groupby("h_a").xG.sum()
        npxg = seg_shots[seg_shots.situation != "Penalty"].groupby("h_a").xG.sum()

        seg_goals = goal_shots[(goal_shots.minute >= start) & (goal_shots.minute < end)] if not last \
            else goal_shots[goal_shots.minute >= start]
        goals = seg_goals.scoring_side.value_counts()
        score = goal_shots[goal_shots.minute + 1 <= start].scoring_side.value_counts()

        rows.append(dict(
            match_id=match.match_id, league=match.league, season=match.season,
            era=era(match.league, match.season, match.datetime[:10]), date=match.datetime,
            seg_idx=i, start=start, end=end, dur=end - start,
            home_players=home_players, away_players=away_players,
            n_home=len(home_players), n_away=len(away_players),
            score_state=int(score.get("h", 0) - score.get("a", 0)),
            xg_home=float(xg.get("h", 0.0)), xg_away=float(xg.get("a", 0.0)),
            npxg_home=float(npxg.get("h", 0.0)), npxg_away=float(npxg.get("a", 0.0)),
            goals_home=int(goals.get("h", 0)), goals_away=int(goals.get("a", 0)),
        ))
    return pd.DataFrame(rows)


def build_league_season(base):
    """base: a data/raw/understat/{league}/{season} directory. Returns its segment table."""
    matches = pd.read_parquet(base / "matches.parquet")
    matches = matches[matches.is_result & ~matches.match_id.isin(QUARANTINE)]
    rosters = pd.read_parquet(base / "rosters.parquet")
    shots = pd.read_parquet(base / "shots.parquet")
    out = []
    for match in matches.itertuples():
        r = rosters[rosters.match_id == match.match_id]
        s = shots[shots.match_id == match.match_id]
        if r.empty:
            continue
        out.append(build_segments(match, r, s))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
