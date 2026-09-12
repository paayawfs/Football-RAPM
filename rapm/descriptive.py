"""Section 5.1: descriptive lineup-variation metrics for one league-season block.

Everything here is derived from segments.parquet plus that league-season's own
rosters.parquet (only needed for the two metrics -- substitution count and minute --
that depend on roster fields segments.parquet doesn't carry).
"""

import numpy as np
import pandas as pd

from rapm.segments import player_intervals


def _side_long(seg):
    """One row per (segment, side): match_id, team_id, players (tuple), dur, seg_idx."""
    home = pd.DataFrame({"match_id": seg.match_id.values, "team_id": seg.home_id.values,
                          "players": seg.home_players.map(tuple).values, "dur": seg.dur.values,
                          "seg_idx": seg.seg_idx.values})
    away = pd.DataFrame({"match_id": seg.match_id.values, "team_id": seg.away_id.values,
                          "players": seg.away_players.map(tuple).values, "dur": seg.dur.values,
                          "seg_idx": seg.seg_idx.values})
    return pd.concat([home, away], ignore_index=True)


def _sub_counts_and_minutes(rosters):
    """Substitutions used per team-match, and each substitute's entry minute, from
    raw roster rows (segments.parquet alone doesn't say *why* a lineup changed)."""
    subs_per_team_match = rosters[rosters.roster_out != 0].groupby(["match_id", "h_a"]).size()
    entry_minutes = []
    for mid, roster in rosters.groupby("match_id"):
        iv = player_intervals(roster)
        entry_minutes += iv[iv.roster_out != 0].entry.tolist()
    return subs_per_team_match, np.array(entry_minutes)


def descriptive_stats(seg, rosters):
    """seg: segments.parquet rows for one league-season (or any match subset).
    rosters: that same subset's rows from rosters.parquet."""
    long = _side_long(seg)
    n_teams = long.team_id.nunique()
    n_matches = seg.match_id.nunique()

    subs_pm, entry_minutes = _sub_counts_and_minutes(rosters[rosters.match_id.isin(seg.match_id)])
    subs_dist = subs_pm.value_counts(normalize=True).sort_index()

    seg_per_match = seg.groupby("match_id").size()
    dur_all = seg.dur.values

    starters = long[long.seg_idx == 0].set_index(["match_id", "team_id"]).players

    # distinct lineups and Herfindahl per team, then averaged to one season number
    per_team_distinct_onpitch, per_team_distinct_starting, per_team_hhi, per_team_corr = [], [], [], []
    for team_id, grp in long.groupby("team_id"):
        onpitch = grp.players.unique()
        per_team_distinct_onpitch.append(len(onpitch))
        start_sets = grp[grp.seg_idx == 0].players.unique()
        per_team_distinct_starting.append(len(start_sets))
        minutes_by_lineup = grp.groupby("players").dur.sum()
        share = minutes_by_lineup / minutes_by_lineup.sum()
        per_team_hhi.append(float((share**2).sum()))

        players_ever = sorted({p for lst in grp.players for p in lst})
        if len(players_ever) > 1:
            pres = pd.DataFrame(0, index=range(len(grp)), columns=players_ever)
            for i, lst in enumerate(grp.players):
                pres.loc[i, list(lst)] = 1
            corr = pres.corr().values
            np.fill_diagonal(corr, np.nan)
            if np.isfinite(corr).any():
                per_team_corr.append(np.nanmean(np.abs(corr)))

    # substitute-minutes share: a player's whole-match minutes count as "sub minutes"
    # if they are absent from that team-match's starting-XI tuple.
    exploded = long.explode("players")
    player_minutes = exploded.groupby(["match_id", "team_id", "players"]).dur.sum().reset_index()
    start_sets_by_tm = starters.to_dict()
    player_minutes["is_sub"] = [
        p not in start_sets_by_tm.get((m, t), ())
        for m, t, p in zip(player_minutes.match_id, player_minutes.team_id, player_minutes.players)
    ]
    sub_share = player_minutes.loc[player_minutes.is_sub, "dur"].sum() / player_minutes.dur.sum()

    return {
        "n_matches": n_matches, "n_teams": n_teams,
        "subs_per_team_match_mean": float(subs_pm.reindex(
            pd.MultiIndex.from_product([seg.match_id.unique(), ["h", "a"]]), fill_value=0).mean()),
        "subs_dist": subs_dist.to_dict(),
        "subs_after_75_share": float((entry_minutes >= 75).mean()) if len(entry_minutes) else np.nan,
        "segments_per_match_median": float(seg_per_match.median()),
        "segment_minutes_median": float(np.median(dur_all)),
        "segment_under_10min_share": float((dur_all < 10).mean()),
        "distinct_onpitch_lineups_per_team_median": float(np.median(per_team_distinct_onpitch)),
        "distinct_starting_lineups_per_team_median": float(np.median(per_team_distinct_starting)),
        "lineup_hhi_per_team_median": float(np.median(per_team_hhi)),
        "sub_minutes_share": float(sub_share),
        "teammate_exposure_corr_mean_abs": float(np.mean(per_team_corr)) if per_team_corr else np.nan,
    }
