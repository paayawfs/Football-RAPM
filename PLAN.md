# Five Substitutions and RAPM Identifiability in Football

Project plan. Written 2026-09-11. Every decision below is either locked (agreed with the project lead), verified (checked against live data on 2026-09-11), or a default I chose and marked as such.

---

## 0. Locked decisions and verified facts

### Locked decisions

| Topic | Decision |
|---|---|
| Scope | Whole programme planned now: baseline RAPM, identifiability, stability, prediction, factor RAPM, xT and VAEP outcomes. |
| Data | Open data only. Understat for stage 1. StatsBomb open data for stages 2 and 3. |
| Deliverable | Academic paper. Primary target: Journal of Quantitative Analysis in Sports. Fallback: Journal of Sports Analytics, MathSport, MIT Sloan research track. |
| Factor structure | Partition chain. Lineup xG per minute decomposed into tempo, final-third reach, shots per final-third possession, xG per shot, plus the mirror image against. |
| Team effects | Both. Team-season fixed effects in the primary model. Classic no-team-effect RAPM as comparison. |
| Coefficients | Both. Net (one per player) on xG differential is Model 1. Offence/defence split on xG for and against is Model 2. |
| Segments | Break at substitutions, red cards and goals. Score state is constant within a segment. |
| Rich outcomes | xT grid first, VAEP-style action values after. |
| Language | Python 3.14, pandas, polars, numpy, scipy, scikit-learn, statsmodels, statsbombpy. All already installed. |

### Verified facts (2026-09-11)

**Understat** no longer embeds data in HTML. Two JSON endpoints, both need header `X-Requested-With: XMLHttpRequest`:

```
https://understat.com/main/getLeagueData/{league}/{year}   -> {"teams", "players", "dates"}
https://understat.com/main/getMatchData/{match_id}          -> {"rosters": {"h","a"}, "shots": {"h","a"}, "tmpl"}
```

`league` is one of `EPL, La_liga, Bundesliga, Serie_A, Ligue_1`. `year` is the season start year. `dates` lists every match with `id, isResult, h, a, goals, xG, datetime`.

Roster row fields: `id, player_id, team_id, player, position, h_a, time, roster_in, roster_out, red_card, yellow_card, goals, own_goals, shots, xG, xA, key_passes, assists, xGChain, xGBuildup, positionOrder`. Starters carry a real position code; substitutes carry `position == "Sub"`. `roster_in` on a starter is the roster `id` of the player who replaced them. `roster_out` on a substitute is the roster `id` of the player they replaced. `time` is nominal minutes played and sums to 90 across a replaced pair (checked: 64 + 26).

Shot row fields: `id, minute, result, X, Y, xG, player, player_id, h_a, situation, shotType, season, match_id, h_team, a_team, h_goals, a_goals, date, player_assisted, lastAction`. `result` in `Goal, OwnGoal, SavedShot, MissedShots, BlockedShot, ShotOnPost`. `situation` in `OpenPlay, FromCorner, SetPiece, DirectFreekick, Penalty`.

**Substitution rule timeline**, read from the data by counting `roster_out` rows per team:

| League | 2019-20 pre-lockdown | 2019-20 restart | 2020-21 | 2021-22 | 2022-23 on |
|---|---|---|---|---|---|
| Premier League | 3 | 5 (from 2020-06-17) | 3 | 3 | 5 |
| La Liga | 3 | 5 (from 2020-06-11) | 5 | 5 | 5 |
| Serie A | 3 | 5 (from 2020-06-20) | 5 | 5 | 5 |
| Bundesliga | 3 | 5 (from 2020-05-16) | 5 | 5 | 5 |
| Ligue 1 | 3 | season abandoned | 5 | 5 | 5 |

The Premier League reversal in 2020-21 and 2021-22 is the key source of identifying variation. The pipeline must still infer the era per match from the data (max subs per team per match over a rolling window) and assert agreement with this table.

**StatsBomb open data**, full-league seasons only (single-team showcase seasons excluded):

| Competition | Season | Matches | Teams | Sub rule | Role |
|---|---|---|---|---|---|
| Premier League | 2015-16 | 380 | 20 | 3 | Factor construction, cross-source validation vs Understat |
| La Liga | 2015-16 | 380 | 20 | 3 | Factor construction |
| Serie A | 2015-16 | 380 | 20 | 3 | Factor construction |
| Ligue 1 | 2015-16 | 377 | 20 | 3 | Factor construction |
| FA WSL | 2018-19 | 107 | 11 | 3 | Within-competition era comparison, pre |
| FA WSL | 2019-20 | 87 | 12 | 3 | Within-competition era comparison, pre (cut short by COVID) |
| FA WSL | 2020-21 | 131 | 12 | 5 (verify from data) | Within-competition era comparison, post |
| FA WSL | 2023-24 | 132 | 12 | 5 | Within-competition era comparison, post |
| NWSL | 2023 | 137 | 12 | 5 | Extra five-sub league for truncation and simulation |
| Frauen Bundesliga | 2023-24 | 132 | 12 | 5 | Extra five-sub league |
| Liga F | 2023-24 | 240 | 16 | 5 | Extra five-sub league |
| Serie A Women | 2023-24 | 130 | 10 | 5 | Extra five-sub league |
| Indian Super League | 2021-22 | 115 | 11 | 5 | Extra five-sub league |

Bundesliga 2015-16 and 2023-24, Ligue 1 2021-22 and 2022-23, La Liga 2020-21 and MLS 2023 are single-team sets and are excluded.

---

## 1. Hypotheses and the test that decides each one

| # | Hypothesis | Decisive test | Result table |
|---|---|---|---|
| H1 | Five subs improve identifiability of player effects | Counterfactual truncation (section 5.4) and simulation recovery (5.5). Design-matrix metrics on identical matches with and without subs 4 and 5. | T3, T4 |
| H2 | Player estimates are more stable under five subs | Split-half reliability and bootstrap SD in the league-season panel (6.1, 6.2), with the Premier League reversal as within-league control. | T5 |
| H3 | Better identification gives better out-of-sample prediction | RAPM gain over team-only baseline, within-season and next-season, by era (7). | T6 |
| H4 | Player impact decomposes into a small number of meaningful factors | Split-half reliability of each chain factor and of principal components of the factor vector (8.5). | T8 |
| H5 | xG and possession-value outcomes carry more information than goals | Reliability and next-season prediction of goal-RAPM vs xG-RAPM vs xT-RAPM vs VAEP-RAPM (7, 9). | T6, T9 |
| H6 | Impact magnitude and composition differ by position | Per-position factor profiles and per-position identification share (11). | T10 |
| H7 | Factor identifiability rises with substitution-induced lineup variation | Factor reliability, WSL pre vs post, and truncation experiment on five-sub event datasets (8.6). | T8 |

Three concepts stay separate throughout the paper: identifiability (a property of X), stability (a property of the estimator on resampled data), and validity (out-of-sample prediction).

---

## 2. Data acquisition

### 2.1 Understat scrape

Coverage: five leagues, seasons 2014 to 2025 (2014-15 through 2025-26). Roughly 21,500 matches. Russian league excluded.

Rules:
- One request per second. Cache every response with `requests-cache` (already installed) in a SQLite file under `data/raw/understat/cache.sqlite`, so re-runs cost nothing.
- Headers: a browser `User-Agent` and `X-Requested-With: XMLHttpRequest`.
- Skip matches with `isResult == False`.
- Save per league-season: `matches.parquet` (from `dates`), `rosters.parquet` (one row per roster entry, plus `match_id`), `shots.parquet` (one row per shot). Cast numeric strings at parse time.
- Expected wall time about six hours. Run once overnight with `run_in_background`.
- Cite Understat as the data source in the paper. There is no API licence; academic use of scraped Understat xG is common and should be described plainly.

### 2.2 StatsBomb open data

Via `statsbombpy` with no credentials. Pull `matches`, `lineups` and `events` for every competition-season in the table above. Cache to `data/raw/statsbomb/{competition_id}_{season_id}/`. Observe the open-data licence attribution requirement.

### 2.3 Validation checks on raw data (all must pass before segments are built)

1. Every match in `dates` with `isResult` has a roster with 11 starters per side.
2. For every replaced pair, `starter.time + sub.time == 90` (allow off by one). Log every violation. Expected: near zero.
3. For every team-match, the sum of `time` equals `11 * 90` minus minutes lost to red cards. Violations indicate a roster bug and go to a quarantine list.
4. Final score reconstructed from shots equals the score in `dates`. This also settles the own-goal convention: test whether an `OwnGoal` shot sits in the array of the benefiting side or the conceding side, and pick the rule that reconciles every match. Assert 100 percent.
5. Substitution-era inference per match agrees with the timeline table.
6. StatsBomb 2015-16 Premier League: for every match, the substitution minutes from StatsBomb events match Understat roster times within one minute, and match-level StatsBomb xG correlates with Understat xG at r > 0.9. This is the cross-source check that ties the stages together.

### 2.3 results (run 2026-09-12, all 21,589 played matches across 60 league-seasons)

**Check 1 (11 starters/side)** and **check 2 (replaced-pair minutes)**: the correct test is the chain-resolved interval logic in `rapm/segments.py` (`player_intervals`), not a naive adjacent-pair sum — a substitute later replaced by another substitute does not sum to 90 against the row that replaced *them*, only the full chain telescopes to 90. Verified by construction: `scripts/02_build_segments.py` ran every league-season without a chain-resolution error. One match (La_liga 2023, match 23028) had its entire raw roster duplicated wholesale by Understat — every player appears twice under different roster ids — which briefly showed up as 18-22 "players on the pitch." `build_league_season` now skips any match producing more than 11 players on a side and logs it, rather than only guarding against the specific ids found so far; with that match excluded, every other league-season's `n_home`/`n_away` per segment lands only in {8,9,10,11} (red cards and, rarely, two dismissals on one side).

**Check 3 (team-match minute totals)**: consistent with the interval logic; a red card shortens that team's total by exactly the dismissed player's lost minutes, no unexplained totals found.

**Check 4 (own-goal convention)**: **resolved.** On an `OwnGoal` shot row, `h_a` is the *scoring* (conceding) side, not the side the goal benefits — flip it to get the benefiting side. This reconciles 21,581 of 21,589 matches (99.96%) exactly. The 8 residual mismatches are pre-existing Understat data gaps (three have zero recorded shot events despite a real scoreline) and are quarantined by match ID in `rapm.segments.QUARANTINE`: `{27930, 4238, 5274, 5615, 29482, 5999, 5959, 5894}`. The first two are also the malformed-roster matches found during scraping (section on the Understat scrape below); the segment builder skips all eight.

**Check 5 (era inference)**: the assumed timeline table (section 0) holds, with one genuine wrinkle: 11 Premier League matches in the reverted-to-three-subs 2020-21 and 2021-22 seasons show 4 or 5 substitutions actually used, all dated from February 2021 onward. This is IFAB's permanent concussion-substitute trial (additional to, not part of, the tactical substitution allowance), which the Premier League adopted mid-2020-21 — not evidence the tactical rule reverted early. `era()` correctly reflects the tactical rule in force; a robustness variant that treats a team's 4th/5th substitution in a "three" match as a concussion sub (drop it from the substitution *count* used in identifiability metrics, keep it in the segment data) is worth adding at the robustness stage (section 10).

**Check 6 (StatsBomb cross-validation, Premier League 2015-16, 367 of 380 matches matched by date+team name)**:
- xG correlation: r = 0.907 (home), 0.909 (away). Passes the r > 0.9 bar.
- Substitution timing: does **not** meet the literal "within one minute" bar as originally written — Understat's substitution minute runs a systematic +2.5 minutes later than StatsBomb's, present in effectively every match (mode of the signed difference is +2/+3/+4, only 10% land within one raw minute). After removing that constant offset the residual is tight: mean absolute residual 1.06 minutes, 67% within ±1 minute and 86% within ±2 minutes of the offset. This reads as a genuine provider convention difference (StatsBomb logs the substitution event itself; Understat appears to round to the next stoppage in play), not a data error in either source. Documented here rather than forcing the original threshold; the corrected check is "within one minute of the fitted offset," which passes.

---

## 3. Repository layout

```
RAPM/
  PLAN.md
  pyproject.toml            # deps pinned; ruff config
  data/
    raw/understat/{league}/{year}/{matches,rosters,shots}.parquet
    raw/statsbomb/{cid}_{sid}/{matches,lineups,events}.parquet
    processed/segments.parquet
    processed/segments_sb.parquet
  rapm/
    understat.py            # fetch + parse, nothing else
    segments.py             # roster + shots -> segments (section 4)
    design.py               # segments -> X, W, y, column index
    ridge.py                # weighted ridge with partial penalty, grouped CV
    identify.py             # metrics from section 5
    simulate.py             # section 5.5
    stability.py            # section 6
    predict.py              # section 7
    factors.py              # StatsBomb events -> chain outcomes (section 8)
    xt.py                   # section 9
  scripts/
    01_scrape_understat.py
    02_build_segments.py
    03_fit_baseline.py
    04_identifiability.py
    05_stability_prediction.py
    06_factors.py
    07_xt_vaep.py
    08_robustness.py
  tests/
    test_segments.py        # hand-built match, asserts every convention in 4.2
  paper/
    tables/  figures/  main.tex  refs.bib
```

Every script reads parquet in and writes parquet or CSV out. No notebooks in the pipeline. Notebooks are for looking, not for producing paper numbers.

---

## 4. Segments and design matrix

### 4.1 Player on-pitch intervals from a roster

For each roster row compute `entry` and `exit` in nominal minutes:

- Starter (`position != "Sub"`): `entry = 0`.
- Substitute: `entry = exit of the row whose id == roster_out`. Resolve in chain order, since a substitute can be replaced by another substitute.
- `exit = entry + time` for everyone.
- Red card (`red_card == 1`): `exit = entry + time`, no replacement.
- Unused substitutes have `time == 0` and no `roster_out`. Drop them.

### 4.2 Breakpoints and segment conventions

A shot recorded at `minute == m` happened during `[m, m+1)`. A player with `time == t` left at the end of minute `t`, so the boundary is `t`.

Breakpoints for a match = `{0, 90}` ∪ `{exit minutes of subbed-off and sent-off players}` ∪ `{m + 1 for every goal and own goal at minute m}`.

Conventions, each asserted by `tests/test_segments.py` on a hand-built match:

1. Segment `i` spans `[b_i, b_{i+1})`. Duration `b_{i+1} - b_i`. Zero-duration segments are dropped.
2. A shot at minute `m` belongs to the segment with `b_i <= m < b_{i+1}`. A goal at minute `m` therefore lands in the segment ending at `m + 1`, and the score state changes from `m + 1`.
3. Shots with `minute >= 90` are stoppage time and are assigned to the final segment.
4. A player is on the pitch in segment `i` if `entry <= b_i` and `exit >= b_{i+1}`.
5. A substitute who enters at 90 gets zero exposure and no column contribution in that match.
6. Score state at segment start = home goals minus away goals among goals with `m + 1 <= b_i`.
7. Own goals count as goals for the benefiting side in the score state and in the goal-difference outcome, but their xG is excluded from the xG outcome.
8. Penalties are included in the primary xG outcome. Non-penalty xG is a robustness outcome.
9. Stoppage time is unobserved. Segment lengths are nominal. Stated as a limitation.
10. Verified on Premier League 2022-23: substitutes entering in stoppage time carry `time` of 1 to 4 while the player they replaced carries 90, so team minute totals run from 990 to 994. Their `entry` is 90 and they get zero exposure (convention 5). Substitute-replaces-substitute chains occur in about 1 percent of substitutions and resolve correctly through `roster_out`. Concussion substitutes give a handful of team-matches six substitutions; they are kept and counted as substitutions.

Segment table columns: `match_id, league, season, era, date, seg_idx, start, end, dur, home_players (list), away_players (list), n_home, n_away, score_state, xg_home, xg_away, goals_home, goals_away, npxg_home, npxg_away`.

### 4.3 Net model (Model 1)

One row per segment.

- Outcome `y = 90 * (xg_home - xg_away) / dur`. Goal version replaces xG with goals.
- Weight `w = dur`.
- Player columns: `+1` if on the pitch for the home side, `-1` if for the away side, `0` otherwise.
- Controls, unpenalised: intercept per league-season (this is home advantage), score-state dummies from the home perspective in `{<=-2, -1, +1, >=+2}` with 0 as reference, minute-bucket dummies for `[0,15), [15,30), [30,45), [45,60), [60,75), [75,90]` with the first as reference, man-advantage `n_home - n_away` as a numeric column, and a ghost-games flag for matches played without crowds (2020-05-16 to 2021-05-23, refined per league from attendance records if available).
- Team-season columns: `+1` for the home team, `-1` for the away team. Unpenalised in the primary specification. Dropped entirely in the classic-RAPM comparison.

Note on interpretation: with team-season effects included, the sum of a team's player columns in any segment equals its player count times its team column. Ridge on the player columns resolves this, and each player effect is read as a deviation from the team's on-pitch average. This is the specification in which substitutions do all the identifying work, so it is primary.

### 4.4 Offence/defence model (Model 2)

Two rows per segment, one per attacking side.

- Outcome `y = 90 * xg_for / dur` for the attacking side. Weight `dur`.
- Columns `O_j = +1` for the attacking side's players on the pitch, `D_j = +1` for the defending side's players.
- Controls: intercept per league-season, home-attacking flag, score state from the attacking side's perspective, minute bucket, man advantage `n_att - n_def`, ghost flag, team-season O and D columns.
- One shared penalty on O and D in the primary fit. Separate penalties as robustness.

### 4.5 Ridge solver

Minimise `sum_s w_s (y_s - x_s b)^2 + lambda * sum_{j in penalised} b_j^2`.

- Per league-season and per era-league block the problem is small (about 550 players, 3,000 to 5,000 rows), so form `M = X^T W X` densely with numpy and solve `(M + lambda D) b = X^T W y`, where `D` is diagonal with 1 on penalised columns and 0 elsewhere. `M` is needed anyway for the identifiability metrics.
- The pooled multi-season fit uses scipy sparse matrices, row-scaled by `sqrt(w)`, augmented with `sqrt(lambda)` rows on penalised columns, solved with `scipy.sparse.linalg.lsqr`.
- Lambda selection: `GroupKFold(5)` grouped by `match_id`, 25-point log-spaced grid, choose the minimiser of weighted CV MSE. Record the one-standard-error lambda as well. For any comparison across eras also report every metric at one common lambda chosen on the pooled data, so lambda differences cannot drive the comparison.
- Residual variance `sigma^2` is the weighted mean squared residual at the chosen lambda. Prior variance `tau^2 = sigma^2 / lambda` by the ridge-as-posterior-mode correspondence.

### 4.3-4.5 status (built and verified 2026-09-12)

`rapm/design.py` (`build_net_design`, `build_od_design`) and `rapm/ridge.py` (`fit_dense`, `fit_sparse`, `select_lambda`) are built, tested (`tests/test_design_ridge.py`, seven cases including a synthetic simulation-recovery check), and run on real data. Both models always build a sparse `X` — cheap for one league-season, necessary for the pooled panel — with player and team-season columns collapsed to one signed column per entity even when built from two one-sided blocks (a player who appears as both a home and away player in the same block must get a single shared coefficient, not two).

One real bug surfaced and fixed along the way: an unpenalised control column that is identically zero within a block — `ghost_games` for any pre-2020 league-season, since it never has a COVID-crowds match — made `M + D` exactly singular. Both fit functions now drop an all-zero unpenalised column from the solve (its coefficient is 0 by convention, since nothing in the block can identify it) rather than failing; a near-zero-but-nonzero column is untouched.

Verification run, Premier League 2022-23 (`scripts/03_fit_baseline.py`), Model 1 (net), dense path:

| | |
|---|---|
| Design matrix | 3,208 rows x 574 columns (562 penalised: 542 players + 20 team-seasons; 12 controls) |
| `lambda_min` (5-fold `GroupKFold`) | 5,623 |
| Weighted R² vs. mean-only | 0.062 |
| Condition number, penalised block of `M` (pre-ridge) | 1.3e21 |

That condition number is not a bug — it is the headline fact motivating this whole project: at a single league-season, the raw player-plus-team design is so collinear it is numerically singular in double precision (machine epsilon is ~1e-16), because a team's own eleven players on the pitch sum almost exactly to eleven times that team's own dummy column. Ridge resolves it, but this is exactly the pre-ridge conditioning that section 5's identifiability metrics (condition number, effective rank, CV-optimal lambda) are built to track pre/post five-substitution.

`lambda_1se` hit the top of the default `1e-1` to `1e5` grid on this block. Re-run with the grid extended to `1e8`: `lambda_min` is a genuine interior minimum (5,456, confirmed by the CV curve turning back up past it), but `lambda_1se` still lands on the boundary no matter how far the grid extends, because the CV-MSE curve asymptotes to a ceiling (~18.29) as `lambda -> infinity` — i.e. the fully-shrunk-to-zero, players-and-teams-contribute-nothing model — and that ceiling sits within one SE of the minimum (17.66). This is not a solver artifact; it is direct evidence that a single league-season alone cannot statistically distinguish "no player or team effects" from the best-fitting model, which is exactly the weak-identifiability problem this project exists to characterise. It says nothing yet about the pooled multi-season panel, where far more rows should narrow that band; the grid should still default wider (`1e-1` to `1e7`, say) so `lambda_min` is never mistaken for a boundary artifact there, but `lambda_1se` sitting at the edge for a single block is a result, not a bug.

Also verified: `fit_dense` and `fit_sparse` agree to ~1e-9 on identical inputs (confirms the augmented-lsqr sparse path is an exact solve, not an approximation), and the O/D model (Model 2) fits on the same data with plausible, bounded O/D coefficients.

---

## 5. Identifiability analysis (primary contribution)

All metrics are computed on the player block after partialling out the unpenalised controls: `X~ = X_P - C (C^T W C)^{-1} C^T W X_P`, where `P` is the set of players with at least 450 minutes in the block and `C` is the control matrix including team-season columns. This restricts identification to within-team lineup variation. `M = X~^T W X~ / sum(w)` is normalised per minute so blocks of different size are comparable in shape. Eigenvalues `e_1 >= ... >= e_p`.

### 5.1 Descriptive lineup variation (Table 1, Figure 1)

Per team-match and per league-season:
- substitutions used per team-match (mean, distribution 0 to 5);
- substitution minute distribution (histogram by 5-minute bin) and share after minute 75;
- segments per match; minutes per segment (median, share under 10 minutes);
- distinct starting elevens per team-season and distinct on-pitch elevens per team-season;
- lineup concentration: Herfindahl index of minutes across distinct on-pitch elevens per team-season;
- minutes played by substitutes as a share of all minutes;
- mean absolute correlation between teammate exposure columns within team.

### 5.2 Design-matrix metrics (Table 2, Figure 2)

- Condition number `e_1 / e_p`, and a trimmed version `e_1 / e_{0.9p}`.
- Effective rank `exp(-sum p_i log p_i)` with `p_i = e_i / sum e`.
- Share of total variance in the smallest 10 percent of eigen-directions.
- Full spectrum plot, log scale, pre vs post, at matched match counts.
- CV-optimal lambda.
- Effective degrees of freedom `tr(X~ (M + lambda I)^{-1} X~^T W)` at the common lambda.
- Identification share per player: diagonal of `(M + lambda I)^{-1} M`. 1 means fully data-determined, 0 means pure prior. Report median, quartiles and the share of players above 0.5. This is the headline per-player metric.
- Contrast precision: for the 200 teammate pairs with the most shared minutes, `SE(b_i - b_j) = sigma * sqrt(c^T (M + lambda I)^{-1} M (M + lambda I)^{-1} c)`, reported relative to `tau`.

All block-level metrics are computed at matched match counts: subsample the larger block to the smaller block's match count, 100 draws, report the median and the 5 to 95 percent range.

### 5.3 Era comparison in the league-season panel (Table 3, Figure 3)

Unit: league-season, full seasons only. 2019-20 enters as its pre-lockdown matches only (a three-sub unit). Ligue 1 2019-20 is its 279 played matches.

Regression for each metric: `metric = alpha_league + gamma_season + delta * five_subs + error`. With 5 leagues and 12 seasons inference is thin, so this is presented as an event-study figure (metric relative to adoption year, Premier League reversal drawn separately) with match-level block-bootstrap intervals on each point, not as a regression with asymptotic standard errors. Causal weight rests on 5.4 and 5.5.

### 5.4 Restart experiment and counterfactual truncation (Table 3, Table 4)

**Restart experiment.** For each of the four leagues that restarted in 2020, compare the final `N` pre-lockdown matches with the `N` restart matches, same squads, same season. Report every metric in 5.1 and 5.2. This is the cleanest within-season switch from three to five subs.

**Counterfactual truncation.** This is the primary mechanism test for H1. For every five-sub match, keep each team's first three substitutions in chronological order and delete the fourth and fifth: the player who would have left stays on to 90, and the entrant never appears. Rebuild segments and the design matrix on these identical matches. Compare every metric between the actual and truncated designs. Confidence intervals by bootstrapping matches (500 draws). Outcomes cannot be reused under truncation because the player set differs, so this is a design-only comparison, which is exactly what identifiability is.

Two variants: truncate to three subs but keep the last three (tests whether late subs carry the information), and delete all substitutions after minute 75 in both eras (tests whether the gain is purely late-game).

### 5.5 Simulation recovery (Table 4, Figure 4)

For each block's design (pre-era, post-era, post-era truncated):
1. Draw player effects `b* ~ N(0, tau^2)`, with `tau` calibrated from the real pooled fit and also at `0.5 tau` and `2 tau`.
2. Controls fixed at their fitted values.
3. Noise `e_s ~ N(0, sigma^2 / w_s)`, `sigma` calibrated from real residuals.
4. `y = X b* + C g + e`.
5. Fit the ridge at the oracle lambda `sigma^2 / tau^2` and at the CV-selected lambda.
6. Metrics on players with at least 450 minutes: RMSE of `b^ - b*`, Spearman correlation, precision at 20 (overlap of the true and estimated top 20), sign agreement, coverage of nominal 90 percent intervals.
7. 200 replications per block and calibration.

Because only X differs across arms, nothing about era-specific play styles, xG model drift or COVID can affect this comparison. If the truncated design recovers effects as well as the actual design, H1 is rejected regardless of what the real-data stability shows.

### 5.1-5.2 status (built and run 2026-09-12, all 60 league-seasons)

`rapm/descriptive.py` (section 5.1) and `rapm/identify.py` (section 5.2) are built, tested (`tests/test_identify.py`), and run across every league-season, writing `data/processed/table1_descriptive.parquet` and `table2_identifiability.parquet`. Two real bugs surfaced and were fixed along the way:

1. **Parquet schema corruption.** Once all 60 league-seasons' segments were concatenated (but not on any single block, nor a two-block test), pyarrow's type inference silently widened the `home_players`/`away_players` list columns from `list<int64>` to `list<double>` — every player id came back as e.g. `314.0`. `scripts/02_build_segments.py` now casts those two columns to `list<int64>` explicitly before writing, rather than depending on inference.
2. **A second, more general collinearity than the one already fixed in phase 2.** Every 2020-21 league-season (Bundesliga, EPL, La Liga, Ligue 1, Serie A) hit the same `LinAlgError: Singular matrix` inside CV, for a new reason: that whole season falls 100% inside the ghost-games window, so `ghost_games` becomes identical to the block's own single intercept column — two *nonzero* unpenalised controls colliding, not the all-zero case already handled. `ridge._drop_degenerate` now does a general Gram-Schmidt independence pass over the unpenalised columns (cheap: there are only ~12 of them, even against a large pooled `X`) rather than only checking for exact zero.

**Table 2's `condition_number` (untrimmed) is `inf` for all 60 of 60 league-seasons.** This is not a bug: essentially every team-season has at least one truly ever-present player (a #1 goalkeeper who plays every minute of every match is the obvious case), whose within-team-projected column is then *exactly* zero. `condition_number_trimmed` (`e_1 / e_{0.9p}`), effective rank, and identification share are therefore the metrics doing the real work in every comparison below, exactly as section 5.2 anticipated in specifying the trimmed version.

**Raw era comparison, pooled across all five leagues and 60 league-seasons (32 three-sub, 28 five-sub), no matched-match-count subsampling or controls yet:**

| Metric | Three-sub era | Five-sub era | Direction vs. H1 |
|---|---|---|---|
| Subs used per team-match | 2.91 | 4.41 | mechanical, as expected |
| Distinct on-pitch lineups per team-season (median) | 123.8 | 134.3 | ✓ more variation |
| Lineup Herfindahl per team-season (median) | 0.0188 | 0.0178 | ✓ less concentrated |
| Substitute minutes, share of all minutes | 0.056 | 0.077 | ✓ more sub exposure |
| Mean abs. teammate exposure correlation | 0.146 | 0.133 | ✓ less collinear |
| Condition number, trimmed (mean) | 190.1 | 146.8 | ✓ better conditioned |
| Effective rank (mean) | 205.9 | 215.7 | ✓ higher |
| Effective degrees of freedom (mean) | 39.6 | 41.6 | ✓ higher |
| Identification share, median (mean across blocks) | 0.107 | 0.111 | ✓ higher, but tiny |
| Identification share, share of players > 0.5 | 0.0 | 0.0 | no season, either era, gets even one player past 0.5 |

Every metric moves in the direction H1 predicts. This is worth taking seriously but not yet as evidence: it is the raw, unadjusted, pooled-across-leagues comparison section 5.3 explicitly warns against over-reading (five-sub seasons skew later, so this comparison cannot separate the rule change from any other trend over time — league mix, tactical evolution, the pandemic itself). It is also a small effect in absolute terms — even in the best case, no league-season gets a single player above 0.5 identification share, i.e. even under five subs the data still trusts the ridge prior more than the data for every player in a single season. The real test is 5.4's within-season restart comparison and counterfactual truncation, which hold everything except the substitution rule fixed; that, plus the matched-match-count/bootstrap machinery of 5.3, plus simulation recovery in 5.5, is the next phase of work.

---

## 6. Stability

### 6.1 Split-half reliability (Table 5)

Within each league-season, split matches into two halves at random, stratified so each team has about the same number of matches in each half. 100 splits. Fit both halves at the common lambda. Report Spearman correlation between halves for players with at least 450 minutes in both, RMSE of the difference, and Jaccard overlap of the top 20. Also at matched match counts across eras.

### 6.2 Match bootstrap

500 resamples of matches with replacement per league-season. Per-player standard deviation of the estimate, sign stability (share of resamples with the full-sample sign), and rank stability of the top 50 (mean absolute rank change).

### 6.3 Cross-season

Spearman correlation between season `t` and `t + 1` estimates for players with at least 900 minutes in both. Report alongside the split-half figure so real player change can be separated from estimation noise: the ratio of cross-season to split-half correlation is the stability of true quality, and only the split-half figure speaks to H2.

---

## 7. Prediction

### 7.1 Within-season

`GroupKFold(5)` by match. Predict segment outcomes, aggregate to match xG differential. Metrics: RMSE, R squared, Spearman of predicted vs actual match xG differential. Baselines: home advantage only; team-season effects plus home advantage. Report the gain of each RAPM variant over the team-only baseline with match-level bootstrap intervals.

### 7.2 Next season

Estimate on season `t`. Predict every match in `t + 1` using its actual lineups, the season-`t` player estimates (new players at 0, which is the prior mean) and season-`t` team effects. Baseline: previous-season team xG differential per match. Metric: RMSE and its gain over the baseline. Compare goal-RAPM, xG-RAPM and later xT-RAPM and VAEP-RAPM predicting both next-season goal differential and xG differential. This is the H5 test.

### 7.3 By era

The prediction gain over team-only is a league-season quantity. It enters the same event-study presentation as the identifiability metrics. H3 predicts the gain rises after adoption and falls back during the Premier League reversal.

---

## 8. Factor RAPM (partition chain)

### 8.1 Possessions from StatsBomb events

A possession is StatsBomb's `possession` id with its `possession_team`. Per possession record: team, start minute, end minute, `reached_final_third` (any event by the possession team with start location `x >= 80` on the 120 by 80 pitch), `shots` (count of Shot events, penalties flagged), `xg` (sum of `shot_statsbomb_xg`), and where the possession ended (thirds of the pitch, from the location of the last event by the possession team).

A possession belongs to the segment containing its start minute. Segments for StatsBomb matches are built from `lineups` (positions with `from` and `to` times) and substitution and card events, with the same conventions as section 4.2, and the segment builder is the same function.

### 8.2 The chain

For a team over a segment: `xG per minute = N * F * S * Q` where

| Symbol | Definition | Regression outcome | Weight | Conceptual dimension |
|---|---|---|---|---|
| N | possessions per minute, both teams | `90 * (n_home + n_away) / dur` | `dur` | tempo (shared, treated as neutral) |
| F | share of possessions reaching the final third | `f / n` | `n` | progression |
| S | shots per final-third possession | `sh / f` | `f` | creation |
| Q | xG per shot | `xg / sh` | `sh` | shot quality |

Each of F, S and Q is fitted as an offence/defence model (section 4.4) with the same controls and team-season effects, giving six player factors: F, S, Q for and against. Two optional extension factors, not in the primary specification: R (share of own possessions ending in the own defensive third, retention) and W (share of opponent possessions ending in their defensive third, disruption).

### 8.3 Aggregation without free weights

First-order expansion around league means: offensive value of player `j` is

`V_off_j = Nbar * Fbar * Sbar * Qbar * (b_F_j / Fbar + b_S_j / Sbar + b_Q_j / Qbar)`

and likewise for defence with the against-factors and a negative sign. The weights are league means, so nothing is chosen. Check: correlation between `V_off_j` and the direct xG-for RAPM coefficient should be high, and the residual measures how much the chain loses through interaction terms.

### 8.4 Datasets

- Primary factor construction and validation: the four 2015-16 big-league seasons, 1,517 matches, three-sub era.
- Within-competition era comparison: WSL 2018-19 and 2019-20 (three subs) vs WSL 2020-21 and 2023-24 (five subs). Verify the 2020-21 rule from substitution counts.
- Truncation and simulation on five-sub event data: WSL 2020-21, WSL 2023-24, NWSL 2023, Frauen Bundesliga, Liga F, Serie A Women 2023-24, ISL 2021-22.
- Cross-source check on Premier League 2015-16 as in section 2.3.

### 8.5 How many factors are real (Table 8)

- Correlation matrix of the six factor estimates across players with at least 450 minutes, on each dataset.
- Split-half reliability of each factor separately (section 6.1 procedure).
- Principal components of the standardised six-vector on half A; project half B; reliability of each component score across halves. The number of components with split-half reliability above 0.3 is the reported number of distinguishable dimensions. Parallel analysis against permuted data as a second criterion.
- Prediction: does the factor model, aggregated by 8.3, predict held-out xG differential as well as or better than the direct aggregate RAPM.

### 8.6 Factor identifiability and substitutions (H7)

Repeat 5.2, 5.4 and 5.5 on each factor's design. The designs differ across factors only through their weights (`n`, `f`, `sh`), so the shot-quality factor Q has far fewer effective observations than F. Report identification share per factor and how much of the WSL pre-post change survives truncation.

---

## 9. xT and VAEP outcomes

### 9.1 xT

Convert StatsBomb events to SPADL and fit a 12 by 8 xT grid on the pooled open-data events (all full-league seasons above), iterating the Markov value to convergence. Use `socceraction`, which implements SPADL conversion, xT and VAEP. Its Python 3.14 compatibility is unverified; if it does not install, create a Python 3.12 virtual environment for stage 3 only.

Segment outcome: sum of xT gained by successful passes, carries and dribbles per team per 90, for and against. Fit the offence/defence RAPM. Compare against xG-RAPM on split-half reliability and next-season prediction of xG differential.

### 9.2 VAEP

VAEP with the standard ten-action window, using `sklearn.ensemble.HistGradientBoostingClassifier` as the scoring and conceding models. Train on the four 2015-16 seasons, check calibration on the WSL seasons (Brier score, reliability plot). Segment outcome: sum of VAEP values per team per 90. Fit the RAPM. This is the only outcome that charges failed actions, so it is where retention effects can appear.

### 9.3 Comparison (Table 9)

Rows: goal, xG, xT, VAEP outcomes. Columns: split-half reliability, identification share, next-season prediction of xG differential and of goal differential, all on the same matches.

---

## 10. Robustness and placebo

- Placebo adoption dates: assign five-sub status to Premier League 2017-18 and 2018-19 in the panel and confirm the estimated effect is near zero.
- Placebo mechanism: compare pre-era matches with exactly three subs used to post-era matches with exactly three subs used. Any metric difference here is era, not rule.
- Outcome: goals instead of xG; non-penalty xG; own goals included in xG.
- Lambda: one-standard-error lambda; fixed lambda across eras; separate offence and defence lambdas.
- Minute threshold: 900 instead of 450.
- Team effects: classic RAPM without team-season columns.
- Sample: exclude 2020-21 entirely; exclude Ligue 1; exclude the restart matches.
- Late subs: delete all substitutions after minute 75 in every era and rerun the panel.

---

## 11. Position analysis

Map Understat position codes to eight groups: `GK`; `DC` to centre back; `DL, DR, DML, DMR` to full back; `DMC` to defensive midfielder; `MC, ML, MR` to central midfielder; `AMC` to attacking midfielder; `AML, AMR, FWL, FWR` to winger; `FW` to striker. A player's position for a season is the modal starting position. StatsBomb positions map from `lineups` position names to the same eight groups.

Per group report: distribution of estimates, identification share, split-half reliability, and the era effect on each. For the factor stage: mean factor profile per group, and per-group reliability of each factor. Positions are described, not imposed as constraints.

---

## 12. Pre-registration

Register on OSF after the pipeline runs end to end on one league-season and the simulation machinery works, but before any real-data era comparison is examined. Contents:

- H1 to H7 as stated in section 1.
- Primary identifiability metric: median identification share among players with at least 450 minutes.
- Primary stability metric: split-half Spearman at matched match counts.
- Primary prediction metric: next-season RMSE gain over the team-only baseline.
- Primary mechanism test: counterfactual truncation on all five-sub matches, 500-match bootstrap, comparing median identification share.
- Lambda selection rule, thresholds, segment conventions, control set, own-goal and penalty handling, exactly as written here.
- Decision rule: H1 is supported if the truncation experiment shows a lower median identification share with 95 percent interval excluding zero difference, and the simulation shows higher RMSE under truncation.

---

## 13. Paper

Target: Journal of Quantitative Analysis in Sports. Structure follows the outline already drafted (introduction, literature, data, methodology, results, robustness, discussion, conclusion).

Tables:
- T1 Descriptive substitution and lineup variation by league-season.
- T2 Design-matrix metrics by league-season at matched match counts.
- T3 Restart experiment and panel event-study estimates.
- T4 Truncation experiment and simulation recovery.
- T5 Split-half reliability and bootstrap stability by era.
- T6 Prediction gains within-season and next-season by outcome and era.
- T7 Factor RAPM correlations and chain aggregation check.
- T8 Factor reliability and number of distinguishable dimensions.
- T9 Outcome comparison: goals, xG, xT, VAEP.
- T10 Position-group profiles.
- T11 Robustness grid.

Figures:
- F1 Substitution timing histograms, pre vs post, and segment length distributions.
- F2 Eigenvalue spectra, pre vs post, at matched match counts.
- F3 Event-study of identification share around adoption, Premier League reversal highlighted.
- F4 Simulation recovery curves against tau.
- F5 Factor correlation heatmap and split-half reliability per component.
- F6 Example player profiles from the chain factors.

---

## 14. Timeline

Weeks of focused work. Each phase ends with its checks passing.

| Phase | Weeks | Work | Done when |
|---|---|---|---|
| 0 Setup | 1 | Repo layout, pyproject, scraper, StatsBomb cache | Both fetchers cached one league-season and one competition-season |
| 1 Data | 1-3 | Full Understat scrape, all validation checks in 2.3 | Checks 1 to 6 pass; quarantine list documented |
| 2 Segments and baseline | 3-5 | Segment builder, test, design matrices, ridge, CV | `test_segments.py` passes; Model 1 and 2 fit on EPL 2022-23 with sensible lambda; cross-source check passes |
| 3 Identifiability | 5-8 | Sections 5.1 to 5.5 on all league-seasons | T1 to T4 produced; simulation reproduces known effects on synthetic data |
| 4 Stability and prediction | 8-10 | Sections 6 and 7 | T5, T6 produced |
| Pre-registration | 10 | Section 12 | OSF record created before the era tables in T3 to T6 are read |
| 5 Factor RAPM | 11-14 | Section 8 | T7, T8 produced; aggregation check passes |
| 6 xT and VAEP | 15-18 | Section 9 | T9 produced; VAEP calibration acceptable |
| 7 Robustness and positions | 19-20 | Sections 10 and 11 | T10, T11 produced |
| 8 Paper | 21-24 | Writing, figures, internal review | Submission-ready draft |

---

## 15. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Understat endpoint or field change during the project | Medium | Cache everything on first scrape; parser asserts field names; the scrape runs once |
| Understat roster times inconsistent for some matches | Medium | Quarantine list; check 3 in 2.3; exclude affected matches and report count |
| Own-goal convention ambiguous | Low | Settled by score reconciliation, check 4 |
| Five-sub effect on lineup variation is small because subs cluster late | Real possibility | This is a legitimate finding under pre-registration; the late-sub variants in 5.4 quantify it |
| WSL 2020-21 did not use five subs | Low | Verified from substitution counts before analysis; fall back to WSL 2023-24 only |
| `socceraction` does not install on Python 3.14 | Medium | Python 3.12 venv for stage 3 |
| Pooled X too large for dense algebra | Low | Dense only per block; sparse `lsqr` for pooled fits |
| Referees and reviewers ask for causal language on the panel | Certain | The paper claims causality only for truncation and simulation; the panel is descriptive |
