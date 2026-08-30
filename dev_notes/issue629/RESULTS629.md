# gcol33/tulpa#629 — the proposal-side lever, measured

The issue asked for one thing before anything is built:

> 1. Whether a fit whose reported k-hat came from the single-Gaussian candidate,
>    on a posterior the mixture or skew-normal candidate fits materially better,
>    is a regime that actually occurs.

Measured. It does not. What the measurement turned up instead is that the
proposal machinery the issue calls "the untouched lever" is fully engaged on the
joint path and **absent from the other three backends**, which report a worse
reliability verdict on 27% of configurations for that reason alone.

## The harness

`dev_notes/issue629/probe629.R`. The device is the one
`tests/testthat/test-joint-pareto-k-*.R` already use: a `res` carrying only
`theta_grid` / `weights` / `axis_offsets` / `blocks`, plus an analytic
`refit_log_marginal`. A `log` axis carries `log_jac = 0`, so the target the PSIS
sees is exactly `g(u)` with `u = log(theta)` — the shape of the hyperparameter
posterior is CONTROLLED, and its true skewness and excess kurtosis are known by
quadrature rather than inferred from a fit.

Eleven targets (gaussian; skew-normal at shape 2 / 5 / 12; Student-t at
df 12 / 8 / 4 / 3 / 2; skew-t at (5, 4) and (12, 3)), spanning true skewness 0
to 6.74 and true excess kurtosis 0 to 141. Fifteen grids per target
(5 / 9 / 15 / 25 / 41 nodes x half-width 3 / 6 / 12 target SDs). Twenty seeds
per cell, at the shipped `k_samples = 500`. 3300 rows, 165 cells, no declines.

The three candidate scorers are wrapped in the namespace so the probe reads the
EXACT objects the shipped dispatch produced — same RNG stream, same draws —
rather than re-deriving them from a fresh seed and comparing two different
realizations. Candidates the dispatch never scored are computed afterwards on
the continuing stream.

**Read cells, not rows.** Taking `min(k_mix, k_skew)` per row and calling the
difference a gain is a selection over two noisy estimates: at row level it
manufactures 68 "materially improvable" cases at a median gain of 0.307. At cell
level (median over 20 seeds) every one of them disappears. The row-level number
is an artifact and is recorded here only so it is not rediscovered as a finding.

## (1) The regime does not occur

Of 165 cells, 18 have a shipped median k-hat at or above the good band. The
largest gain any rescue candidate offers over the shipped choice on any of them
is **0.089**, which crosses no band boundary. On 12 of the 18 the shipped choice
is already the best candidate (gain <= 0).

The four cells with a positive gain are all `5 nodes / half-width 12`, where
`k_mix` is `NA` — the mixture cannot be BUILT, because after the
`.K_DIAG_MIX_FLOOR` cut fewer than two cells carry weight. That is a grid too
coarse to represent the posterior, i.e. a grid problem, and the sweep shows
refinement fixes it (gaussian at that half-width: 0.811 at 5 nodes, 0.159 at
15).

### Why: skewness is not what an outer k-hat measures

The skew-normal rescue is SCORED on 0–5.3% of rows and ADOPTED on 0–1.7%, and
this is not a gate being too tight. On the skewed targets the single Gaussian is
already deep in the good band, so there is nothing to rescue — median
grid-moment k-hat 0.224 at true skewness 0.851, **-0.073 at true skewness
0.967**. A skew-normal target has Gaussian tails on both sides and a lighter one
on the left, so the importance ratio against a Gaussian proposal stays bounded.
What defeats a Gaussian proposal is a heavy TAIL, and there the skew-normal
cannot help by construction — median `k_skew` 0.80 to 1.19 against the
Gaussian's 0.46 to 0.57 on the Student-t targets. The source comment in
`.joint_pareto_score_skew` already says this ("a genuinely heavy-tailed target
keeps its high k-hat here"); the measurement is that the case it CAN absorb does
not produce a bad k-hat in the first place.

So the issue's premise — that on a genuinely skewed hyperparameter posterior "no
amount of grid refinement brings that k-hat down, because a Gaussian cannot fit
the target" — does not hold at the shipped configuration. The premise names the
wrong shape.

## (2) The escalation should not decline; the grid rung is live

Every one of the eleven targets reaches the good band on some grid, including
`skewt(a=12, df=3)` at true skewness 6.74 and excess kurtosis 141 (best cell
median k-hat -1.216). There is no target in the sweep for which the band is
unreachable by grid alone, so a decline rung would be declining on a regime that
does not exist here.

One thing the sweep does say about the rung, on `heavy(df=2)`, median
grid-moment k-hat:

```
        half-width 3   6      12
 5 nodes      1.304  0.792  2.290
 9            1.535  0.410  0.359
15            1.750  0.412  0.128
25            1.929  0.421  0.077
41            2.092  0.433  0.088
```

Densifying a grid that is too NARROW makes the k-hat monotonically worse
(1.304 -> 2.092); widening fixes it (0.077). The shipped `"grid"` rung extends
the boundary where integrand mass piles at an edge and densifies the interior,
so it carries the half that works — but a rung that densified only would move
this fit the wrong way.

## (3) The report already carries the distinction — on one backend

`pareto_k_proposal_source` names which proposal family produced the reported
number, and `.joint_pareto_skew_rescue` records `outer_skew` whether or not the
rescue is adopted. That is the "the posterior is not Gaussian" half of the
distinction the issue asks for, and it exists. It exists only on the joint path.

## What the measurement found instead

`.nested_grid_pareto_k` (`tulpa_nested_laplace()`), `.spde_pareto_k`
(`fit_spde()`) and `.nested_outer_pareto_k` (`tulpa_re_cov_nested()`) each call
`.nested_is_pareto_k` exactly ONCE. They score the raw grid-moment Gaussian and
report its k-hat: no moment-matching loop, no grid mixture, no skew-normal
rescue, and no `proposal_source` field to say so.

Scored over the same 165 cells, the band each read reports:

| layer | good | ok | unreliable |
|---|---|---|---|
| raw grid-moment Gaussian (the three non-joint backends) | 87 | 25 | 53 |
| + moment matching | 90 | 40 | 35 |
| + mixture / skew rescue (the full joint dispatch) | 147 | 10 | 8 |

Of the 53 cells the non-joint read calls `unreliable`, moment matching alone
rescues 18, the two rescues take 27 more, and 8 remain. Median k-hat over those
cells: **1.159 -> 0.736 -> 0.259**. The joint read is never worse on any cell —
it keeps the minimum over candidates, so it cannot be.

So on 45 of 165 configurations (27%) the SAME hyperparameter posterior on the
SAME grid gets a materially better reliability verdict from the joint backend
than from the other three, for no reason except which backend ran it. That is
the "decide it for the concept, not one of four backends" point the issue
raised, with the fix direction inverted: the joint path does not need a fourth
proposal rung, the other three need the dispatch it already has.

Filed separately rather than folded in here.

## An open observation, not a verdict

`dev_notes/issue629/probe_budget.R` and `probe_control.R`. Holding the target,
the grid and the proposal fixed and moving ONLY `k_samples`, the reported k-hat
climbs monotonically on every non-Gaussian target — `skew(a=5)` -0.174 at 200
draws to 3.231 at 20000, `heavy(df=8)` -0.918 to 4.863 — with the ten-seed
ranges at adjacent budgets not overlapping, so it is not seed noise. A
closed-form control (target `N(0, sp^2)` under proposal `N(0, sq^2)`, exact
Pareto weight of index `1/(1 - sq^2/sp^2)`) is FLAT over the same budgets:
k_true 0.5 reads 0.353 at 200 draws and 0.460 to 0.505 from 2000 to 50000. So
the estimator wiring is sound and the climb belongs to the targets.

`?tulpa_nested_laplace_joint` documents `k_samples` as the precision knob —
"more actual tail ratios => tighter k". The measurement says the number itself
moves, not just its interval. This is recorded as an observation and filed
separately; it is not established here what the converged value is, and one
target (`skew(a=2)`) is non-monotone at the largest budget, so nothing in the
#629 verdict above rests on it. Every number in this write-up is read at the
shipped `k_samples = 500`.

## Files

- `probe629.R` -> `sweep629.csv` (3300 rows) — the candidate sweep.
- `probe_cap.R` -> `cap629.csv` — capped vs uncapped scoring at a fixed proposal.
- `probe_budget.R` -> `budget629.csv` — the draw-budget read.
- `probe_control.R` -> `control629.csv` — the closed-form Pareto control.
