# gcol33/tulpa#630 — one candidate dispatch behind all four outer-k backends

The outer Pareto-k-hat asks the same question on every backend: how well does a
tractable proposal cover the hyperparameter posterior the integrator weights.
The answer depended on which backend asked. `tulpa_nested_laplace_joint()`
scored four proposal candidates and kept the best; `tulpa_nested_laplace()`,
`fit_spde()` and `tulpa_re_cov_nested()` each called `.nested_is_pareto_k` once
and reported the raw first pass.

## What was done

`R/outer_pareto_candidates.R` now holds the candidate layer, on a
backend-agnostic contract. `.k_cand_spec()` carries what every scorer needs,
already restricted to the subspace being scored — the target closure `lt`, the
Gaussian proposal `(u_hat, Su)`, the integration nodes `(u_grid, w)` where the
backend has them, the `proposal_source`, and axis `names`. `.k_dispatch()` is
the choice between candidates; `.k_dispatch_report()` adds the reporting fields
and writes the `.kdiag_capture()` aperture from the SELECTED proposal.

Each backend supplies a spec instead of a scorer:

| backend | proposal | nodes | candidates available |
|---|---|---|---|
| `tulpa_nested_laplace_joint()` | grid moments / CCD Hessian / FD rescue | yes | all four |
| `tulpa_nested_laplace()` | grid moments | yes | all four |
| `fit_spde(method = "grid")` | grid moments | yes | all four |
| `fit_spde(method = "ccd")` | mode Hessian | no | Gaussian, moment matching, skew rescue |
| `tulpa_re_cov_nested()` | mode Hessian | no | Gaussian, moment matching, skew rescue |

A spec with no node set WITHHOLDS the grid mixture rather than inventing one:
the mixture's bump width is a grid RESOLUTION, and a CCD design's spacing is not
one. It also leaves the radius cap at `Inf`, which is what those paths always
did.

Two new reported fields, on every backend: `pareto_k_proposal_source` (which
family produced the number) and `pareto_k_first_pass` (the k-hat of the proposal
exactly as the backend placed it, before any candidate refined it). They answer
different questions — the reported k-hat is how reliable the integration is, the
first pass is whether the PLACEMENT was any good — which is the distinction
gcol33/tulpa#629 item (3) asked for and could not make.

## The joint path is unchanged, bit for bit

The extraction is a pure parameter-object substitution, and the arbiter is
#629's own sweep: 3300 configurations x 24 recorded columns, re-run on the
refactored code and compared to the committed baseline. Every column identical,
including `k_gm` / `k_gauss` / `k_mix` / `k_skew` / `k_shipped` and the adopted
source. Nothing about the joint path's numbers moved.

## What it buys, predicted

Scored over #629's 165 synthetic cells (11 hyperparameter-posterior shapes
spanning true skewness 0 to 6.74 and true excess kurtosis 0 to 141, x 15 grids,
x 20 seeds, at `k_samples = 500`):

| layer | good | ok | unreliable |
|---|---|---|---|
| raw grid-moment Gaussian (what three backends reported) | 87 | 25 | 53 |
| + moment matching | 90 | 40 | 35 |
| + mixture / skew rescue | 147 | 10 | 8 |

Of the 53 cells the single-candidate read calls `unreliable`, moment matching
alone rescues 18, the two rescues take 27 more, 8 remain. Median k-hat over those
cells: 1.159 -> 0.736 -> 0.259.

## What it buys, realized on fits

### `tulpa_re_cov_nested()` — the largest change, and it overturns a claim

`probe_recov.R`, on the two fixtures `test-psis.R` uses. First pass against
reported, five seeds each:

| fixture | first pass | reported |
|---|---|---|
| well identified (30 groups x 25 gaussian) | 0.49 / 0.56 / 0.69 / 0.69 / 0.75 | 0.49 / 0.56 / 0.69 / 0.69 / 0.55 |
| tiny binary (25 groups x 3 binary) | 14.6 / 15.1 / 24.3 / 39.4 / 49.1 | 0.29 / 0.61 / 0.63 / 0.78 / 39.4 |

**`CLAUDE.md` recorded that small-group binary RE-covariance posteriors are
genuinely skewed and that a high k-hat there is a correct signal. Measured, four
of those five k-hats were the PROPOSAL's scale, not the posterior's shape.**
Moment matching — re-estimating the proposal from its own PSIS-smoothed
importance-weighted moments, Paananen et al. 2021, a candidate the joint path
has scored all along — takes 14.6-49.1 to 0.29-0.78. The skew-normal rescue is
never even reached (0 of 10 fits), because it fires only above the good band and
moment matching has already cleared it.

The regime is not uniformly benign: one fit of five (seed 203) stays at 39.4, so
some small-group binary posteriors really are beyond a Gaussian's reach. What
changed is that the report now separates those from the four that were only
badly scaled.

`test-psis.R`'s "outer k-hat orders well-identified below tiny-binary" was the
arbiter that claim rested on, and it asserted a two-order-of-magnitude
separation. That separation was the un-refined proposal's blow-up on the tiny
side and does not survive. The test is rewritten to assert what is measured: the
first pass reproduces the old reading (so that half was real), refinement
repairs it, and at least one fit stays past the escalation threshold.

### `tulpa_nested_laplace()`

`probe_backends.R`, ICAR 5x5 binomial, three replicate counts x five seeds:
6 of 15 fits adopt a candidate other than the first pass (moment matching x5,
grid mixture x1), and 2 of 15 cross a band boundary. Median first pass against
reported: 0.796/0.784 at 2 reps, 0.837/0.600 at 4, 0.908/0.851 at 10. Smaller
than the RE-covariance change, and the grid mixture is reached at least once —
all four candidates are live on this backend.

### `fit_spde()`

Three seeds x both methods, binomial SPDE field: no candidate improves on the
first pass on any of them (reported == first pass throughout). The machinery is
available and simply not needed on these fixtures, which is the honest result.

One incidental, from the new `proposal_source` field: two of the three
`method = "ccd"` fits report `grid_moment`, because `fit_spde_nested_ccd()`
falls back to `fit_spde_nested_grid()` in three documented cases. The field
reports which integrator actually ran, which was previously not visible from the
fit.

## Guards

- `test-outer-proposal-lever.R` gains the no-grid spec: the mixture must
  DECLINE, the Gaussian and the rescue must remain, every draw must be evaluated
  (no coverage envelope without nodes), and the first pass must bound the chosen
  k. Plus the sub-floor budget guard — a `n_samples` below `.PSIS_MIN_EVAL` now
  declines in `.k_dispatch()` before any candidate can pay a target evaluation,
  which the mixture would otherwise have done.
- `test-psis.R`'s RE-covariance arbiter, rewritten as above.
- Full tier-1 suite: 16752 passing, 0 failures. The pareto-k files also run
  clean under `NOT_CRAN=true`.

## Files

- `probe_recov.R` -> `recov630.csv` — the RE-covariance layer decomposition.
- `probe_backends.R` -> `backends630.csv` — first pass vs reported on
  `tulpa_nested_laplace()` and `fit_spde()`.
- The bit-identity check re-runs `dev_notes/issue629/probe629.R` and diffs
  `sweep629.csv` against the committed baseline.
