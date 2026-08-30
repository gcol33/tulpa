# gcol33/tulpa#631 — the outer k-hat moves with the draw budget, and why

`?tulpa_nested_laplace_joint` documents `control$k_samples` as the estimate's
precision knob: more tail ratios, a tighter k. Measured, the number itself
moves. This is what it is, and it is not a defect in the estimator.

## The estimator is correct

`tulpa_psis()` reproduces `loo::psis()` to **1e-13 at every budget**, including
the climb and the collapse:

| n | tulpa | loo | diff |
|---|---|---|---|
| 500 | 0.6811 | 0.6811 | 6e-15 |
| 2000 | 1.5338 | 1.5338 | 6e-15 |
| 10000 | 3.4270 | 3.4270 | 1e-14 |
| 50000 | 7.8618 | 7.8618 | 0 |
| 200000 | 0.5667 | 0.5667 | 1e-13 |

So the movement is a property of the PSIS recipe as published, not of this
implementation.

## What moves is the tail FRACTION, not the estimate

`.psis_tail_len(S) = min(S/5, 3 sqrt(S))`. Past `S = 225` the second term binds,
so the fitted tail fraction is `3 / sqrt(S)` and **shrinks as the budget grows**:

| S | tail points | fraction |
|---|---|---|
| 200 | 40 | 20.0% |
| 500 | 68 | 13.6% |
| 2000 | 135 | 6.8% |
| 10000 | 300 | 3.0% |
| 50000 | 671 | 1.3% |
| 200000 | 1342 | 0.7% |

A larger budget therefore describes a DEEPER quantile of the weight
distribution. That is the right thing for loo's setting, where `S` is the
posterior draw count and is essentially fixed for a given fit. It is not the
right thing here, where `k_samples` is a knob both the user and the `k_quality`
escalation turn, and the k-hat is read against FIXED band thresholds (0.5 / 0.7)
and a fixed escalation trigger (`k_usable`).

**Held at a fixed fraction, the k-hat stops moving.** Same targets, same
proposals, same draws; only the tail rule differs (medians over 8 seeds):

| target | rule | 500 | 2000 | 10000 | 50000 | 200000 |
|---|---|---|---|---|---|---|
| skew a=5 | automatic | 0.19 | 0.75 | 2.24 | 5.89 | 0.42 |
| skew a=5 | fixed 20% | 0.030 | 0.118 | 0.101 | 0.102 | 0.097 |
| heavy df=8 | automatic | 0.57 | 1.41 | 3.53 | 7.94 | 0.56 |
| heavy df=8 | fixed 20% | 0.385 | 0.419 | 0.419 | 0.414 | 0.412 |
| heavy df=3 | automatic | -0.21 | 0.57 | 1.19 | 2.42 | 4.82 |
| heavy df=3 | fixed 20% | -0.378 | 0.125 | 0.112 | 0.107 | 0.106 |

At the fraction `k_samples = 500` already implies (13.6%), the same holds and the
seed spread NARROWS, which is what a precision knob does:

| target | 500 | 2000 | 10000 | 50000 |
|---|---|---|---|---|
| skew a=5 | 0.203 | 0.262 | 0.254 | 0.253 |
| heavy df=8 | 0.569 | 0.702 | 0.708 | 0.692 |
| heavy df=3 | -0.194 [-1.67, 0.40] | 0.257 | 0.232 | 0.213 [0.195, 0.243] |

## An independent read of the same exceedances

`probe_tail.R` reads each sample three ways. The Hill estimator — assumption-
light, on the same order statistics — stays between 0.02 and 0.36 on every
target and budget where the GPD shape runs to 7.9. And a regression of the
log-ratio on the squared whitened radius gives a coefficient near zero over the
realized range, i.e. **the weights are nearly bounded where they were actually
sampled**. The ratio only becomes genuinely heavy far beyond the draws, so
neither reading is "the" answer — the tail index of these ratios is a function
of depth, and the automatic rule lets the budget silently choose the depth.

## What was changed

The `k_quality` escalation's precision rung (gcol33/tulpa#627) doubles
`k_samples` on a miss whose bootstrap CI straddles a band boundary. Under the
automatic rule that rung MOVES THE ESTIMAND — the one thing #627's own design
says it must not do; it is the fallback precisely because it only narrows the
interval around whatever the k-hat already is. It now pins the GPD tail size to
the fraction the fit's own first pass used, so the extra draws sharpen the same
number. The fraction is at most 1/5 by construction, so this never trips the 20%
cap warning, and a caller who set `control$k_tail_points` themselves keeps it.

The documentation on all four backends is corrected to say what the knob does.

## What was NOT changed, and why

The obvious next step is to make the outer-k default a fixed tail FRACTION
everywhere, so `k_samples` becomes a precision knob by construction. It is not
taken here, for one measured reason and one unmeasured one:

- **No single fraction preserves the shipped defaults.** `k_samples` defaults to
  200 on `tulpa_re_cov_nested()` (fraction 20%, the `S/5` cap binding) and 500 on
  the joint and grid paths (13.6%). One constant reproduces one of them and moves
  the other; a per-backend constant reproduces both and is more machinery than
  the evidence yet justifies.
- **Which fraction is statistically better is not measured here.** The automatic
  rule exists for a bias-variance reason: a deeper tail is a purer Pareto region
  but has fewer points. Picking 20% because it happens to be stable, or 13.6%
  because it happens to preserve one default, would be choosing a constant by
  what it preserves rather than by what it estimates.

`tulpa_psis()`'s own default stays the reference rule, so the `loo::psis`
equivalence the package maintains as a test oracle is untouched.

## Files

- `probe_tail.R` -> `tail631.csv` — the three-way read (PSIS / Hill /
  closed form) across budgets.
- The budget and closed-form-control tables come from
  `dev_notes/issue629/probe_budget.R` and `probe_control.R`.
- Pinned in `tests/testthat/test-outer-k-budget.R`.
