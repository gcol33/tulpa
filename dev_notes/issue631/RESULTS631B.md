# gcol33/tulpa#631, part two -- making the budget a precision knob

`RESULTS631.md` established the defect (the outer k-hat moves with
`control$k_samples`, which is documented as a precision knob), showed the
estimator is not what moves, and pinned the `k_quality` escalation's precision
rung so that rung at least does not move the estimand. It declined the general
fix for two reasons. One is now void and one turned out not to bind.

## The blocker is gone

> No single fraction preserves the shipped defaults. `k_samples` defaults to 200
> on `tulpa_re_cov_nested()` (fraction 20%) and 500 on the joint and grid paths
> (13.6%). One constant reproduces one of them and moves the other.

gcol33/tulpa#632 consolidated the budget to ONE default, 500. (That quote is also
wrong about the grid path, which read the registry's 200 -- the split was joint
against all three siblings. #632 records the correction.) With one default there
is one fraction to hold, and holding it reproduces the shipped default exactly.

## The second reason does not bind, because the fraction is not chosen

> Which fraction is statistically better is not measured here. Picking 20%
> because it happens to be stable, or 13.6% because it happens to preserve one
> default, would be choosing a constant by what it preserves rather than by what
> it estimates.

The shipped rule does not choose one. It INHERITS the fraction from
`.nl_diag("k_samples")`, so the default fit is the published rule evaluated at
the default budget, unchanged to the bit, and every other budget is measured
against that same estimand. Which fraction is best remains unanswered and
unneeded: the property being bought is that one fit's band does not depend on
another fit's cost knob.

## The rule

`.k_outer_tail_points(n_samples, tail_points)` (`R/psis.R`), resolved once in
`.k_dispatch()` -- the single candidate loop gcol33/tulpa#630 put behind all four
outer backends, so all four inherit it.

    held = floor(n * .psis_tail_len(ref) / ref),  ref = .nl_diag("k_samples")
    used = max(held, .psis_tail_len(n))

The `max()` is load-bearing and was NOT in the first version. Below the
reference budget the published rule is in its `S / 5` regime and is the MORE
generous of the two, so taking the fraction as a replacement bought a stable
estimand by making every cheap diagnostic noisier:

| S | published | held | shipped (max) | fraction |
|---|---|---|---|---|
| 100 | 20 | 13 | 20 | 20.0% |
| 200 | 40 | 27 | 40 | 20.0% |
| 500 | 68 | 68 | 68 | 13.6% |
| 1000 | 95 | 136 | 136 | 13.6% |
| 4000 | 190 | 544 | 544 | 13.6% |
| 50000 | 671 | 6800 | 6800 | 13.6% |

So no budget is fitted on fewer tail points than it is today, and the fitted
fraction is confined to `[13.5%, 20%]` over a 500x range of budgets instead of
collapsing (20% -> 1.3% at 50000, 0.7% at 200000). The cost of getting this
wrong was measured, not imagined: the first version crossed a reported band on
`test-joint-pareto-k-proposal.R`'s 200-draw per-arm fixture.

## What it changes, measured on the engine's own skew fixture

`test-outer-skew-rescue.R` scored a skewness-0.9 target -- the shape a variance
component has -- and asserted the symmetric proposal reads UNRELIABLE and the
skew-normal rescue repairs it. Same target, same spec, five seeds, median:

| budget | tail (published) | k (published) | tail (held) | k (held) | adopted (published) |
|---|---|---|---|---|---|
| 500 | 68 | 0.017 | -- | 0.017 | mode_hessian |
| 1000 | 95 | 0.290 | 136 | 0.132 | mode_hessian |
| 2000 | 135 | 0.597 | 272 | 0.201 | skew_normal |
| 4000 | 190 | 0.914 | 544 | 0.142 | skew_normal |
| 10000 | 300 | 1.627 | 1360 | 0.169 | moment_matched |

At the shipped budget that target reads **0.017**. The fixture reached the
rescue only because it scores at 4000, eight times the default. That is
gcol33/tulpa#629's finding arriving from the other side: a Gaussian proposal's
importance ratio stays BOUNDED on a skew-normal target, so skewness alone does
not inflate an outer k-hat -- and #629 measured exactly this, median 0.224 at
true skewness 0.851, at `k_samples = 500`. The two readings were never in
conflict about the target; they were reading different budgets.

The test now asserts the measured property (good band at every budget, spread
below 0.25) instead of a repair that was an artifact of its own budget.

## Scope, and what is NOT claimed

- `tulpa_psis()`'s own default is untouched, so the `loo::psis()` equivalence
  the package maintains as a test oracle is untouched. Only the four outer
  backends resolve a tail size before calling it.
- A DEFAULT fit is bit-for-bit unchanged on every backend: the helper returns
  `NULL` at the reference budget and the explicit-request path, with its 20%
  cap and cap warning, is not entered.
- Which fraction best estimates the tail index is still not measured, and this
  does not answer it. Neither does it answer #631's own open question of which
  of the two readings a reliability verdict should be taken at -- it makes the
  verdict independent of the budget, which is a prerequisite for asking that
  question, not an answer to it.
- One thing it surfaces: with the read budget-stable, the skew-normal rescue is
  not adopted on any fixture in this repo, consistent with #629 measuring
  adoption at 0-1.7% of rows. Whether it is reachable at all is
  gcol33/tulpa#634.

## Files

- `R/psis.R` (`.k_outer_tail_points`), `R/outer_pareto_candidates.R` (resolved
  in `.k_dispatch()` / `.k_dispatch_report()`).
- `k_tail_points` reaches `tulpa_nested_laplace()`, `fit_spde()` and
  `tulpa_re_cov_nested()`, which previously documented it without accepting it.
- Tests: `test-outer-k-budget.R`.
