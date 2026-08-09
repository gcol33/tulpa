# Test tiers

Tests are organized into three cost tiers. The gate is the first line of a
`test_that()` block (or the top of a file when every block shares the tier).
The single source of truth for the gates is `helper-tiers.R`.

| Tier | Gate | Runs on | What lives here |
|------|------|---------|-----------------|
| 1 structural | *(none)* | everywhere, incl. CRAN | shape / class / dispatch / formula / closed-form / finite-difference-gradient checks; sub-second |
| 2 recovery | `skip_on_cran()` | default local + CI | one (or a few) fits to convergence; parameter recovery vs a single simulated truth; equivalence (dense == sparse, serial == parallel, cpp == R) |
| 3 full | `skip_if_not_slow()` | opt-in only | MCMC / HMC sampler chains, multi-seed coverage aggregates, multi-recovery loops |

## Running

```r
devtools::test()                                  # tiers 1 + 2  (default dev loop)
Sys.setenv(TULPA_FAST = "1"); devtools::test()            # tier 1 only  (fast smoke)
Sys.setenv(TULPA_SLOW_TESTS = "true"); devtools::test()   # tiers 1 + 2 + 3  (full validation)
```

`TULPA_FAST=1` is the fast smoke profile: every tier-2 fit and tier-3 sampler
skips (reported as skips, never silently dropped), leaving only the structural
checks, so the whole suite runs in seconds for plumbing / dispatch iteration.

`R CMD check` / CRAN run **tier 1 only** (`NOT_CRAN` unset trips `skip_on_cran()`,
`TULPA_SLOW_TESTS` unset trips `skip_if_not_slow()`).

Files run in parallel across cores (`Config/testthat/parallel: true`); CRAN caps
this at 2 cores. Keep tests file-independent (use `tempfile()` for any on-disk
artefact) so parallel execution stays correct.

## Adding a test

Pick the tier by cost and put exactly one gate at the top of the block:
a model fit or MCMC sample always carries a gate; a pure structural check never
does (so CRAN still exercises it).

## Posterior arbiters

Three instruments judge a posterior approximation, and they do not collapse into
each other. Use all three before promoting a backend or a numerical rule.

| instrument | where | what it answers |
|---|---|---|
| fixed-truth coverage and width | `recov_sweep()`, `test-nested-laplace-recovery.R` | frequentist performance at one configuration chosen to resemble a real fit |
| SBC / PIT against a simultaneous ECDF band | `recov_sbc()`, `helper-sbc.R`, `test-sbc-crps.R` | is the Bayesian computation self-consistent across the generative distribution -- the whole marginal CDF, not two of its points |
| CRPS, paired seed by seed | `sbc_crps()` / `sbc_crps_compare()`, same files | sharpness subject to calibration, as a strictly proper score |

`helper-sbc.R` is the single implementation of both scorers and of the band; it
depends on base + stats only, so `source()`ing it from a `dev_notes` script
works as well as testthat does. Its section 6 is the engine fixture -- a
gaussian random-intercept design whose exact posterior is available in closed
form, so the engine's read is scored against an independent computation rather
than against another approximation.

Two things about it are easy to get wrong and are held by tests:

* A **pointwise** binomial band is not a **simultaneous** band. At n = 100,
  holding each order statistic at 95% holds all of them together at 0.4471. The
  band here is calibrated against an exact crossing probability and its
  constant-width member reproduces the published Kolmogorov critical value.
* A PIT read off ranks or off a discrete grid has atoms, and reading
  `rank / n_ref` against a continuous uniform is a silent miscalibration. Every
  discrete quantity is randomized within its atom, so one uniform reference and
  one band serve all of them.

CRPS ranks posterior approximations only in a prior-predictive experiment. On a
fixed-truth sweep it is a descriptive loss -- the CRPS-optimal forecast there is
a point mass at the truth -- so `recov_sbc()` records which experiment produced
a result and `sbc_crps_compare()` refuses to rank a fixed-truth one.
`dev_notes/issue335/RESULTS.md` carries the measurement the harness was built
for.
