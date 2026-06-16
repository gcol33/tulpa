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
