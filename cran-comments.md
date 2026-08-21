# CRAN comments

## R CMD check results

0 errors | 0 warnings | 2 notes

* This is a new submission.

* The remaining note is the Windows-only assembler flag described below.

## Test environments

* local: Windows 11, R 4.6.0
* win-builder: R-devel, R-release
* GitHub Actions, on every push: ubuntu-latest (R-release and R-devel),
  macos-latest (R-release), windows-latest (R-release), all compiled with
  -ffp-contract=off
* Linux: Rocky Linux 9.8, R 4.5.2, gcc 14.3, built at the x86-64 baseline
  with -ffp-contract=off

## Notes

* The package contains a large compiled codebase (C++ inference kernels);
  installed size may exceed the default threshold on some platforms.

* On Windows the build requires the assembler flag -Wa,-mbig-obj
  (Makevars.win): several templated inference kernels exceed the default
  COFF section limit. The flag is Windows-only and flagged as non-portable
  by R CMD check; without it the package does not compile with Rtools.

* Long-running model fits in examples are wrapped in \donttest{}; each
  retains a small runnable form where feasible. Recovery and sampler tests
  are gated by testthat's skip_on_cran() and by the package's own tier
  variables (NOT_CRAN, TULPA_SLOW_TESTS). A named subset of the recovery
  tier runs on every push in GitHub Actions, on Linux and on macOS, and a
  named subset of the sampler and coverage tier runs there on a schedule.
  Both tiers are run in full by the maintainer before each release.

* A small number of examples remain in \dontrun{}. Each either references
  symbols the user supplies (a per-model E-step / M-step callback pair, a
  compiled latent block, a mesh-backed SPDE spec) and so has nothing to
  execute, or requires an observation likelihood from a companion model
  package that is not on CRAN (tulpaRatio).

* The vignettes fit models, which does not fit inside the ten-minute check
  budget. Each vignette sets `eval` from NOT_CRAN in its setup chunk, so the
  code is shown but not run on the check farm. It is evaluated in full
  locally and on the package website.

* Intra-chain OpenMP teams are resolved in one place
  (`tulpa_omp_team_size()`, src/omp_threads.h), which reads
  `_R_CHECK_LIMIT_CORES_` and caps the team at two threads under R CMD check,
  in addition to honouring OMP_NUM_THREADS and OMP_THREAD_LIMIT.

* Two internal helpers seed the RNG: `.spde_mean_marginal_var()` uses a fixed
  probe matrix so the field normalization is deterministic and smooth across
  the hyperparameter grid, and the SBC driver seeds each replicate. Both run
  inside `.with_preserved_seed()`, which saves `.Random.seed` and restores the
  caller's RNG state on exit, so no user-visible RNG state is changed.

## Downstream dependencies

No reverse dependencies on CRAN. The in-development packages tulpaObs and
tulpaRatio (GitHub) track the engine and are checked against each release.
