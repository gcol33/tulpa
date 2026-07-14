# CRAN comments

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## Test environments

* local: Windows 11, R 4.5.x
* win-builder: R-devel, R-release
* mac-builder: R-release

## Notes

* The package contains a large compiled codebase (C++ inference kernels);
  installed size may exceed the default threshold on some platforms.
* On Windows the build requires the assembler flag -Wa,-mbig-obj
  (Makevars.win): several templated inference kernels exceed the default
  COFF section limit. The flag is Windows-only and flagged as non-portable
  by R CMD check; without it the package does not compile with Rtools.
* Long-running model fits in examples are wrapped in \donttest{}; each
  retains a small runnable form where feasible. Recovery and sampler tests
  are skipped on CRAN via testthat gating and run in CI instead.
* A small number of examples remain in \dontrun{}: they exercise
  experimental model classes (multiscale spatial, SVC, non-separable
  spatiotemporal, GP-temporal, TVC/RTR decompositions) whose fitting paths
  are exported for the companion model packages but not yet supported
  end-to-end through the tulpa() front door, so the examples cannot run.

## Downstream dependencies

No reverse dependencies on CRAN. The in-development packages tulpaObs,
tulpaRatio, and tulpaGlmm (GitHub) track the engine and are checked against
each release.
