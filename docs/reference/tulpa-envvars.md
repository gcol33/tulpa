# Environment variables tulpa reads

Switches read from the process environment rather than from a fitter's
`control` list, because each one changes how the compiled kernels
partition work rather than what model is fitted. Each is read once, when
the package's shared library loads, so it has to be set before
[`library(tulpa)`](https://github.com/gcol33/tulpa) – setting it later
in a session has no effect.

## Reproducibility of a parallel scatter

- `TULPA_GRID_WORKSTEAL=0`:

  Forces the serial per-cell coupling scatter. The work-stealing
  partition is pinned by the grid geometry, so a grid solved with
  several outer threads already reproduces the serial reduction; this is
  the escape hatch if that ever has to be verified against a plain
  sequential pass.

- `TULPA_COUPLING_FORCE_PARALLEL`:

  Set (to any value) to take the chunked parallel reduce on every
  coupled cell instead of only where the cell count pays for the
  per-chunk partial gradient and Hessian buffers. A small grid then
  exercises the parallel path. The reduce runs in fixed chunk order, so
  the answer is identical either way, which is what
  `tests/testthat/test-coupling-force-parallel.R` asserts.

## Test-suite tiers

`TULPA_FAST`, `TULPA_SLOW_TESTS` and `TULPA_FULL_RECOVERY` select which
cost tier of the test suite runs; they are read by the suite, not by the
package, and are described in `tests/testthat/README.md`.
