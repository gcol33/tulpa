# Reject non-finite fitting inputs.

The design is built with `na.action = na.pass`, so a missing predictor
or response survives into `X` / `y`. No fitter drops incomplete cases,
so an NA/NaN/Inf would propagate into the C++ kernels as a NaN estimate.
Fail loudly with the offending row instead.

## Usage

``` r
.assert_finite_model_inputs(X, y, n_trials = NULL, offset = NULL, where = NULL)
```

## Arguments

- X, y:

  Design matrix and response; either may be `NULL` to skip that arm.

- n_trials, offset:

  Per-observation binomial denominators and offset, or `NULL` to skip
  that arm.

- where:

  Caller name for the message prefix, or `NULL` for none.

## Details

Reached from
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md), from
the direct beta-only doors, and from
[`.validate_glm_design()`](https://gillescolling.com/tulpa/reference/dot-validate_glm_design.md),
which is what carries it to every fitter taking a `(y, X, n_trials)`
bundle.

Every per-observation numeric the kernels index is checked here, not
only the design and the response: `n_trials` and the offset enter eta
and the likelihood the same way, and an NA in either came back as a
silent all-zero coefficient vector on the Laplace door
(gcol33/tulpa#665).
