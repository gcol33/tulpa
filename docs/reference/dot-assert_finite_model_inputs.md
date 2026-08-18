# Reject non-finite fitting inputs.

The design is built with `na.action = na.pass`, so a missing predictor
or response survives into `X` / `y`. tulpa() does not drop incomplete
cases, so an NA/NaN/Inf would propagate into the C++ kernels as a NaN
estimate. Fail loudly with the offending row instead.

## Usage

``` r
.assert_finite_model_inputs(X, y)
```
