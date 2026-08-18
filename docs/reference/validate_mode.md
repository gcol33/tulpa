# Validate that a fit used the expected mode

Ensures the fit object was created with the expected inference mode.
Useful for enforcing mode requirements in downstream analysis.

## Usage

``` r
validate_mode(fit, expected_mode, error = TRUE)
```

## Arguments

- fit:

  A tulpa_fit object

- expected_mode:

  Mode that should have been used

- error:

  If TRUE (default), error on mismatch. If FALSE, return logical.

## Value

If error = FALSE, returns TRUE/FALSE. Otherwise errors on mismatch.
