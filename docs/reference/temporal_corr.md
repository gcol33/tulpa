# Extract temporal correlation parameters from a fitted model

Returns posterior summary for temporal hyperparameters (tau, rho for
AR1, sigma/lengthscale for temporal GP).

## Usage

``` r
temporal_corr(object, probs = c(0.025, 0.975))
```

## Arguments

- object:

  A `tulpa_fit` object fitted with a temporal component.

- probs:

  Quantile probabilities (default 0.025, 0.975).

## Value

A data.frame with rows for each temporal hyperparameter.
