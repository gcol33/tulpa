# Extract spatial range and variance from a fitted spatial model

Summarises the posterior of spatial hyperparameters. For sampler-tier
fits this reads the raw hyperparameter draws; for a nested-Laplace
spatial fit it summarises the outer hyperparameter grid. Works with
ICAR, BYM2, GP (NNGP), CAR, SPDE, and SVC spatial types.

## Usage

``` r
spatial_range(object, probs = c(0.025, 0.975))
```

## Arguments

- object:

  A `tulpa_fit` object fitted with a spatial component.

- probs:

  Quantile probabilities for the summary (default 0.025, 0.975).

## Value

A data.frame with rows for each spatial hyperparameter and columns
`mean`, `sd`, and one quantile column per entry of `probs` (named from
the probability, e.g. `q2.5`, `q97.5` for the defaults).
