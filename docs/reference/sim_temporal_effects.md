# Generate temporal random effects for simulation

Generate temporal random effects for simulation

## Usage

``` r
sim_temporal_effects(n_times, sigma = 0.5, type = "rw1", rho = 0.7)
```

## Arguments

- n_times:

  Number of time points

- sigma:

  Standard deviation

- type:

  Temporal type: "rw1", "rw2", "ar1"

- rho:

  Autocorrelation for AR(1). Default 0.7.

## Value

Numeric vector of temporal effects
