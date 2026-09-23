# SPDE field hyperprior log density on (range, sigma)

The SPDE field's hyperprior on `(range, sigma)`, evaluated as a density
on `(log range, log sigma)` – the coordinates every SPDE outer
integrator works in (grid, CCD and k-hat alike, gcol33/tulpa#731).
Intended for a caller running its own outer-grid loop over
`(range, sigma)` around a coupled model tulpa's own SPDE front doors do
not fit directly.

## Usage

``` r
tulpa_spde_log_hyperprior(range, sigma, sp, hyperprior = "proper")
```

## Arguments

- range, sigma:

  Numeric vectors of range / marginal-SD nodes (natural scale, recycled
  against each other).

- sp:

  A
  [`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md)
  /
  [`spatial_spde_custom()`](https://gillescolling.com/tulpa/reference/spatial_spde_custom.md)
  spec, carrying the declared or defaulted `prior_range` / `prior_sigma`
  anchors.

- hyperprior:

  `"proper"` (default) or `"flat"`; see
  [`?spatial_spde`](https://gillescolling.com/tulpa/reference/spatial_spde.md).

## Value

Numeric vector of log-density values, one per `(range, sigma)` pair.

## See also

[`tulpa_spde_precision_Q()`](https://gillescolling.com/tulpa/reference/tulpa_spde_precision_Q.md),
[`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md)
