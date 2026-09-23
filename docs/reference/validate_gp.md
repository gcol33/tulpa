# Validate a GP spatial specification against data

Fills in the neighbor structure
([`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md))
or structures
([`spatial_multiscale()`](https://gillescolling.com/tulpa/reference/spatial_multiscale.md))
that
[`tulpa_gibbs()`](https://gillescolling.com/tulpa/reference/tulpa_gibbs.md)
and
[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
need from a `spatial =` spec, since those design-matrix doors take no
`data` argument to validate one themselves.
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) calls
this internally, so it only needs to be called directly when fitting
through a design-matrix door.

## Usage

``` r
validate_gp(gp, data)
```

## Arguments

- gp:

  A
  [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
  or
  [`spatial_multiscale()`](https://gillescolling.com/tulpa/reference/spatial_multiscale.md)
  object.

- data:

  Data frame containing the coordinate columns named in `gp`.

## Value

Updated spatial object with computed neighbor structure.

## See also

[`validate_temporal_multiscale()`](https://gillescolling.com/tulpa/reference/validate_temporal_multiscale.md),
[`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md),
[`spatial_multiscale()`](https://gillescolling.com/tulpa/reference/spatial_multiscale.md)
