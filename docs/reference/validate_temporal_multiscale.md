# Validate a multi-scale temporal specification against data

Fills in the time index and point count that
[`tulpa_gibbs()`](https://gillescolling.com/tulpa/reference/tulpa_gibbs.md)
needs from a `temporal =` spec built with
[`temporal_multiscale()`](https://gillescolling.com/tulpa/reference/temporal_multiscale.md),
since that design-matrix door takes no `data` argument to validate one
itself. [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)
calls this internally, so it only needs to be called directly when
fitting through a design-matrix door.

## Usage

``` r
validate_temporal_multiscale(temporal, data)
```

## Arguments

- temporal:

  A
  [`temporal_multiscale()`](https://gillescolling.com/tulpa/reference/temporal_multiscale.md)
  object (or a single-component temporal spec, dispatched to
  [`validate_temporal()`](https://gillescolling.com/tulpa/reference/validate_temporal.md)).

- data:

  Data frame containing the time column named in `temporal`.

## Value

Updated object with indices computed.

## See also

[`validate_gp()`](https://gillescolling.com/tulpa/reference/validate_gp.md),
[`temporal_multiscale()`](https://gillescolling.com/tulpa/reference/temporal_multiscale.md)
