# Restricted temporal regression (RTR)

The temporal analogue of
[`spatial_rsr()`](https://gillescolling.com/tulpa/reference/spatial_rsr.md):
constrain a temporal random effect to be orthogonal to a set of
covariates, so a temporally smooth covariate does not have its
fixed-effect coefficient attenuated by a confounded temporal field,
\\u\_{RTR} = (I - P_X) u\\.

No tulpa backend applies that projection to a temporal field, so this
constructor errors rather than returning a specification that would fit
as the unrestricted temporal model.
[`spatial_rsr()`](https://gillescolling.com/tulpa/reference/spatial_rsr.md)
is the wired spatial analogue.

## Usage

``` r
temporal_rtr(temporal, restrict_to)
```

## Arguments

- temporal:

  A `tulpa_temporal` specification (e.g.
  [`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
  [`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)).

- restrict_to:

  A one-sided formula giving the covariate space the temporal effect is
  made orthogonal to, e.g. `~ x`.

## Value

Nothing: the call always signals an error.

## See also

[`spatial_rsr()`](https://gillescolling.com/tulpa/reference/spatial_rsr.md),
[`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)
