# Restricted temporal regression (RTR)

The temporal analogue of
[`spatial_rsr()`](https://gillescolling.com/tulpa/reference/spatial_rsr.md):
constrain a temporal random effect to be orthogonal to a set of
covariates, so a temporally smooth covariate does not have its
fixed-effect coefficient attenuated by a confounded temporal field.

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

The temporal specification with RTR enabled (class `tulpa_rtr`
prepended), for use in the `temporal =` argument of
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md).

## Details

RTR modifies the temporal random effect to be orthogonal to the
fixed-effect design: \\u\_{RTR} = (I - P_X) u\\ where \\P_X =
X(X'X)^{-1}X'\\ projects onto the column space of the restricted
covariates. Use it when a covariate is temporally smooth and its
coefficient appears attenuated toward zero; avoid it when the temporal
effect is itself the quantity of interest.

## See also

[`spatial_rsr()`](https://gillescolling.com/tulpa/reference/spatial_rsr.md),
[`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)

## Examples

``` r
rtr <- temporal_rtr(temporal_rw1("year"), restrict_to = ~ x)
print(rtr)
#> tulpa temporal specification
#> ============================
#> 
#> Type: RW1 (First-order Random Walk) 
#> Time variable: year 
#> Shared: Yes (enters both processes) 
#> 
#> Restricted Temporal Regression (RTR):
#>   Orthogonal to: ~x 
#>   (Temporal effect constrained to be orthogonal to covariate space)
```
