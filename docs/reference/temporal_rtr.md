# Restricted temporal regression (RTR)

The temporal analogue of
[`spatial_rsr()`](https://gillescolling.com/tulpa/reference/spatial_rsr.md):
constrain a temporal random effect to be orthogonal to a set of
covariates, so a covariate that moves with the temporal trend does not
share its signal with the temporal field, \\u\_{RTR} = (I - P_X) u\\.

## Usage

``` r
temporal_rtr(temporal, restrict_to)

# S3 method for class 'tulpa_rtr'
print(x, ...)
```

## Arguments

- temporal:

  A single-walk temporal field:
  [`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
  [`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md)
  (cyclic or not) or
  [`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md),
  without a `group_var`. The projection is applied by nested Laplace,
  which reads the field through a projector in place of its time index;
  a grouped (panel) walk,
  [`temporal_multiscale()`](https://gillescolling.com/tulpa/reference/temporal_multiscale.md),
  [`temporal_gp()`](https://gillescolling.com/tulpa/reference/temporal_gp.md)
  and
  [`temporal_tvc()`](https://gillescolling.com/tulpa/reference/temporal_tvc.md)
  are not carried by that path and are refused here rather than accepted
  and then fitted unrestricted.

- restrict_to:

  A one-sided formula giving the covariate space the temporal effect is
  made orthogonal to, e.g. `~ x`.

- x:

  A `tulpa_rtr` specification.

- ...:

  Passed to the print method of the restricted field.

## Value

The temporal specification, flagged as restricted (class `tulpa_rtr` in
front of its own classes). It keeps its type, so it validates against
the data as the field it is.

## Details

The projector is built at the field's own resolution: one row per time
point, with the restricted design averaged over the observations at
each, \\P\_\perp = I - Q Q'\\ with \\Q\\ an orthonormal basis of that
design. Observation \\i\\ at time \\t_i\\ receives \\(P\_\perp
u)\_{t_i}\\, so the field reaches the linear predictor through \\S
P\_\perp\\, \\S\\ the observation-to-time incidence. A design that
carries an intercept annihilates the field's constant, so an RW1 / RW2
walk's level reaches the predictor through the intercept alone.

**What RTR estimates.** As for
[`spatial_rsr()`](https://gillescolling.com/tulpa/reference/spatial_rsr.md),
the restriction puts no part of the shared temporal signal in the field,
so the fixed effect takes all of it: RTR targets the MARGINAL
association between the covariate and the response, where the
unrestricted temporal model targets the association conditional on the
field. The two differ by exactly the covariate's projection onto the
field, so they are different estimands rather than a biased and an
unbiased version of one (Bradley, 2024). Measured on a confounded
Gaussian fixture (5 seeds, 50 time points with 4 observations each,
conditional slope 1.0, marginal slope 1.51 on average): the restricted
fit averaged 0.018 from the marginal value and 0.51 from the conditional
one under each of RW1, RW2 and AR1, the unrestricted fit 0.45 and 0.06
to 0.08.

**The restriction is not free.** For the spatial restriction Hanks et
al. (2015) and Khan and Calder (2022) measured POORER interval coverage
than under the model that does not restrict; that has not been measured
here for a temporal field. Read an RTR interval as an interval for the
marginal association under a correctly specified model.

A non-cyclic RW2 walk leaves its linear trend unpenalised; a
`restrict_to` that spans the linear time trend (e.g. `~ x + year`) would
leave that direction identified by neither the prior nor the data, and
the fit refuses it.

A restricted field is fitted by nested Laplace (`mode = "auto"`,
`"structured"` or `"nested_laplace"`) under any family that path takes,
on its own or alongside an areal or NNGP spatial field, smoothers,
random effects or
[`latent()`](https://gillescolling.com/tulpa/reference/latent.md)
blocks.
[`temporal()`](https://gillescolling.com/tulpa/reference/temporal.md)
reports the restricted field \\P\_\perp u\\, the quantity that enters
the linear predictor.

## References

Hodges, J. S., & Reich, B. J. (2010). Adding spatially-correlated errors
can mess up the fixed effect you love. The American Statistician, 64(4),
325-334.

Hanks, E. M., Schliep, E. M., Hooten, M. B., & Hoeting, J. A. (2015).
Restricted spatial regression in practice: geostatistical models,
confounding, and robustness under model misspecification.
Environmetrics, 26(4), 243-254.

Khan, K., & Calder, C. A. (2022). Restricted spatial regression methods:
implications for inference. Journal of the American Statistical
Association, 117(537), 482-494.

Bradley, J. R. (2024). Restricted spatial regression is reasonable
statistical practice: clarifications, interpretations, and new
developments. arXiv:2408.05106.

## See also

[`spatial_rsr()`](https://gillescolling.com/tulpa/reference/spatial_rsr.md),
[`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
[`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md),
[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)

## Examples

``` r
temporal_rtr(temporal_rw1("year"), restrict_to = ~ x)
#> tulpa temporal specification
#> ============================
#> 
#> Type: RW1 (First-order Random Walk) 
#> Time variable: year 
#> Shared: Yes (enters both processes) 
#> 
#> Restricted Temporal Regression (RTR):
#>   Orthogonal to: ~x 

# \donttest{
set.seed(2)
n_t <- 40
trend <- as.numeric(arima.sim(list(ar = 0.9), n_t, sd = 0.3))
df <- data.frame(year = rep(seq_len(n_t), each = 3))
df$x <- trend[df$year] + rnorm(nrow(df), 0, 0.3)
df$y <- rnorm(nrow(df), 1 + df$x + trend[df$year], 0.3)
fit <- tulpa(y ~ x, data = df, family = "gaussian",
             temporal = temporal_rtr(temporal_ar1("year"), ~ x))
#> Warning: NaNs produced
coef(fit)
#> (Intercept)           x 
#>   0.9717938   1.7874263 
# }
```
