# Restricted Spatial Regression (RSR)

Apply Restricted Spatial Regression to mitigate spatial confounding. RSR
orthogonalizes the spatial effect to the covariate space, preventing the
spatial random effect from absorbing covariate information.

This is important when covariates are spatially smooth (e.g., climate
variables, elevation) because the spatial random effect can "steal"
variance from these covariates, leading to biased coefficient estimates.

## Usage

``` r
spatial_rsr(spatial, restrict_to)
```

## Arguments

- spatial:

  An areal specification –
  [`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md),
  `spatial_icar()`,
  [`spatial_bym2()`](https://gillescolling.com/tulpa/reference/spatial_bym2.md)
  or a proper-CAR spec – or an NNGP one,
  [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md).
  The projection is applied by the binomial Polya-Gamma Gibbs sampler,
  which carries the field's own prior precision as an adjacency or as
  Vecchia factors; an HSGP basis (`spatial_gp(approx = "hsgp")`) and an
  SPDE mesh
  ([`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md))
  are neither, and are refused at construction rather than accepted and
  then unfittable.

- restrict_to:

  Formula specifying which covariates to orthogonalize against (e.g.,
  `~ depth + temp`). The spatial effect will be constrained to be
  orthogonal to the column space of these covariates.

## Value

A modified spatial specification with RSR enabled

## Details

The RSR approach (Reich et al., 2006; Hodges & Reich, 2010) modifies the
spatial random effect to be orthogonal to the fixed effect design
matrix:

\$\$w\_{RSR} = (I - P_X) w\$\$

where \\P_X = X(X'X)^{-1}X'\\ is the projection matrix onto the column
space of X. The projector is built at the FIELD's own resolution: one
row per areal unit, or one per unique location for an NNGP field, with
the restricted design averaged over the observations at each.

**What RSR estimates.** The restriction puts no part of the shared,
smooth signal in the field, so the fixed effect takes all of it: RSR
targets the MARGINAL association between the covariate and the response,
where the unrestricted spatial model targets the association conditional
on the field. The two differ by exactly the covariate's projection onto
the field, so they are different estimands rather than a biased and an
unbiased version of one (Bradley, 2024). Measured on a confounded
continuous fixture (5 seeds, conditional slope 1.0, marginal 1.71): the
restricted fit averaged 0.06 from the marginal value and 0.74 from the
conditional one, the unrestricted fit 0.10 and 0.63.

**The restriction is not free.** In the geostatistical setting Hanks et
al. (2015) measured POORER coverage under RSR than under the spatial
model that does not restrict, and credible intervals that can be
inappropriately narrow under model misspecification; Khan and Calder
(2022) report the same on areal structure, with a non-spatial model
competitive on coverage and higher Type-S error rates under RSR. Read an
RSR interval as an interval for the marginal association under a
correctly specified model, and prefer a posterior-predictive check
(Hanks et al., 2015) where that is in doubt.

**When to use RSR:**

- Covariates are spatially smooth (environmental gradients)

- The marginal association is the quantity of interest

- Coefficients appear attenuated toward zero

**When NOT to use RSR:**

- Covariates are spatially uncorrelated

- Spatial effect is the primary quantity of interest

- Prediction is the main goal

- Interval coverage matters more than the point estimate

RSR fits are binomial, through `mode = "gibbs"` (which `mode = "auto"`
selects for it).

## References

Reich, B. J., Hodges, J. S., & Zadnik, V. (2006). Effects of residual
smoothing on the posterior of the fixed effects in disease-mapping
models. Biometrics, 62(4), 1197-1206.

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

[`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md),
[`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md)

## Examples

``` r
# Create RSR spatial structure on an areal field
W <- matrix(0, 4, 4)
for (i in 1:3) W[i, i + 1] <- W[i + 1, i] <- 1
rsr <- spatial_rsr(
  spatial_car(W, level = "obs"),
  restrict_to = ~ depth + temp
)
print(rsr)
#> tulpa spatial specification
#> ===========================
#> 
#> Type: ICAR (Intrinsic CAR) 
#> Level: obs 
#> Spatial units: 4 
#> Shared: Yes (enters both processes) 
#>   (rho fixed at 1, sum-to-zero constraint applied)
#> 
#> Restricted Spatial Regression (RSR):
#>   Orthogonal to: ~depth + temp 
#>   (Spatial effect constrained to be orthogonal to covariate space)

# \donttest{
# Areal binomial data on a chain of regions, covariate spatially confounded
set.seed(404)
n_regions <- 12
W <- matrix(0, n_regions, n_regions)
for (i in 1:(n_regions - 1)) W[i, i + 1] <- W[i + 1, i] <- 1
df <- data.frame(region = factor(rep(1:n_regions, each = 5)))
df$x <- as.integer(df$region) / 4 + rnorm(nrow(df), 0, 0.5)
df$y <- rbinom(nrow(df), 20, plogis(-0.5 + 0.6 * df$x))

# RSR orthogonalises the spatial field to x, protecting its coefficient
fit <- tulpa(
  y ~ x + spatial(region),
  data = df,
  family = "binomial",
  n_trials = rep(20L, nrow(df)),
  spatial = spatial_rsr(spatial_car(W, level = "obs"), restrict_to = ~ x),
  mode = "auto",
  control = list(n_iter = 500L, warmup = 250L)
)
summary(fit)
#>               estimate  std.error      2.5 %      97.5 %
#> (Intercept) -0.3172248 0.11295433 -0.5502363 -0.09914267
#> x            0.5278281 0.06516213  0.4022617  0.65216922

# The same modifier on a continuous field: one observation per location,
# the projector built at the unique coordinates.
set.seed(7)
n <- 80
pts <- data.frame(lon = runif(n), lat = runif(n))
pts$x <- as.numeric(scale(pts$lon + pts$lat + rnorm(n, 0, 0.3)))
pts$y <- rbinom(n, 25, plogis(-0.2 + pts$x))
fit_gp <- tulpa(
  y ~ x,
  data = pts,
  family = "binomial",
  n_trials = rep(25L, n),
  spatial = spatial_rsr(spatial_gp(~ lon + lat), restrict_to = ~ x),
  mode = "gibbs",
  control = list(n_iter = 500L, warmup = 250L)
)
summary(fit_gp)
#>               estimate  std.error      2.5 %      97.5 %
#> (Intercept) -0.1682499 0.05011257 -0.2674313 -0.07419048
#> x            0.9771012 0.05916971  0.8525881  1.08864375
# }
```
