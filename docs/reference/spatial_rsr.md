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

  A spatial specification (`spatial_gp`, `spatial_car`, etc.)

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
space of X.

**When to use RSR:**

- Covariates are spatially smooth (environmental gradients)

- Interested in causal interpretation of covariate effects

- Coefficients appear attenuated toward zero

**When NOT to use RSR:**

- Covariates are spatially uncorrelated

- Spatial effect is the primary quantity of interest

- Prediction is the main goal (not causal inference)

## References

Reich, B. J., Hodges, J. S., & Zadnik, V. (2006). Effects of residual
smoothing on the posterior of the fixed effects in disease-mapping
models. Biometrics, 62(4), 1197-1206.

Hodges, J. S., & Reich, B. J. (2010). Adding spatially-correlated errors
can mess up the fixed effect you love. The American Statistician, 64(4),
325-334.

## See also

[`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md),
[`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md)

## Examples

``` r
# Create RSR spatial structure
rsr <- spatial_rsr(
  spatial_gp(~ lon + lat),
  restrict_to = ~ depth + temp
)
print(rsr)

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
# }
```
