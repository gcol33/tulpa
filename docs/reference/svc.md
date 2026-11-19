# Extract spatially-varying coefficients from a fitted model

Extract posterior distributions of spatially-varying coefficients (SVCs)
from a fitted tulpa model with SVC specification.

## Usage

``` r
svc(object, terms = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...)

# S3 method for class 'tulpa_fit'
svc(object, terms = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...)

# S3 method for class 'tulpa_svc_posterior'
summary(object, probs = c(0.025, 0.5, 0.975), ...)

# S3 method for class 'tulpa_svc_posterior'
print(x, ...)

# S3 method for class 'tulpa_svc_summary'
print(x, ...)
```

## Arguments

- object:

  A `tulpa_fit` object fitted with `svc` argument

- terms:

  Which SVC terms to extract. If NULL (default), extracts all.

- summary:

  Logical; if TRUE, return summary statistics instead of full posterior
  draws.

- probs:

  Quantiles to compute if `summary = TRUE`.

- ...:

  Ignored

- x:

  A `tulpa_svc_posterior` or `tulpa_svc_summary` object.

## Value

A `tulpa_svc_posterior` object containing:

- `draws`: Array of posterior draws (draws x locations x terms)

- `coords`: Coordinate matrix

- `term_names`: Names of SVC terms

[`summary()`](https://rdrr.io/r/base/summary.html) on a
`tulpa_svc_posterior` returns a `tulpa_svc_summary` data frame with one
row per location and term, holding the observation index, term name,
coordinates, and the posterior mean, SD and requested quantiles of each
spatially varying coefficient.

## See also

[`spatial_svc()`](https://gillescolling.com/tulpa/reference/spatial_svc.md),
[`plot.tulpa_svc_posterior()`](https://gillescolling.com/tulpa/reference/plot.tulpa_svc_posterior.md)

## Examples

``` r
# \donttest{
set.seed(303)
n <- 25L
df <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
bsurf <- 0.9 * sin(2.8 * df$lon) + 0.7 * cos(2.2 * df$lat)
df$count <- rpois(n, exp(0.2 + (0.8 + bsurf) * df$x))

# The varying slope on `x` is a spatial field; SVC is exact-mode only.
fit <- tulpa(
  count ~ x,
  data = df,
  family = "poisson",
  spatial = spatial_svc(~ lon + lat, terms = ~ x - 1, nn = 5L),
  mode = "exact",
  control = list(n_iter = 80L, n_warmup = 40L, seed = 1L)
)
#> Warning: tulpa(): the 'hmc' chain has not converged -- x: Rhat 1.03, bulk-ESS 77; (Intercept): Rhat 1.01, bulk-ESS 94. Its estimates and intervals are not yet the posterior. Run longer (control$n_iter / control$warmup); see check_diagnostics().

svc_post <- svc(fit)
summary(svc_post)
#> Spatially varying coefficient posterior summary, 160 posterior draws
#>    obs term    coord_1    coord_2        mean        sd        q2.5         q50
#> 1    1    x 0.30961600 0.61092037 -0.14973993 0.2807046 -0.72729412 -0.18046485
#> 2    2    x 0.83457367 0.80210956 -0.01033393 0.3589492 -0.68981585 -0.04210483
#> 3    3    x 0.60981555 0.11600718  0.22048219 0.3214810 -0.45335313  0.21608289
#> 4    4    x 0.73997696 0.23340995  0.35444704 0.1269614  0.04847449  0.36339532
#> 5    5    x 0.64968413 0.36752677  0.21924627 0.2888125 -0.38864715  0.22977445
#> 6    6    x 0.98059526 0.16249526  0.17379453 0.3698403 -0.58705435  0.15744675
#> 7    7    x 0.52733891 0.04603014  0.11379287 0.3401746 -0.61081178  0.09892752
#> 8    8    x 0.46789913 0.78060351 -0.25384299 0.3789217 -0.91418353 -0.26745539
#> 9    9    x 0.28156115 0.27811056 -0.20459695 0.3368724 -0.86460882 -0.17676584
#> 10  10    x 0.27876133 0.31064937 -0.18913550 0.3334519 -0.91068927 -0.15637536
#> 11  11    x 0.90555893 0.27413493  0.26627186 0.2979694 -0.21566533  0.25323764
#> 12  12    x 0.87261579 0.32569479  0.28807511 0.2490759 -0.15543508  0.26143868
#> 13  13    x 0.10369257 0.46935352 -0.32243314 0.3400281 -1.07178786 -0.28810869
#> 14  14    x 0.03638615 0.71076074 -0.34978458 0.3246479 -1.03896830 -0.33091020
#> 15  15    x 0.23365584 0.82966674 -0.37978314 0.2831971 -0.92687185 -0.38305362
#> 16  16    x 0.21302249 0.71354593 -0.39897435 0.1820162 -0.83842960 -0.37404105
#> 17  17    x 0.40852481 0.34971824  0.16956815 0.3436681 -0.42215873  0.12682346
#> 18  18    x 0.23366216 0.63985821 -0.23319287 0.2477869 -0.71868687 -0.24997577
#> 19  19    x 0.72840439 0.51725032  0.16945757 0.3870604 -0.41004823  0.12958815
#> 20  20    x 0.65473955 0.16251311  0.23052042 0.3567541 -0.76511340  0.25733293
#> 21  21    x 0.84652193 0.77875231 -0.01810220 0.3720595 -0.63912627 -0.04594777
#> 22  22    x 0.65537863 0.24884834  0.30511948 0.2129838 -0.16679046  0.31736682
#> 23  23    x 0.82734509 0.18598061  0.26432357 0.3166972 -0.27014071  0.26386147
#> 24  24    x 0.24047672 0.78207493 -0.47698610 0.1630740 -0.85310506 -0.47053433
#> 25  25    x 0.82767551 0.38519677  0.21180661 0.3151492 -0.34480444  0.20182822
#>         q97.5
#> 1   0.4644430
#> 2   0.7712883
#> 3   0.8911190
#> 4   0.5737905
#> 5   0.7597764
#> 6   1.0039009
#> 7   0.7978157
#> 8   0.5365832
#> 9   0.3633356
#> 10  0.3423264
#> 11  0.8079401
#> 12  0.7519173
#> 13  0.2510709
#> 14  0.2644508
#> 15  0.2703946
#> 16 -0.0857971
#> 17  0.8768021
#> 18  0.2328369
#> 19  1.0151024
#> 20  0.8032101
#> 21  0.7793628
#> 22  0.6841474
#> 23  0.8587813
#> 24 -0.1913242
#> 25  0.8196588
# }
```
