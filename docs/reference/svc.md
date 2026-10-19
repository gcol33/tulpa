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
#> Warning: tulpa(): the 'hmc' chain has not converged -- x: Rhat 1.08, bulk-ESS 75; (Intercept): Rhat 1.06, bulk-ESS 83. Its estimates and intervals are not yet the posterior. Run longer (control$n_iter / control$warmup), or use mode = 'hmc'; see check_diagnostics().

svc_post <- svc(fit)
summary(svc_post)
#> Spatially varying coefficient posterior summary, 160 posterior draws
#>    obs term    coord_1    coord_2        mean        sd        q2.5         q50
#> 1    1    x 0.30961600 0.61092037 -0.29955276 0.2754405 -0.83258898 -0.28358041
#> 2    2    x 0.83457367 0.80210956  0.07684232 0.5137690 -0.80691643  0.02130666
#> 3    3    x 0.60981555 0.11600718  0.31467963 0.3146108 -0.22604839  0.32560296
#> 4    4    x 0.73997696 0.23340995  0.33696713 0.1564360 -0.05245735  0.33990260
#> 5    5    x 0.64968413 0.36752677  0.27862680 0.2447322 -0.20136058  0.27398674
#> 6    6    x 0.98059526 0.16249526  0.22643969 0.4004588 -0.60962565  0.21284090
#> 7    7    x 0.52733891 0.04603014  0.23684137 0.3922178 -0.53231554  0.24342274
#> 8    8    x 0.46789913 0.78060351 -0.27176240 0.4035566 -0.87923196 -0.30979724
#> 9    9    x 0.28156115 0.27811056 -0.26053960 0.3257951 -0.90718482 -0.25519256
#> 10  10    x 0.27876133 0.31064937 -0.22190751 0.2799448 -0.83635274 -0.21140732
#> 11  11    x 0.90555893 0.27413493  0.31147715 0.2745749 -0.15269345  0.28268969
#> 12  12    x 0.87261579 0.32569479  0.34187649 0.2000977 -0.04559340  0.34979228
#> 13  13    x 0.10369257 0.46935352 -0.43397330 0.3166352 -0.95220572 -0.44007890
#> 14  14    x 0.03638615 0.71076074 -0.52574158 0.3907374 -1.21073090 -0.49549134
#> 15  15    x 0.23365584 0.82966674 -0.51370739 0.2588695 -1.13598678 -0.49314375
#> 16  16    x 0.21302249 0.71354593 -0.43514052 0.2000422 -0.89112415 -0.41569495
#> 17  17    x 0.40852481 0.34971824  0.07108789 0.2883693 -0.43770592  0.03688384
#> 18  18    x 0.23366216 0.63985821 -0.33225264 0.1944496 -0.65892967 -0.33687258
#> 19  19    x 0.72840439 0.51725032  0.22163497 0.3582945 -0.41519648  0.20622132
#> 20  20    x 0.65473955 0.16251311  0.29457893 0.2782473 -0.20617223  0.30690111
#> 21  21    x 0.84652193 0.77875231  0.09886014 0.5253306 -0.83512163  0.03070179
#> 22  22    x 0.65537863 0.24884834  0.29802986 0.2299392 -0.11763532  0.30136161
#> 23  23    x 0.82734509 0.18598061  0.35009241 0.3242799 -0.28753375  0.36869740
#> 24  24    x 0.24047672 0.78207493 -0.50923955 0.1573022 -0.80090618 -0.49704394
#> 25  25    x 0.82767551 0.38519677  0.34578249 0.3450844 -0.15577561  0.30474151
#>           q97.5
#> 1   0.260490230
#> 2   1.507156430
#> 3   0.965364913
#> 4   0.595960825
#> 5   0.741558173
#> 6   1.015070588
#> 7   0.886641767
#> 8   0.474778654
#> 9   0.294706611
#> 10  0.261061638
#> 11  0.840042438
#> 12  0.694373585
#> 13  0.162077862
#> 14  0.091937749
#> 15  0.013569761
#> 16 -0.004261005
#> 17  0.687604541
#> 18  0.015105274
#> 19  0.997678626
#> 20  0.886060150
#> 21  1.505796594
#> 22  0.765099534
#> 23  0.983745477
#> 24 -0.254437878
#> 25  1.248545267
# }
```
