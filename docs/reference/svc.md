# Extract spatially-varying coefficients from a fitted model

Extract posterior distributions of spatially-varying coefficients (SVCs)
from a fitted tulpa model with SVC specification.

## Usage

``` r
svc(object, terms = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...)

# S3 method for class 'tulpa_fit'
svc(object, terms = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...)
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

## Value

A `tulpa_svc_posterior` object containing:

- `draws`: Array of posterior draws (draws x locations x terms)

- `coords`: Coordinate matrix

- `term_names`: Names of SVC terms

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

svc_post <- svc(fit)
summary(svc_post)
#>    obs term     coord_1    coord_2        mean        sd         q2.5
#> 1    1    x -0.80526094  0.6679185 -0.28394241 0.3048324 -1.002760314
#> 2    2    x  1.04004117  1.4295180  0.03817524 0.4517597 -0.721449585
#> 3    3    x  0.24998389 -1.3035611  0.28187967 0.2746177 -0.153336040
#> 4    4    x  0.70752003 -0.8358888  0.40763906 0.1540074  0.063806534
#> 5    5    x  0.39012772 -0.3016364  0.22867897 0.2812797 -0.384554448
#> 6    6    x  1.55332816 -1.1183765  0.23477673 0.3636138 -0.703134761
#> 7    7    x -0.03993344 -1.5823136  0.20569704 0.3631128 -0.513660458
#> 8    8    x -0.24887286  1.3438489 -0.28506146 0.3996089 -1.050949356
#> 9    9    x -0.90387778 -0.6578246 -0.19275763 0.3667506 -1.013321060
#> 10  10    x -0.91371954 -0.5282067 -0.21652893 0.3572421 -1.009928136
#> 11  11    x  1.28956460 -0.6736614  0.29356757 0.2464367 -0.191978887
#> 12  12    x  1.17376469 -0.4682735  0.31208753 0.1857748 -0.026228217
#> 13  13    x -1.52911155  0.1039890 -0.41246621 0.3650218 -1.313778702
#> 14  14    x -1.76570334  1.0656313 -0.38742024 0.2743868 -0.901818385
#> 15  15    x -1.07227189  1.5392916 -0.43166318 0.2613804 -0.940744144
#> 16  16    x -1.14480110  1.0767260 -0.40695538 0.1514109 -0.773519518
#> 17  17    x -0.45758217 -0.3725764  0.04536641 0.2829039 -0.551820940
#> 18  18    x -1.07224966  0.7831920 -0.30133216 0.2383761 -0.815364723
#> 19  19    x  0.66684078  0.2947853  0.09306988 0.2853509 -0.393837394
#> 20  20    x  0.40789824 -1.1183054  0.31254287 0.2283443 -0.098609129
#> 21  21    x  1.08204104  1.3364747  0.03543261 0.4134322 -0.649543711
#> 22  22    x  0.41014468 -0.7743902  0.33231079 0.1967510 -0.004105197
#> 23  23    x  1.01463165 -1.0248229  0.31069263 0.2643876 -0.266652272
#> 24  24    x -1.04829551  1.3497103 -0.45373830 0.1654450 -0.840701176
#> 25  25    x  1.01579314 -0.2312482  0.23994890 0.2761857 -0.288231924
#>             q50       q97.5
#> 1  -0.287009756  0.24334403
#> 2   0.001249469  1.09715159
#> 3   0.247678042  0.94798075
#> 4   0.395102399  0.70890551
#> 5   0.225442188  0.75026363
#> 6   0.284866347  0.86861424
#> 7   0.191283677  0.92125387
#> 8  -0.263621874  0.91776506
#> 9  -0.183056280  0.43727980
#> 10 -0.195542729  0.42447108
#> 11  0.270983858  0.81211540
#> 12  0.316657786  0.65875422
#> 13 -0.358349420  0.11724582
#> 14 -0.381158552  0.08446740
#> 15 -0.413950391  0.06347097
#> 16 -0.387866349 -0.15610266
#> 17  0.067229526  0.61338989
#> 18 -0.297501077  0.11992256
#> 19  0.105441016  0.65003307
#> 20  0.302504108  0.82691563
#> 21  0.002508192  0.91244169
#> 22  0.315304112  0.72757542
#> 23  0.322431730  0.71798240
#> 24 -0.434073769 -0.18072004
#> 25  0.239676808  0.69437561
# }
```
