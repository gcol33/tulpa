# Extract temporally-varying coefficients from a fitted model

Extract posterior distributions of temporally-varying coefficients
(TVCs) from a fitted tulpa model with TVC specification.

## Usage

``` r
tvc(object, terms = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...)

# S3 method for class 'tulpa_fit'
tvc(object, terms = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...)
```

## Arguments

- object:

  A `tulpa_fit` object fitted with `tvc` argument

- terms:

  Which TVC terms to extract. If NULL (default), extracts all.

- summary:

  Logical; if TRUE, return summary statistics instead of full posterior
  draws.

- probs:

  Quantiles to compute if `summary = TRUE`.

- ...:

  Ignored

## Value

A `tulpa_tvc_posterior` object containing:

- `draws`: Array of posterior draws (draws x times x terms)

- `time_levels`: Time point labels

- `term_names`: Names of TVC terms

## See also

[`temporal_tvc()`](https://gillescolling.com/tulpa/reference/temporal_tvc.md),
[`plot.tulpa_tvc_posterior()`](https://gillescolling.com/tulpa/reference/plot.tulpa_tvc_posterior.md)

## Examples

``` r
# \donttest{
set.seed(160)
n_t <- 10L; reps <- 5L
walk <- cumsum(rnorm(n_t, 0, 0.35)); walk <- walk - mean(walk)
year <- rep(seq_len(n_t), each = reps)
df <- data.frame(year = year, x = rnorm(length(year)))
df$count <- rpois(nrow(df), exp(0.3 + (0.5 + walk[year]) * df$x))

# The slope on `x` walks in time; TVC is exact-mode only.
fit <- tulpa(
  count ~ x,
  data = df,
  family = "poisson",
  temporal = temporal_tvc("year", terms = ~ x - 1, structure = "rw1"),
  mode = "exact",
  control = list(n_iter = 200L, n_warmup = 100L, seed = 1L)
)

tvc_post <- tvc(fit)
summary(tvc_post)
#>    time_idx time term        mean        sd          q2.5         q50
#> 1         1    1    x -0.46767697 0.2727967 -1.0488167585 -0.46027876
#> 2         2    2    x -0.42963412 0.2952008 -1.1740689169 -0.41494907
#> 3         3    3    x -0.17554472 0.1558346 -0.4505894021 -0.19949291
#> 4         4    4    x -0.08200355 0.2427412 -0.5312185654 -0.08422902
#> 5         5    5    x  0.26052798 0.2091863 -0.1846805834  0.25763408
#> 6         6    6    x  0.72405049 0.1873680  0.3827999789  0.70784629
#> 7         7    7    x  0.41721585 0.2596555  0.0002152879  0.39862375
#> 8         8    8    x  0.19159886 0.2552583 -0.2832599527  0.17544803
#> 9         9    9    x -0.14741052 0.2886050 -0.7479633605 -0.12615575
#> 10       10   10    x -0.29134514 0.3537593 -0.9920605968 -0.26005880
#>         q97.5
#> 1  0.02412607
#> 2  0.05863057
#> 3  0.16806367
#> 4  0.38931157
#> 5  0.70117992
#> 6  1.09390869
#> 7  1.04389485
#> 8  0.74931456
#> 9  0.37872456
#> 10 0.29510485
plot(tvc_post, "x")

# }
```
