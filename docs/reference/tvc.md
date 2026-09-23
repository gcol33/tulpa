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
#>    time_idx time term        mean        sd        q2.5         q50     q97.5
#> 1         1    1    x -0.51812356 0.3438930 -1.26815867 -0.48358081 0.1138761
#> 2         2    2    x -0.43151069 0.3186817 -1.08518130 -0.39639008 0.1291343
#> 3         3    3    x -0.17678201 0.1581185 -0.48672751 -0.17809841 0.1306088
#> 4         4    4    x -0.04903702 0.2429977 -0.52076934 -0.06248125 0.4830875
#> 5         5    5    x  0.25799312 0.2008314 -0.17772448  0.26136132 0.6231688
#> 6         6    6    x  0.71976261 0.1672618  0.39749237  0.70680250 1.0518190
#> 7         7    7    x  0.35624069 0.2307417 -0.08571445  0.36628699 0.8236376
#> 8         8    8    x  0.17305269 0.2655832 -0.29044750  0.16995112 0.7065457
#> 9         9    9    x -0.13007529 0.2739828 -0.61089948 -0.13256620 0.3671592
#> 10       10   10    x -0.20152056 0.3324161 -0.84936798 -0.20867196 0.4724814
plot(tvc_post, "x")

# }
```
