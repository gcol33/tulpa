# Extract temporally-varying coefficients from a fitted model

Extract posterior distributions of temporally-varying coefficients
(TVCs) from a fitted tulpa model with TVC specification.

## Usage

``` r
tvc(object, terms = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...)

# S3 method for class 'tulpa_fit'
tvc(object, terms = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...)

# S3 method for class 'tulpa_tvc_posterior'
print(x, ...)

# S3 method for class 'tulpa_tvc_summary'
print(x, ...)

# S3 method for class 'tulpa_tvc_posterior'
summary(object, probs = c(0.025, 0.5, 0.975), ...)
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

- x:

  A `tulpa_tvc_posterior` or `tulpa_tvc_summary` object.

## Value

A `tulpa_tvc_posterior` object containing:

- `draws`: Array of posterior draws (draws x units x terms). A unit is
  one time point, or on a grouped fit (`temporal_tvc(group_var =)`) one
  (group, time point) pair, group-major.

- `time_levels`: Time point labels

- `groups`: The group of each unit (`NULL` on an ungrouped fit)

- `term_names`: Names of TVC terms

[`summary()`](https://rdrr.io/r/base/summary.html) on a
`tulpa_tvc_posterior` returns a `tulpa_tvc_summary` data frame with one
row per time point and term (per group, time point and term on a grouped
fit, with a `group` column), holding the posterior mean, SD and
requested quantiles of each temporally varying coefficient.

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
#> Warning: tulpa(): the 'hmc' chain has not converged -- x: Rhat 1.07, bulk-ESS 43; (Intercept): Rhat 1.05, bulk-ESS 79. Its estimates and intervals are not yet the posterior. Run longer (control$n_iter / control$warmup); see check_diagnostics().

tvc_post <- tvc(fit)
summary(tvc_post)
#> Temporally varying coefficient posterior summary, 400 posterior draws
#>    time_idx time term        mean        sd        q2.5         q50      q97.5
#> 1         1    1    x -0.54722860 0.2824330 -1.10977185 -0.55503388 0.02178354
#> 2         2    2    x -0.45083386 0.2838631 -1.11247267 -0.44752115 0.09539058
#> 3         3    3    x -0.19708071 0.1631584 -0.54414150 -0.18840216 0.08850889
#> 4         4    4    x -0.10701945 0.2510124 -0.63271347 -0.10155304 0.39963033
#> 5         5    5    x  0.23988419 0.1820944 -0.09387735  0.24098046 0.59838279
#> 6         6    6    x  0.70010911 0.1696473  0.32367347  0.70084133 1.04963581
#> 7         7    7    x  0.40122629 0.2212774 -0.02021544  0.39401375 0.86828022
#> 8         8    8    x  0.23620052 0.2658691 -0.26109811  0.24505041 0.75585052
#> 9         9    9    x -0.07632102 0.2452147 -0.53943781 -0.05775278 0.43777507
#> 10       10   10    x -0.19893647 0.3483548 -0.94827009 -0.16517174 0.38142247
plot(tvc_post, "x")

# }
```
