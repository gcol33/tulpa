# Extract spatiotemporal effects from fitted model

Extract posterior distributions of spatiotemporal interaction effects
from a fitted tulpa model.

## Usage

``` r
spatiotemporal_effects(
  object,
  format = c("array", "long", "summary"),
  probs = c(0.025, 0.5, 0.975),
  ...
)

# Default S3 method
spatiotemporal_effects(
  object,
  format = c("array", "long", "summary"),
  probs = c(0.025, 0.5, 0.975),
  ...
)

# S3 method for class 'tulpa_fit'
spatiotemporal_effects(
  object,
  format = c("array", "long", "summary"),
  probs = c(0.025, 0.5, 0.975),
  ndraws = 1000L,
  seed = NULL,
  ...
)

# S3 method for class 'tulpa_st_summary'
print(x, ...)
```

## Arguments

- object:

  A nested-Laplace fit of
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)
  carrying a
  [`spatiotemporal()`](https://gillescolling.com/tulpa/reference/spatiotemporal.md)
  interaction, or any fit storing a `$spatiotemporal` Knorr-Held block
  (`n_spatial`, `n_times`) with its draws in
  `$.internal$spatiotemporal_draws` (a model package's own fitter),
  which the default method reads.

- format:

  Output format: `"array"` (default, S x T x draws), `"long"` (data
  frame with s, t, draw, value columns), or `"summary"` (posterior
  summaries).

- probs:

  Quantiles to compute if `format = "summary"`.

- ...:

  Ignored

- ndraws:

  Number of joint posterior draws for `format = "array"` or `"long"` on
  a fit carrying a
  [`spatiotemporal()`](https://gillescolling.com/tulpa/reference/spatiotemporal.md)
  interaction.

- seed:

  Optional integer seed for those draws (RNG state is restored on exit).

- x:

  A `tulpa_st_summary` table.

## Value

For `format = "array"`, an S x T x draws numeric array carrying
`n_spatial`, `n_times` and `n_draws` attributes; for `"long"`, a data
frame with columns `s`, `t`, `draw` and `value`; for `"summary"`, a
`tulpa_st_summary` data frame with columns `s`, `t`, `mean`, one column
per entry of `probs` and `sd`, which has
[`print()`](https://rdrr.io/r/base/print.html) and
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) methods.

## Details

On a fit carrying a
[`spatiotemporal()`](https://gillescolling.com/tulpa/reference/spatiotemporal.md)
interaction the summary is the exact Gaussian mixture over the outer
grid of each cell's posterior (mean, SD and the `probs` quantiles), the
read
[`latent_effects()`](https://gillescolling.com/tulpa/reference/latent_effects.md)
makes for a
[`latent()`](https://gillescolling.com/tulpa/reference/latent.md) block.
The `"array"` and `"long"` formats are joint draws of all S x T cells: a
grid cell drawn by its weight, then the latent vector from that cell's
Gaussian conditioned on the model's constraints, so a derived quantity
over cells (a sum over areas, a change between two times) carries the
cells' correlation.

## Examples

``` r
# A Knorr-Held interaction block is attached by a model package's own
# fitter (see spatiotemporal() for what the engine fits). The accessor reads
# its layout and an n_draws x (S * T) draw matrix stored s-fastest: column
# (t - 1) * S + s holds region s at time t. Here both are simulated for
# S = 3 regions over T = 4 times.
set.seed(1)
S <- 3L; n_t <- 4L
st_mean <- outer(c(-0.5, 0, 0.5), seq(-0.3, 0.3, length.out = n_t))
draws <- matrix(rnorm(100 * S * n_t, mean = rep(c(st_mean), each = 100),
                      sd = 0.1), nrow = 100)
fit <- structure(
  list(spatiotemporal = list(n_spatial = S, n_times = n_t),
       .internal = list(spatiotemporal_draws = draws)),
  class = "tulpa_fit")

st_effects <- spatiotemporal_effects(fit, format = "summary")
head(st_effects)
#> Spatiotemporal interaction posterior summary: 3 locations x 4 times, 100 posterior draws
#>   s t         mean        q2.5         q50      q97.5         sd
#> 1 1 1  0.160888737 -0.01712975  0.16139092 0.32974683 0.08981994
#> 2 2 1 -0.003780808 -0.15911316 -0.01772172 0.20815038 0.09578791
#> 3 3 1 -0.147032646 -0.37872770 -0.15016061 0.02914034 0.10342655
#> 4 1 2  0.055160186 -0.15293965  0.05247047 0.22115838 0.09916751
#> 5 2 2 -0.003913424 -0.20671992 -0.01381795 0.23192542 0.11696122
#> 6 3 2 -0.054451936 -0.26602449 -0.05567811 0.11191994 0.09661804
dim(spatiotemporal_effects(fit, format = "array"))
#> [1]   3   4 100
```
