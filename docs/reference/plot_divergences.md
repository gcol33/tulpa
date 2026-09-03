# Plot Divergent Transitions

Creates visualizations to investigate divergent transitions. Parallel
coordinates and scatter plots highlight where in parameter space
divergences occur.

## Usage

``` r
plot_divergences(fit, pars = NULL, type = c("parcoord", "scatter"))
```

## Arguments

- fit:

  A `tulpa_fit` object (HMC backend).

- pars:

  Character vector of parameter names. If NULL, uses variance
  parameters.

- type:

  Plot type: "parcoord" (parallel coordinates) or "scatter".

## Value

A ggplot object or base R plot (invisible).

## Details

Divergent transitions indicate regions of high posterior curvature that
the sampler cannot efficiently explore. Common causes:

- Very narrow funnels (hierarchical models)

- Strong correlations

- Multi-modality

## See also

[`plot_pairs()`](https://gillescolling.com/tulpa/reference/plot_pairs.md),
[`n_divergent()`](https://gillescolling.com/tulpa/reference/n_divergent.md)

## Examples

``` r
# \donttest{
set.seed(123)
df <- data.frame(x = rnorm(60))
df$y <- rpois(60, exp(0.5 + 0.3 * df$x))
fit <- tulpa(y ~ x, data = df, family = "poisson", mode = "hmc",
             control = list(n_iter = 500L, warmup = 250L, n_chains = 2L,
                            seed = 1L))
plot_divergences(fit)
plot_divergences(fit, type = "scatter")
# }
```
