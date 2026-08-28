# Dispatch a spatial Polya-Gamma Gibbs fit to the correct sampler

The Gibbs analogue of
[`dispatch_laplace_spatial()`](https://gillescolling.com/tulpa/reference/dispatch_laplace_spatial.md):
routes on `spatial$type` to the matching
`cpp_pg_<family>_gibbs_<structure>` sampler, building the neighbour-list
/ coordinate inputs each one needs. The binomial Polya-Gamma
augmentation backs the full areal (icar/bym2/rsr) + continuous (gp/nngp/
multiscale_gp) family; `neg_binomial_2` is backed by the single areal
ICAR negbin sampler (`cpp_pg_negbin_gibbs_spatial`), the only negbin
spatial kernel.

## Usage

``` r
dispatch_gibbs_spatial(
  y,
  n_trials,
  X,
  re_group,
  n_re_groups,
  spatial,
  family,
  iter,
  warmup,
  thin = 1L,
  prior_beta_sd = .tulpa_prior_sd("gibbs"),
  prior_sigma_re_scale = 2.5,
  verbose = FALSE,
  n_threads = 1L
)
```
