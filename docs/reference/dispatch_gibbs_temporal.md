# Dispatch a temporal Polya-Gamma Gibbs fit

The temporal analogue of
[`dispatch_gibbs_spatial()`](https://gillescolling.com/tulpa/reference/dispatch_gibbs_spatial.md):
maps a validated
[`temporal_multiscale()`](https://gillescolling.com/tulpa/reference/temporal_multiscale.md)
spec onto the multiscale temporal Polya-Gamma sampler
(`cpp_pg_binomial_gibbs_temporal`), which composes an additive RW1
trend + cyclic-RW1 seasonal + AR1/IID short-term decomposition. Binomial
only. The C++ kernel implements an RW1 trend (rw2 is rejected here
rather than silently downgraded).

## Usage

``` r
dispatch_gibbs_temporal(
  y,
  n_trials,
  X,
  re_group,
  n_re_groups,
  temporal,
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
