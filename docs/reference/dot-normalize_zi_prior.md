# Normalize a zero-inflation prior to a scalar SD

The compiled zero-inflated kernels carry one mean-zero Gaussian prior
over the whole `beta_zi` block (`ModelData::zi_prior_sd` on the sampler
paths, the appended `BetaPrior` tail on the Laplace path), so the prior
is a single SD rather than the per-coefficient `list(mean, sd)` that
`beta_prior` takes. The list form is kept so the two priors read alike
at the front door and so a per-coefficient or non-zero-mean ZI prior can
be added without a signature change.

## Usage

``` r
.normalize_zi_prior(zi_prior)
```

## Arguments

- zi_prior:

  `NULL`, or a list with a scalar `sd`.

## Value

A scalar prior SD; the engine default when `zi_prior` is `NULL`.
