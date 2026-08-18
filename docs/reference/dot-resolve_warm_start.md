# Resolve a `warm_start` request into sampler init / inverse-mass vectors

Accepts the front door's `warm_start`: `"eb"` or `"laplace"` to run that
fit first, or an already-fitted object from either. Returns `NULL` when
no warm start was asked for, so the caller passes nothing and the kernel
keeps its own defaults.

## Usage

``` r
.resolve_warm_start(warm_start, args, re_terms, sigma_re, beta_prior, n_chains)
```

## Arguments

- warm_start:

  `NULL`, `"eb"`, `"laplace"`, or a fitted `tulpa_fit`.

- args:

  The assembled sampler arguments (the same values the kernel will
  receive), used both to run the source fit and to probe the target
  layout.

- re_terms:

  Random-effect terms in
  [`tulpa_eb()`](https://gillescolling.com/tulpa/reference/tulpa_eb.md)
  /
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  form.

- sigma_re:

  Random-effect SDs to condition on, for the Laplace source.

- beta_prior:

  Optional fixed-effect prior, threaded into the source fit.

- n_chains:

  Number of chains the sampler will run.
