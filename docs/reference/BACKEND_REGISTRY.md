# Backend registry – single source of truth for the inference backends.

One entry per backend. Adding a backend is a single entry here; tier
membership, family support, the input contract, and R-level reachability
all derive from this list – the single source of truth for per-tier
`backends` and family support.

Fields:

- `emits` – the kind of posterior representation the backend's draws
  hold, which fixes whether MCMC chain diagnostics apply (independent of
  `tier`: an exact SMC sampler emits `"iid"` particles, a Tier-3 VI fit
  also emits `"iid"`, while a nested-Laplace Tier-2 fit emits `"iid"`
  too):

  - `"chain"` : autocorrelated MCMC output – Rhat and
    autocorrelation-ESS are meaningful
    ([`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
    computes them).

  - `"iid"` : exchangeable draws from a deterministic approximation, or
    resampled particles – split-Rhat is vacuous and ESS = n by
    construction, so they say nothing about approximation bias.

  - `"point"` : no posterior sample (mode + covariance only).

- `tier` – tier key (`"exact"`, `"structured"`, `"optimized"`).

- `input` – the input contract the backend's fitter consumes:

  - `"design"` : design-matrix bundle (`y`, `n_trials`, `X`, RE
    structure).

  - `"logpost"` : a `log_posterior(theta)` closure plus dimension/init.

  - `"modeldata"` : a tulpa `ModelData` / `LikelihoodSpec` (C-ABI NUTS
    path).

  - `"nested"` : a design-matrix bundle plus one or more latent prior
    blocks (`latent(tgmrf(...))`), integrated over the block
    hyperparameters by the nested-Laplace driver.

  - `"spde"` : a design bundle plus a self-contained SPDE spec (mesh +
    FEM matrices); fit_spde()'s own CCD / grid hyperparameter engine.

- `fitter` – name of the R function implementing the backend, or `NULL`
  when only a C++ kernel exists with no R entry point yet. Stored as a
  *string* (not the function object) so the registry is independent of
  source-load order; resolved lazily via
  [`resolve_backend_fitter()`](https://gillescolling.com/tulpa/reference/resolve_backend_fitter.md).
  `NULL` =\> not selectable from R; dispatch fails loudly.

- `families` – character vector of supported family identifiers, or
  `NULL` for an unrestricted backend.

- `cabi` – the registered C-ABI callable backing the backend (the symbol
  a model package reaches via `LinkingTo: tulpa`, and the one an R
  wrapper would call), or `NULL`.

- `note` – optional human-readable note.

Family identity (for `families`) is checked against `family$name`,
`family$distribution`, and `family$numerator$distribution`
(tulpaRatio-style ratio families nest a per-process distribution).

## Usage

``` r
BACKEND_REGISTRY
```
