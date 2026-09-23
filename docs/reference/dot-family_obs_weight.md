# Observed curvature (-d^2 log-lik / d eta^2 at the realized `y`), elementwise.

One path, the compiled dispatch (`obs_grad_hess_for_family`), for every
family. It used to answer from the registry's `obs_weight` closure where
a family registered one and substitute the y-free EXPECTED weight where
none was registered. That substitution is exact only where the response
enters the log-likelihood linearly in `eta` (Poisson, binomial, the
zero-truncated Poisson); for beta, gamma, inverse_gaussian,
beta_binomial, tweedie and t it is a different function, and can carry
the opposite sign – at `y = 12`, `eta = -0.9` the beta_binomial observed
curvature is `-0.125` against an expected weight of `+0.658`.
beta_binomial is in `.ZI_FAMILIES`, so the zero-inflation mixture in
`R/family_zi.R` differentiated through the wrong one (gcol33/tulpa#824).

## Usage

``` r
.family_obs_weight(eta, y, family, n_trials = NULL, phi = 1, phi2 = NULL)
```

## Details

The registry's three `obs_weight` closures stay as the R oracle the
compiled dispatch is pinned against in
`test-family-registry-compiled.R`.
