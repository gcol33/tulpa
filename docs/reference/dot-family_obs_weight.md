# Observed curvature (-d^2 log-lik / d eta^2 at the realized `y`), elementwise.

Falls back to the y-free `weight` for families whose curvature carries
no `y` – those where the response enters the log-likelihood linearly in
`eta` (Poisson, binomial, and the zero-truncated Poisson), for which the
observed and expected curvature coincide identically rather than only in
expectation.

## Usage

``` r
.family_obs_weight(eta, y, family, n_trials = NULL, phi = 1, phi2 = NULL)
```
