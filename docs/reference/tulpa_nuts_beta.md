# Fit a beta-regression model via NUTS (joint sampling of beta + log_phi)

Bayesian counterpart to
[`tulpa_laplace_beta()`](https://gillescolling.com/tulpa/reference/tulpa_laplace_beta.md).
Routes the Beta GLM through tulpa's full NUTS backend via the generic
`LikelihoodSpec` interface, sampling the precision `phi` jointly with
the regression coefficients on the log scale. NUTS performs exact
marginalisation over `phi`, replacing the Brent outer-opt in
[`tulpa_laplace_beta()`](https://gillescolling.com/tulpa/reference/tulpa_laplace_beta.md)
and the deterministic nested-Laplace grid that would otherwise integrate
over the Beta dispersion.

Mean-precision parameterisation, default logit link:
`y_i ~ Beta(mu_i * phi, (1 - mu_i) * phi)` with
`mu_i = 1 / (1 + exp(-eta_i))` and `eta_i = X_i %*% beta`.

## Usage

``` r
tulpa_nuts_beta(
  y,
  X,
  beta_prior = .tulpa_default_beta_prior("beta_nuts"),
  log_phi_prior_sd = 3,
  log_phi_init = 0,
  control = list()
)
```

## Arguments

- y:

  Response vector, strictly in `(0, 1)`.

- X:

  Fixed-effects design matrix.

- beta_prior:

  Fixed-effect prior as `list(mean, sd)`: a mean-zero (`mean = 0`)
  Gaussian on each coefficient with SD `sd` (default the engine default,
  `prior_normal(0, 2.5)`).

- log_phi_prior_sd:

  Prior SD on `log(phi)` (`log_phi ~ N(0, log_phi_prior_sd)`). Default
  `3` (very weak; covers `phi` from ~0.001 to ~1000 within +-2 SD).

- log_phi_init:

  Starting value for `log(phi)`. Default `0` (i.e. `phi = 1`); a
  method-of-moments warm start can speed warmup in highly concentrated
  regimes.

- control:

  A named list of numerical / sampler knobs (statistical arguments stay
  in the signature): `n_iter` (default 2000), `n_warmup` (default 1000),
  `max_treedepth` (default 10), `adapt_delta` (default 0.8), `seed`
  (`NULL` draws from the session RNG), `verbose` (default FALSE).

## Value

A list with:

- `draws` – `n_samples x (p + 1)` matrix of post-warmup draws, columns
  `beta[1] ... beta[p], log_phi`.

- `means` – posterior means.

- `phi_summary` – posterior mean / median / quantiles of
  `phi = exp(log_phi)`.

- `accept_prob`, `divergent`, `treedepth`, `epsilon` – NUTS diagnostics
  from the underlying chain.

## See also

[`tulpa_laplace_beta()`](https://gillescolling.com/tulpa/reference/tulpa_laplace_beta.md)
for the Laplace + Brent point estimate;
[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
for the underlying Laplace engine.

## Examples

``` r
set.seed(1)
n <- 150L
X <- cbind(1, rnorm(n))
mu <- plogis(X %*% c(0.2, 0.7)); phi <- 8
y <- rbeta(n, mu * phi, (1 - mu) * phi)
# \donttest{
fit <- tulpa_nuts_beta(y, X, control = list(n_iter = 500L, n_warmup = 250L))
colMeans(fit$draws)
# }
```
