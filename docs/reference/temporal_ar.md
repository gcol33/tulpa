# AR(p) temporal latent field (general-order autoregressive)

A stationary autoregressive process of order `p`, \\w_t =
\sum\_{j=1}^{p} \phi_j w\_{t-j} + \varepsilon_t\\, as a user-defined
GMRF latent block for a model formula via `latent(temporal_ar(...))` –
the general-order extension of
[`temporal_ar2()`](https://gillescolling.com/tulpa/reference/temporal_ar2.md),
sharing the same construction: the precision is the exact stationary
AR(p) GMRF (the banded inverse of the Yule-Walker Toeplitz
autocovariance), and stationarity is enforced by the
partial-autocorrelation parameterization (Levinson-Durbin map of `p`
unconstrained `atanh_psi` hyperparameters), so the hyperparameter
integration never leaves the stationary region.

The block's hyperparameters are `log_tau` (log innovation precision) and
`atanh_psi1`..`atanh_psi<p>`; the AR coefficients are recovered through
`psi_j = tanh(atanh_psi_j)` and the Levinson-Durbin recursion. Note the
outer integration cost grows with the hyperparameter count `p + 1`;
orders beyond 3-4 are better served by a sampler tier.

## Usage

``` r
temporal_ar(
  time_idx,
  p = 2L,
  n_times = NULL,
  prior_tau_sd = 2,
  prior_psi_sd = 1.5,
  name = NULL
)
```

## Arguments

- time_idx:

  Integer vector (1-based) of the time point for each observation.

- p:

  Autoregressive order (\>= 1).

- n_times:

  Number of distinct time points; defaults to `max(time_idx)`.

- prior_tau_sd, prior_psi_sd:

  Prior SDs for `log_tau` and the `atanh_psi` hyperparameters
  (weakly-informative Gaussian). Defaults 2 / 1.5.

- name:

  Optional block label (default `ar<p>`).

## Value

A `tgmrf` / `tulpa_latent_block` object.

## See also

[`temporal_ar2()`](https://gillescolling.com/tulpa/reference/temporal_ar2.md)
(the `p = 2` shorthand),
[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)
for the first-order formula-integrated process,
[`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md) for the
general user-defined GMRF interface.

## Examples

``` r
# \donttest{
set.seed(1)
Tt <- 150L
w <- numeric(Tt); w[1:3] <- rnorm(3)
for (t in 4:Tt) w[t] <- 0.4 * w[t-1] + 0.2 * w[t-2] - 0.2 * w[t-3] + rnorm(1, 0, 0.4)
d <- data.frame(t = seq_len(Tt), y = w + rnorm(Tt, 0, 0.3))
fit <- tulpa(y ~ latent(temporal_ar(d$t, p = 3)), data = d,
             family = "gaussian", mode = "nested_laplace")
# }
```
