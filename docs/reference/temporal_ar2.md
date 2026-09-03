# AR(2) temporal latent field (second-order autoregressive)

A stationary second-order autoregressive process \\w_t = \phi_1
w\_{t-1} + \phi_2 w\_{t-2} + \varepsilon_t\\ as a user-defined GMRF
latent block, for use in a model formula via
`latent(temporal_ar2(...))`. It reuses tulpa's nested-Laplace / NUTS
machinery (no dedicated C++ kernel). The precision is the exact
stationary AR(2) GMRF; stationarity is enforced by the PACF
parameterization, so the hyperparameter grid never leaves the
stationarity region.

The block's hyperparameters are `log_tau` (log innovation precision) and
`atanh_psi1`, `atanh_psi2` (unconstrained partial autocorrelations); the
AR coefficients are recovered as `phi2 = tanh(atanh_psi2)`,
`phi1 = tanh(atanh_psi1) (1 - phi2)`.

## Usage

``` r
temporal_ar2(
  time_idx,
  n_times = NULL,
  prior_tau_sd = 2,
  prior_psi_sd = 1.5,
  name = "ar2"
)
```

## Arguments

- time_idx:

  Integer vector (1-based) of the time point for each observation.

- n_times:

  Number of distinct time points; defaults to `max(time_idx)`.

- prior_tau_sd, prior_psi_sd:

  Prior SDs for `log_tau` and the two `atanh_psi` hyperparameters
  (weakly-informative Gaussian). Defaults 2 / 1.5.

- name:

  Optional block label.

## Value

A `tgmrf` / `tulpa_latent_block` object.

## See also

[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)
for the first-order (formula-integrated) process;
[`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md) for the
general user-defined GMRF interface.

## Examples

``` r
# \donttest{
set.seed(1)
Tt <- 120L
w <- numeric(Tt); w[1:2] <- rnorm(2)
for (t in 3:Tt) w[t] <- 0.5 * w[t-1] + 0.3 * w[t-2] + rnorm(1, 0, 0.4)
d <- data.frame(t = seq_len(Tt), y = w + rnorm(Tt, 0, 0.3))
fit <- tulpa(y ~ latent(temporal_ar2(d$t)), data = d, family = "gaussian",
             mode = "nested_laplace")
# }
```
