# temporal_ar2.R
# ------------------------------------------------------------------------------
# Second-order autoregressive AR(2) temporal field as a user-defined GMRF block
# (tgmrf), so it reuses the nested-Laplace / IMH / NUTS machinery with no new C++
# kernel. The precision is the EXACT stationary AR(2) GMRF: the marginal
# covariance is the Toeplitz matrix of the Yule-Walker autocovariances and the
# precision is its inverse, which is pentadiagonal by the order-2 Markov property.
# Stationarity is guaranteed by the partial-autocorrelation (PACF)
# parameterization phi2 = psi2, phi1 = psi1 (1 - psi2) with psi1, psi2 in (-1, 1),
# which maps the open square bijectively onto the AR(2) stationarity triangle.
#
# Used through the formula latent-block hook: latent(temporal_ar2(time_idx)).
# The formula-integrated temporal = temporal_ar2("year") path (a C++ pentadiagonal
# kernel on the dedicated temporal route) is future work; this block takes the
# per-observation integer time index directly.
# ------------------------------------------------------------------------------

# Exact stationary AR(2) precision on `n` equally-spaced time points.
# `phi1`, `phi2` are the AR coefficients (assumed stationary); `tau` is the
# innovation precision (1 / Var(eps)). Yule-Walker autocovariances:
#   gamma0 = (1/tau) (1 - phi2) / [(1 + phi2)((1 - phi2)^2 - phi1^2)]
#   gamma1 = phi1 gamma0 / (1 - phi2);  gamma_k = phi1 gamma_{k-1} + phi2 gamma_{k-2}
# The precision is solve(Toeplitz(gamma)), truncated to its pentadiagonal band
# (the exact structure; off-band entries are floating-point error only).
.ar2_precision <- function(n, phi1, phi2, tau) {
  if (n < 3L) stop("AR(2) needs at least 3 time points.", call. = FALSE)
  denom <- (1 + phi2) * ((1 - phi2)^2 - phi1^2)
  if (!is.finite(denom) || abs(denom) < 1e-10) {
    stop("AR(2) coefficients too close to the stationarity boundary.",
         call. = FALSE)
  }
  s2 <- 1 / tau
  g <- numeric(n)
  g[1] <- s2 * (1 - phi2) / denom
  g[2] <- phi1 * g[1] / (1 - phi2)
  for (k in 3:n) g[k] <- phi1 * g[k - 1] + phi2 * g[k - 2]
  Sigma <- stats::toeplitz(g)
  Q <- solve(Sigma)
  Q[abs(row(Q) - col(Q)) > 2] <- 0        # pentadiagonal (order-2 Markov)
  Q <- 0.5 * (Q + t(Q))                    # symmetrize float round-off
  methods::as(methods::as(Q, "generalMatrix"), "CsparseMatrix")
}

# PACF (psi) -> AR coefficients (phi), always inside the stationarity triangle.
.ar2_pacf_to_phi <- function(psi1, psi2) {
  c(phi1 = psi1 * (1 - psi2), phi2 = psi2)
}

#' AR(2) temporal latent field (second-order autoregressive)
#'
#' @description
#' A stationary second-order autoregressive process
#' \eqn{w_t = \phi_1 w_{t-1} + \phi_2 w_{t-2} + \varepsilon_t} as a user-defined
#' GMRF latent block, for use in a model formula via
#' `latent(temporal_ar2(...))`. It reuses tulpa's nested-Laplace / NUTS
#' machinery (no dedicated C++ kernel). The precision is the exact stationary
#' AR(2) GMRF; stationarity is enforced by the PACF parameterization, so the
#' hyperparameter grid never leaves the stationarity region.
#'
#' The block's hyperparameters are `log_tau` (log innovation precision) and
#' `atanh_psi1`, `atanh_psi2` (unconstrained partial autocorrelations); the AR
#' coefficients are recovered as `phi2 = tanh(atanh_psi2)`,
#' `phi1 = tanh(atanh_psi1) (1 - phi2)`.
#'
#' @param time_idx Integer vector (1-based) of the time point for each
#'   observation.
#' @param n_times Number of distinct time points; defaults to `max(time_idx)`.
#' @param prior_tau_sd,prior_psi_sd Prior SDs for `log_tau` and the two
#'   `atanh_psi` hyperparameters (weakly-informative Gaussian). Defaults 2 / 1.5.
#' @param name Optional block label.
#'
#' @return A `tgmrf` / `tulpa_latent_block` object.
#'
#' @seealso [temporal_ar1()] for the first-order (formula-integrated) process;
#'   [tgmrf()] for the general user-defined GMRF interface.
#' @examples
#' \donttest{
#' set.seed(1)
#' Tt <- 120L
#' w <- numeric(Tt); w[1:2] <- rnorm(2)
#' for (t in 3:Tt) w[t] <- 0.5 * w[t-1] + 0.3 * w[t-2] + rnorm(1, 0, 0.4)
#' d <- data.frame(t = seq_len(Tt), y = w + rnorm(Tt, 0, 0.3))
#' fit <- tulpa(y ~ latent(temporal_ar2(d$t)), data = d, family = "gaussian",
#'              mode = "nested_laplace")
#' }
#' @export
temporal_ar2 <- function(time_idx, n_times = NULL,
                         prior_tau_sd = 2, prior_psi_sd = 1.5,
                         name = "ar2") {
  time_idx <- as.integer(time_idx)
  if (anyNA(time_idx) || any(time_idx < 1L)) {
    stop("`time_idx` must be positive 1-based integers.", call. = FALSE)
  }
  n_times <- if (is.null(n_times)) max(time_idx) else as.integer(n_times)
  if (n_times < 3L) stop("AR(2) needs at least 3 time points.", call. = FALSE)

  Q_fun <- function(theta) {
    # `[[` strips element names so the derived phi keeps clean names (a named
    # theta element would otherwise give c(phi1 = <named>) a compound name).
    tau  <- exp(theta[[1]])
    psi1 <- tanh(theta[[2]]); psi2 <- tanh(theta[[3]])
    phi  <- .ar2_pacf_to_phi(psi1, psi2)
    .ar2_precision(n_times, phi[["phi1"]], phi[["phi2"]], tau)
  }
  prior_fun <- function(theta) {
    stats::dnorm(theta[1], 0, prior_tau_sd, log = TRUE) +
      stats::dnorm(theta[2], 0, prior_psi_sd, log = TRUE) +
      stats::dnorm(theta[3], 0, prior_psi_sd, log = TRUE)
  }

  tgmrf(
    Q       = Q_fun,
    prior   = prior_fun,
    init    = c(log_tau = 0, atanh_psi1 = atanh(0.3), atanh_psi2 = atanh(0.1)),
    obs_idx = time_idx,
    name    = name
  )
}
