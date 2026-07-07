# temporal_ar2.R
# ------------------------------------------------------------------------------
# Stationary autoregressive AR(p) temporal fields as user-defined GMRF blocks
# (tgmrf), so they reuse the nested-Laplace / IMH / NUTS machinery with no new
# C++ kernel. The precision is the EXACT stationary AR(p) GMRF: the marginal
# covariance is the Toeplitz matrix of the Yule-Walker autocovariances and the
# precision is its inverse, banded at width p by the order-p Markov property.
# Stationarity is guaranteed by the partial-autocorrelation (PACF)
# parameterization: the Levinson-Durbin recursion (Barndorff-Nielsen & Schou
# 1973; Monahan 1984) maps psi_1..psi_p in (-1, 1)^p bijectively onto the
# stationary AR(p) region, so the hyperparameter grid never leaves it.
#
# Used through the formula latent-block hook: latent(temporal_ar2(time_idx)) /
# latent(temporal_ar(time_idx, p = )). The formula-integrated `temporal =`
# route (a C++ banded kernel) is future work; these blocks take the
# per-observation integer time index directly.
# ------------------------------------------------------------------------------

# PACF (psi in (-1,1)^p) -> stationary AR coefficients phi, via the
# Levinson-Durbin recursion. p = 2 reduces to phi1 = psi1 (1 - psi2),
# phi2 = psi2.
.arp_pacf_to_phi <- function(psi) {
  p <- length(psi)
  phi <- numeric(0)
  for (k in seq_len(p)) {
    if (k > 1L) {
      phi <- c(phi - psi[k] * rev(phi), psi[k])
    } else {
      phi <- psi[1L]
    }
  }
  phi
}

# Yule-Walker autocovariances gamma_0..gamma_{n-1} of a stationary AR(p) with
# coefficients `phi` and innovation variance s2. gamma_0..gamma_p solve the
# linear Yule-Walker system gamma_k - sum_j phi_j gamma_{|k-j|} = s2 1{k=0},
# k = 0..p (with gamma_{-k} = gamma_k); higher lags extend by the recursion
# gamma_k = sum_j phi_j gamma_{k-j}.
.arp_autocov <- function(n, phi, s2) {
  p <- length(phi)
  A <- diag(1, p + 1L)
  for (k in 0:p) {
    for (j in seq_len(p)) {
      idx <- abs(k - j) + 1L
      A[k + 1L, idx] <- A[k + 1L, idx] - phi[j]
    }
  }
  b <- c(s2, rep(0, p))
  g <- as.numeric(solve(A, b))            # gamma_0..gamma_p
  if (n > p + 1L) {
    g <- c(g, numeric(n - p - 1L))
    for (k in (p + 2L):n) {
      g[k] <- sum(phi * g[k - seq_len(p)])
    }
  }
  g[seq_len(n)]
}

# Exact stationary AR(p) precision on `n` equally-spaced time points: the
# inverse of the Toeplitz autocovariance, truncated to its width-p band (the
# exact structure by the order-p Markov property; off-band entries are
# floating-point error only). `tau` is the innovation precision.
.arp_precision <- function(n, phi, tau) {
  p <- length(phi)
  if (n < p + 1L) {
    stop(sprintf("AR(%d) needs at least %d time points.", p, p + 1L),
         call. = FALSE)
  }
  g <- .arp_autocov(n, phi, 1 / tau)
  if (!all(is.finite(g)) || g[1] <= 0) {
    stop(sprintf("AR(%d) coefficients too close to the stationarity boundary.",
                 p), call. = FALSE)
  }
  Sigma <- stats::toeplitz(g)
  Q <- solve(Sigma)
  Q[abs(row(Q) - col(Q)) > p] <- 0        # width-p band (order-p Markov)
  Q <- 0.5 * (Q + t(Q))                    # symmetrize float round-off
  methods::as(methods::as(Q, "generalMatrix"), "CsparseMatrix")
}

# Order-2 wrappers (the AR(2) block predates the general construction; kept as
# the p = 2 specialization of the shared helpers).
.ar2_precision <- function(n, phi1, phi2, tau) {
  .arp_precision(n, c(phi1, phi2), tau)
}
.ar2_pacf_to_phi <- function(psi1, psi2) {
  stats::setNames(.arp_pacf_to_phi(c(psi1, psi2)), c("phi1", "phi2"))
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
  temporal_ar(time_idx, p = 2L, n_times = n_times,
              prior_tau_sd = prior_tau_sd, prior_psi_sd = prior_psi_sd,
              name = name)
}

#' AR(p) temporal latent field (general-order autoregressive)
#'
#' @description
#' A stationary autoregressive process of order `p`,
#' \eqn{w_t = \sum_{j=1}^{p} \phi_j w_{t-j} + \varepsilon_t}, as a
#' user-defined GMRF latent block for a model formula via
#' `latent(temporal_ar(...))` -- the general-order extension of
#' [temporal_ar2()], sharing the same construction: the precision is the exact
#' stationary AR(p) GMRF (the banded inverse of the Yule-Walker Toeplitz
#' autocovariance), and stationarity is enforced by the partial-autocorrelation
#' parameterization (Levinson-Durbin map of `p` unconstrained
#' `atanh_psi` hyperparameters), so the hyperparameter integration never
#' leaves the stationary region.
#'
#' The block's hyperparameters are `log_tau` (log innovation precision) and
#' `atanh_psi1`..`atanh_psi<p>`; the AR coefficients are recovered through
#' `psi_j = tanh(atanh_psi_j)` and the Levinson-Durbin recursion. Note the
#' outer integration cost grows with the hyperparameter count `p + 1`; orders
#' beyond 3-4 are better served by a sampler tier.
#'
#' @param time_idx Integer vector (1-based) of the time point for each
#'   observation.
#' @param p Autoregressive order (>= 1).
#' @param n_times Number of distinct time points; defaults to `max(time_idx)`.
#' @param prior_tau_sd,prior_psi_sd Prior SDs for `log_tau` and the
#'   `atanh_psi` hyperparameters (weakly-informative Gaussian). Defaults 2 / 1.5.
#' @param name Optional block label (default `ar<p>`).
#'
#' @return A `tgmrf` / `tulpa_latent_block` object.
#'
#' @seealso [temporal_ar2()] (the `p = 2` shorthand), [temporal_ar1()] for the
#'   first-order formula-integrated process, [tgmrf()] for the general
#'   user-defined GMRF interface.
#' @examples
#' \donttest{
#' set.seed(1)
#' Tt <- 150L
#' w <- numeric(Tt); w[1:3] <- rnorm(3)
#' for (t in 4:Tt) w[t] <- 0.4 * w[t-1] + 0.2 * w[t-2] - 0.2 * w[t-3] + rnorm(1, 0, 0.4)
#' d <- data.frame(t = seq_len(Tt), y = w + rnorm(Tt, 0, 0.3))
#' fit <- tulpa(y ~ latent(temporal_ar(d$t, p = 3)), data = d,
#'              family = "gaussian", mode = "nested_laplace")
#' }
#' @export
temporal_ar <- function(time_idx, p = 2L, n_times = NULL,
                        prior_tau_sd = 2, prior_psi_sd = 1.5,
                        name = NULL) {
  p <- as.integer(p)
  if (is.na(p) || p < 1L) stop("`p` must be an integer >= 1.", call. = FALSE)
  time_idx <- as.integer(time_idx)
  if (anyNA(time_idx) || any(time_idx < 1L)) {
    stop("`time_idx` must be positive 1-based integers.", call. = FALSE)
  }
  n_times <- if (is.null(n_times)) max(time_idx) else as.integer(n_times)
  if (n_times < p + 1L) {
    stop(sprintf("AR(%d) needs at least %d time points.", p, p + 1L),
         call. = FALSE)
  }

  Q_fun <- function(theta) {
    # `[[` strips element names so the derived phi keeps clean names.
    tau <- exp(theta[[1]])
    psi <- tanh(vapply(seq_len(p) + 1L, function(j) theta[[j]], numeric(1)))
    .arp_precision(n_times, .arp_pacf_to_phi(psi), tau)
  }
  prior_fun <- function(theta) {
    stats::dnorm(theta[1], 0, prior_tau_sd, log = TRUE) +
      sum(stats::dnorm(theta[seq_len(p) + 1L], 0, prior_psi_sd, log = TRUE))
  }

  # Mildly-positive first PACF start, tapering to 0 for higher orders
  # (matching the temporal_ar2 defaults at p = 2).
  psi_init <- c(0.3, 0.1, rep(0, max(0L, p - 2L)))[seq_len(p)]
  init <- c(0, atanh(psi_init))
  names(init) <- c("log_tau", paste0("atanh_psi", seq_len(p)))

  tgmrf(
    Q       = Q_fun,
    prior   = prior_fun,
    init    = init,
    obs_idx = time_idx,
    name    = name %||% paste0("ar", p)
  )
}
