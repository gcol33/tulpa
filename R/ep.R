# ep.R
# ------------------------------------------------------------------------------
# Expectation Propagation (Minka 2001; Rasmussen & Williams 2006, Alg. 3.5) for a
# GLM with a Gaussian prior on the coefficients: beta ~ N(0, s0^2 I),
# y_i | beta ~ family(eta_i = x_i' beta). EP approximates the posterior by a
# Gaussian N(m, V) built from one Gaussian site per observation on the 1-D linear
# predictor eta_i. Each sweep, for every site: form the cavity (posterior with
# that site removed), match the moments of the tilted distribution
# cavity(eta_i) x p(y_i | eta_i) by Gauss-Hermite quadrature, and update the site.
#
# EP is the "approximation" layer counterpart to Laplace/VI: it is EXACT when the
# likelihood is Gaussian (every factor is then Gaussian), and unlike Laplace it
# matches marginal moments rather than the mode's curvature, so it is typically
# more accurate on skewed GLM likelihoods -- the regime the debias story targets.
# ------------------------------------------------------------------------------

# Gauss-Hermite nodes/weights (physicists', weight exp(-x^2)) via Golub-Welsch:
# eigendecomposition of the symmetric tridiagonal Jacobi matrix (dependency-free).
.gauss_hermite <- function(n) {
  if (n < 2L) return(list(x = 0, w = sqrt(pi)))
  b <- sqrt(seq_len(n - 1L) / 2)
  J <- matrix(0, n, n)
  J[cbind(seq_len(n - 1L), 2:n)] <- b
  J[cbind(2:n, seq_len(n - 1L))] <- b
  e <- eigen(J, symmetric = TRUE)
  x <- e$values
  w <- sqrt(pi) * (e$vectors[1L, ])^2
  o <- order(x)
  list(x = x[o], w = w[o])
}

# Moments of the tilted distribution N(eta; mu, s2) * exp(loglik(eta; y)) over a
# single site, via Gauss-Hermite quadrature. Returns Z (normalizer), the tilted
# mean and variance. `gh` is the output of .gauss_hermite().
.ep_tilted_moments <- function(mu, s2, y, family, phi, n_trials, gh) {
  # Gaussian likelihood: the tilted distribution is exactly Gaussian, so use its
  # closed-form moments (EP is then exact, no quadrature error). phi = variance.
  if (family == "gaussian") {
    v  <- 1 / (1 / s2 + 1 / phi)
    m1 <- v * (mu / s2 + y / phi)
    return(list(logZ = NA_real_, mean = m1, var = max(v, 1e-10)))
  }
  sd <- sqrt(s2)
  eta <- mu + sqrt(2) * sd * gh$x                 # change of variable
  wt  <- gh$w / sqrt(pi)
  ll  <- family_loglik(eta, rep(y, length(eta)), family,
                       n_trials = rep(n_trials, length(eta)), phi = phi)
  lw  <- log(wt) + ll
  m   <- max(lw)
  ew  <- exp(lw - m)
  Z0  <- sum(ew)
  if (!is.finite(Z0) || Z0 <= 0) return(NULL)
  p   <- ew / Z0
  m1  <- sum(p * eta)
  m2  <- sum(p * eta^2)
  v   <- m2 - m1^2
  list(logZ = log(Z0) + m, mean = m1, var = max(v, 1e-10))
}

#' Expectation-Propagation fit for a GLM
#'
#' @description
#' Fits a generalized linear model with a Gaussian coefficient prior by
#' Expectation Propagation: the posterior is approximated by a Gaussian whose
#' per-observation sites match the moments of the tilted distribution (via
#' Gauss-Hermite quadrature). EP is exact for a Gaussian likelihood and typically
#' more accurate than Laplace on skewed likelihoods, since it matches marginal
#' moments rather than the mode curvature.
#'
#' @param formula Model formula.
#' @param data A data frame.
#' @param family Character family name (see [family_names()]).
#' @param phi Dispersion / precision passed to the family (held fixed).
#' @param n_trials Binomial denominators (length `nrow(data)`), or `NULL` (= 1).
#' @param beta_prior_sd SD of the mean-zero Gaussian prior on every coefficient
#'   (default 10).
#' @param control List: `max_sweeps` (default 50), `tol` (default 1e-6),
#'   `damping` (default 0.8), `n_quad` (Gauss-Hermite nodes, default 20),
#'   `n_draws` (default 2000), `seed`.
#'
#' @return A `tulpa_fit` (subclass `tulpa_ep`) with `coefficients` (posterior
#'   mean), `vcov`, `draws`, `log_marginal` (the EP approximation), `converged`.
#'
#' @references Minka (2001). Expectation Propagation for approximate Bayesian
#'   inference. UAI. Rasmussen & Williams (2006). Gaussian Processes for Machine
#'   Learning, Algorithm 3.5.
#' @seealso [tulpa()] (Laplace / sampler tiers), [pathfinder()] (VI).
#' @examples
#' \donttest{
#' set.seed(1)
#' d <- data.frame(x = rnorm(200))
#' d$y <- rbinom(200, 1, plogis(-0.3 + 0.8 * d$x))
#' fit <- tulpa_ep(y ~ x, data = d, family = "binomial")
#' coef(fit)
#' }
#' @export
tulpa_ep <- function(formula, data, family = "binomial", phi = 1.0,
                     n_trials = NULL, beta_prior_sd = 10, control = list()) {
  if (is.null(.FAMILY_OPS[[family]])) {
    stop(sprintf("Unknown family '%s'. Supported: %s.",
                 family, paste(family_names(), collapse = ", ")), call. = FALSE)
  }
  max_sweeps <- as.integer(control$max_sweeps %||% 50L)
  tol        <- control$tol %||% 1e-6
  damping    <- control$damping %||% 0.8
  n_quad     <- as.integer(control$n_quad %||% 20L)
  n_draws    <- as.integer(control$n_draws %||% 2000L)

  mf <- stats::model.frame(formula, data)
  y  <- as.numeric(stats::model.response(mf))
  X  <- stats::model.matrix(stats::terms(mf), mf)
  n  <- nrow(X); p <- ncol(X)
  nt <- if (is.null(n_trials)) rep(1L, n) else as.integer(n_trials)
  gh <- .gauss_hermite(n_quad)

  P0 <- diag(1 / beta_prior_sd^2, p)               # prior precision
  tau <- rep(0, n); nu <- rep(0, n)                # site natural params (eta-space)

  recompute <- function(tau, nu) {
    P <- P0 + crossprod(X, tau * X)                # P0 + X' diag(tau) X
    V <- chol2inv(chol(P))
    m <- as.numeric(V %*% crossprod(X, nu))
    list(P = P, V = V, m = m)
  }
  st <- recompute(tau, nu)

  converged <- FALSE
  for (sweep in seq_len(max_sweeps)) {
    max_dtau <- 0
    for (i in seq_len(n)) {
      xi <- X[i, ]
      # Site marginal of eta_i under the current posterior.
      Vx  <- st$V %*% xi
      s2  <- as.numeric(xi %*% Vx)
      mu  <- as.numeric(xi %*% st$m)
      # Cavity: remove site i.
      inv_cav <- 1 / s2 - tau[i]
      if (inv_cav <= 1e-10) next                   # skip ill-defined cavity
      s2_cav <- 1 / inv_cav
      mu_cav <- s2_cav * (mu / s2 - nu[i])
      tm <- .ep_tilted_moments(mu_cav, s2_cav, y[i], family, phi, nt[i], gh)
      if (is.null(tm)) next
      tau_new <- 1 / tm$var - inv_cav
      nu_new  <- tm$mean / tm$var - mu_cav / s2_cav
      if (!is.finite(tau_new) || tau_new <= 0) next # keep the site PSD
      tau_d <- damping * (tau_new - tau[i])
      nu_d  <- damping * (nu_new  - nu[i])
      tau[i] <- tau[i] + tau_d; nu[i] <- nu[i] + nu_d
      max_dtau <- max(max_dtau, abs(tau_d))
    }
    st <- recompute(tau, nu)
    if (max_dtau < tol) { converged <- TRUE; break }
  }

  m <- st$m; V <- st$V
  pn <- colnames(X)
  names(m) <- pn; dimnames(V) <- list(pn, pn)

  # The EP posterior mean / covariance is the deliverable. The EP normalizing
  # constant (approximate marginal likelihood) is a separate multi-term formula
  # (R&W eq. 3.73); it is not returned here rather than shipped only partially
  # computed.
  log_marginal <- NA_real_

  if (!is.null(control$seed)) set.seed(as.integer(control$seed))
  draws <- .ps_rmvnorm(n_draws, m, V)

  fit <- list(
    coefficients = m, vcov = V, draws = draws, means = m, param_names = pn,
    log_marginal = log_marginal, converged = converged, n_sweeps = sweep,
    family = family, formula = formula, model_matrix = X,
    backend = "ep", inference_tier = 2L, inference_mode = "structured",
    draws_kind = "iid"
  )
  class(fit) <- c("tulpa_ep", "tulpa_fit")
  fit
}

#' @export
vcov.tulpa_ep <- function(object, ...) object$vcov

#' @export
coef.tulpa_ep <- function(object, ...) object$coefficients
