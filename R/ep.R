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
.ep_tilted_moments <- function(mu, s2, y, family, phi, n_trials, gh,
                               phi2 = NULL) {
  # Gaussian likelihood: the tilted distribution is exactly Gaussian, so use its
  # closed-form moments (EP is then exact, no quadrature error). phi = variance.
  # The tilted normalizer is the Gaussian convolution N(y; mu, s2 + phi).
  if (family == "gaussian") {
    v  <- 1 / (1 / s2 + 1 / phi)
    m1 <- v * (mu / s2 + y / phi)
    return(list(logZ = stats::dnorm(y, mu, sqrt(s2 + phi), log = TRUE),
                mean = m1, var = max(v, 1e-10)))
  }
  sd <- sqrt(s2)
  eta <- mu + sqrt(2) * sd * gh$x                 # change of variable
  wt  <- gh$w / sqrt(pi)
  ll  <- family_loglik(eta, rep(y, length(eta)), family,
                       n_trials = rep(n_trials, length(eta)), phi = phi,
                       phi2 = phi2)
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
#' @param phi2 Optional second dispersion (Student-t degrees of freedom for
#'   `family = "t"`; default 4 when `NULL`).
#' @param n_trials Binomial denominators (length `nrow(data)`), or `NULL` (= 1).
#' @param beta_prior Fixed-effect prior as `list(mean, sd)`: a mean-zero
#'   (`mean = 0`) Gaussian on every coefficient with SD `sd` (default
#'   `list(mean = 0, sd = 10)`). EP's site parameterisation assumes a mean-zero
#'   coefficient prior, so a non-zero `mean` errors -- use a sampler
#'   (`mode = "mala"`) for a shifted prior.
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
                     phi2 = NULL, n_trials = NULL,
                     beta_prior = list(mean = 0, sd = 10),
                     control = list()) {
  .check_control(control, .CONTROL_KEYS$ep, "tulpa_ep")
  mf <- stats::model.frame(formula, data)
  y  <- as.numeric(stats::model.response(mf))
  X  <- stats::model.matrix(stats::terms(mf), mf)
  fit <- ep_fit(y = y, X = X, family = family, phi = phi, phi2 = phi2,
                n_trials = n_trials, beta_prior = beta_prior, control = control)
  fit$formula <- formula
  fit
}

# Resolve a `beta_prior = list(mean, sd)` to the mean-zero SD EP consumes.
# EP's site parameterisation is built for a mean-zero coefficient prior, so a
# non-zero mean is rejected rather than silently centred.
#' @keywords internal
.ep_beta_prior_sd <- function(beta_prior, p) {
  bp <- beta_prior %||% list(mean = 0, sd = 10)
  m0 <- bp$mean %||% 0
  sd <- bp$sd %||% 10
  if (any(m0 != 0)) {
    stop("tulpa_ep() supports only a mean-zero fixed-effect prior ",
         "(`beta_prior$mean` must be 0); use a sampler (mode = 'mala') for a ",
         "shifted prior.", call. = FALSE)
  }
  if (!is.numeric(sd) || any(!is.finite(sd)) || any(sd <= 0) ||
      !(length(sd) == 1L || length(sd) == p)) {
    stop("`beta_prior$sd` must be a positive scalar or length-p vector.",
         call. = FALSE)
  }
  sd
}

# EP engine over a design bundle (y, X). The exported tulpa_ep() parses a
# formula and calls this; the registry `ep` backend dispatches here directly.
#' @keywords internal
ep_fit <- function(y, X, family = "binomial", phi = 1.0, phi2 = NULL,
                   n_trials = NULL, beta_prior = list(mean = 0, sd = 10),
                   control = list()) {
  .check_control(control, .CONTROL_KEYS$ep, "tulpa_ep")
  .family_or_stop(family)
  if (!is.null(phi2)) .phi2_or_stop(family, phi2)
  max_sweeps <- as.integer(control$max_sweeps %||% 50L)
  tol        <- control$tol %||% 1e-6
  damping    <- control$damping %||% 0.8
  n_quad     <- as.integer(control$n_quad %||% 20L)
  n_draws    <- as.integer(control$n_draws %||% 2000L)

  y  <- as.numeric(y)
  X  <- as.matrix(X)
  n  <- nrow(X); p <- ncol(X)
  beta_prior_sd <- .ep_beta_prior_sd(beta_prior, p)
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
      tm <- .ep_tilted_moments(mu_cav, s2_cav, y[i], family, phi, nt[i], gh,
                               phi2 = phi2)
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

  # EP approximate marginal likelihood (R&W eq. 3.65 in the GLM site
  # parameterization): Z_EP = int N(beta; 0, S0) prod_i ttilde_i(x_i' beta),
  # with each site ttilde_i(eta) = exp(-tau_i eta^2/2 + nu_i eta + C_i). The
  # Gaussian integral gives
  #   log G = -1/2 log|S0| - 1/2 log|P| + 1/2 m' X' nu,   P = S0^-1 + X'TX,
  # and each site constant C_i is fixed by matching the tilted normalizer
  # Zhat_i = int q_cav(eta) p(y_i | eta) deta against the same integral of the
  # site Gaussian, evaluated at the converged cavities:
  #   C_i = log Zhat_i + 1/2 log(1 + tau_i s2_cav)
  #         - (mu_cav / s2_cav + nu_i)^2 / (2 / s2_marg) + mu_cav^2 / (2 s2_cav)
  # where 1/s2_marg = 1/s2_cav + tau_i. Exact for the gaussian family (equals
  # the closed-form evidence of the conjugate linear model). NA when any cavity
  # or tilted normalizer is unavailable -- never a partial sum.
  log_marginal <- local({
    b  <- as.numeric(crossprod(X, nu))
    lg <- -p * log(beta_prior_sd) - sum(log(diag(chol(st$P)))) +
      0.5 * sum(st$m * b)
    csum <- 0
    for (i in seq_len(n)) {
      xi <- X[i, ]
      s2 <- as.numeric(xi %*% (st$V %*% xi))
      mu <- as.numeric(xi %*% st$m)
      inv_cav <- 1 / s2 - tau[i]
      if (inv_cav <= 1e-10) return(NA_real_)
      s2c <- 1 / inv_cav
      muc <- s2c * (mu / s2 - nu[i])
      tm  <- .ep_tilted_moments(muc, s2c, y[i], family, phi, nt[i], gh,
                                phi2 = phi2)
      if (is.null(tm) || !is.finite(tm$logZ)) return(NA_real_)
      ci <- tm$logZ + 0.5 * log1p(tau[i] * s2c) -
        (muc / s2c + nu[i])^2 * s2 / 2 + muc^2 / (2 * s2c)
      if (!is.finite(ci)) return(NA_real_)
      csum <- csum + ci
    }
    lg + csum
  })

  .seed_scoped(control$seed)
  draws <- .ps_rmvnorm(n_draws, m, V)

  fit <- list(
    coefficients = m, vcov = V, draws = draws, means = m, param_names = pn,
    n_fixed = p, fixed_names = pn,
    log_marginal = log_marginal, converged = converged, n_sweeps = sweep,
    family = family, model_matrix = X,
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
