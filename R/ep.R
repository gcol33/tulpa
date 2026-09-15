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

# Mode and curvature of the tilted log-density h(eta) = -(eta - mu)^2 / (2 s2)
# + loglik(eta; y) in eta, for laying adaptive Gauss-Hermite nodes at the
# tilted distribution rather than at the cavity. `stats::optimize()` (a
# derivative-free, bracketed golden-section search) is used rather than a hand
# Newton loop: the observed curvature the engine's own `obs_weight` registers
# is EXPECTED (Fisher) information for some families (inverse_gaussian has no
# registered `obs_weight`, so `.family_obs_weight()` falls back to it), so a
# Newton step built from it is only approximately right -- fine for placing
# nodes (the importance-reweighting below corrects for a mis-scaled proposal),
# but not something to trust for a step-size-sensitive root find. The bracket
# widens (up to 3 times) when the optimum lands on its edge, which happens
# when the likelihood pulls the tilted mode well outside the cavity's own
# range (e.g. a large Poisson count against a wide cavity).
.ep_site_mode <- function(mu, s2, y, family, phi, n_trials, phi2) {
  h <- function(eta) {
    -(eta - mu)^2 / (2 * s2) +
      family_loglik(eta, y, family, n_trials = n_trials, phi = phi, phi2 = phi2)
  }
  half_width <- 10 * sqrt(s2)
  converged  <- FALSE
  eta_hat    <- mu
  for (k in seq_len(3L)) {
    lo <- mu - half_width; hi <- mu + half_width
    opt <- stats::optimize(h, interval = c(lo, hi), maximum = TRUE,
                           tol = .Machine$double.eps^0.4)
    eta_hat <- opt$maximum
    at_edge <- (eta_hat - lo) < 1e-6 * half_width ||
      (hi - eta_hat) < 1e-6 * half_width
    if (!at_edge) { converged <- TRUE; break }
    half_width <- half_width * 4
  }
  prec <- 1 / s2 + as.numeric(.family_obs_weight(eta_hat, y, family,
                                                 n_trials, phi, phi2))
  if (!is.finite(prec) || prec <= 0) { prec <- 1 / s2; eta_hat <- mu }
  list(eta_hat = eta_hat, prec = prec, converged = converged)
}

# Moments of the tilted distribution N(eta; mu, s2) * exp(loglik(eta; y)) over a
# single site, by ADAPTIVE Gauss-Hermite quadrature: the nodes are laid at the
# tilted distribution's own mode and curvature (`.ep_site_mode()`) rather than
# at the cavity, and reweighted by the density ratio against that Gaussian
# proposal (self-normalized importance sampling on a Gauss-Hermite grid) -- so
# the result is exact for any proposal placement, including the non-adaptive
# cavity-centred one this replaces (`log_cav - log_prop` cancels exactly there,
# reproducing the old rule bit for bit). A likelihood far narrower than the
# cavity (a large count) or with a flat stretch in eta (inverse_gaussian, log
# link) starves the OLD rule's fixed cavity-centred nodes of mass; recentring
# on the tilted mode is what the fix direction calls for.
#
# `mu` is the FULL linear predictor mean (cavity mean plus any offset);
# returns Z (normalizer) and the tilted mean/variance of that same full linear
# predictor, plus `site_converged` (the mode search did not run into its
# bracket) and `floored` (the returned variance hit the numerical floor,
# i.e. the site is not well resolved by the quadrature). `gh` is the output of
# .gauss_hermite().
.ep_tilted_moments <- function(mu, s2, y, family, phi, n_trials, gh,
                               phi2 = NULL) {
  # Gaussian likelihood: the tilted distribution is exactly Gaussian, so use its
  # closed-form moments (EP is then exact, no quadrature error). phi = variance.
  # The tilted normalizer is the Gaussian convolution N(y; mu, s2 + phi).
  if (family == "gaussian") {
    v  <- 1 / (1 / s2 + 1 / phi)
    m1 <- v * (mu / s2 + y / phi)
    return(list(logZ = stats::dnorm(y, mu, sqrt(s2 + phi), log = TRUE),
                mean = m1, var = max(v, 1e-10),
                site_converged = TRUE, floored = FALSE))
  }
  sm <- .ep_site_mode(mu, s2, y, family, phi, n_trials, phi2)
  eta_hat <- sm$eta_hat
  sd_hat  <- 1 / sqrt(sm$prec)

  eta <- eta_hat + sqrt(2) * sd_hat * gh$x        # change of variable
  wt  <- gh$w / sqrt(pi)
  ll  <- family_loglik(eta, rep(y, length(eta)), family,
                       n_trials = rep(n_trials, length(eta)), phi = phi,
                       phi2 = phi2)
  log_cav  <- stats::dnorm(eta, mu, sqrt(s2), log = TRUE)
  log_prop <- stats::dnorm(eta, eta_hat, sd_hat, log = TRUE)
  lw  <- log(wt) + log_cav + ll - log_prop
  m   <- max(lw)
  ew  <- exp(lw - m)
  Z0  <- sum(ew)
  if (!is.finite(Z0) || Z0 <= 0) return(NULL)
  p   <- ew / Z0
  m1  <- sum(p * eta)
  m2  <- sum(p * eta^2)
  v   <- m2 - m1^2
  list(logZ = log(Z0) + m, mean = m1, var = max(v, 1e-10),
       site_converged = sm$converged, floored = v <= 1e-10)
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
#' @template phi
#' @param phi2 Optional second dispersion (Student-t degrees of freedom for
#'   `family = "t"`; default 4 when `NULL`).
#' @param n_trials Binomial denominators (length `nrow(data)`), or `NULL` (= 1).
#' @param beta_prior Fixed-effect prior as `list(mean, sd)`: a mean-zero
#'   (`mean = 0`) Gaussian on every coefficient with SD `sd` (default
#'   the engine default, `prior_normal(0, 2.5)`). EP's site parameterisation assumes a mean-zero
#'   coefficient prior, so a non-zero `mean` errors -- use a sampler
#'   (`mode = "mala"`) for a shifted prior.
#' @param control List: `max_sweeps` (default 50), `tol` (default 1e-6),
#'   `damping` (default 0.8), `n_quad` (Gauss-Hermite nodes, default 20),
#'   `n_draws` (default 2000), `seed`.
#'
#' An `offset(...)` term in `formula` is honoured: it shifts each
#' observation's linear predictor before the likelihood is evaluated, and is
#' not itself part of the fitted posterior.
#'
#' @return A `tulpa_fit` (subclass `tulpa_ep`) with `means` (posterior mean) and
#'   `cov` (posterior covariance) of the EP Gaussian, which is the posterior
#'   [coef()], [vcov()], [summary()] and [confint()] report; `draws` sampled
#'   from that Gaussian; `log_marginal` (the EP approximation); `converged`
#'   (the sweep tolerance was met AND every site's adaptive quadrature mode
#'   search converged AND no site's tilted variance floored -- a site that is
#'   not well resolved by the quadrature turns this `FALSE` even when the
#'   outer sweep loop itself settled); `n_site_not_converged`,
#'   `n_site_floored`.
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
                     beta_prior = .tulpa_default_beta_prior("ep"),
                     control = list()) {
  tulpa_check_control(control, .CONTROL_KEYS$ep, "tulpa_ep")
  mf <- stats::model.frame(formula, data)
  y  <- as.numeric(stats::model.response(mf))
  X  <- stats::model.matrix(stats::terms(mf), mf)
  off <- stats::model.offset(mf)
  fit <- ep_fit(y = y, X = X, family = family, phi = phi, phi2 = phi2,
                n_trials = n_trials, beta_prior = beta_prior,
                offset = off, control = control)
  fit$formula <- formula
  fit
}

# EP engine over a design bundle (y, X). The exported tulpa_ep() parses a
# formula and calls this; the registry `ep` backend dispatches here directly.
#' @keywords internal
ep_fit <- function(y, X, family = "binomial", phi = 1.0, phi2 = NULL,
                   n_trials = NULL, beta_prior = .tulpa_default_beta_prior("ep"),
                   offset = NULL, control = list()) {
  tulpa_check_control(control, .CONTROL_KEYS$ep, "tulpa_ep")
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
  beta_prior_sd <- .beta_prior_ridge_sd(beta_prior, .tulpa_prior_sd("ep"))
  nt <- if (is.null(n_trials)) rep(1L, n) else as.integer(n_trials)
  off <- if (is.null(offset)) rep(0, n) else as.numeric(offset)
  if (length(off) != n) {
    stop(sprintf("length(offset) (%d) must equal nrow(X) (%d).",
                 length(off), n), call. = FALSE)
  }
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
      # tau[i]/nu[i] are site natural parameters on eta' = x_i'beta (the offset
      # is not part of the posterior being approximated), so the tilted moments
      # -- taken on the FULL linear predictor eta' + offset[i], where the
      # likelihood actually lives -- are shifted back by offset[i] before
      # feeding the site update.
      tm <- .ep_tilted_moments(mu_cav + off[i], s2_cav, y[i], family, phi,
                               nt[i], gh, phi2 = phi2)
      if (is.null(tm)) next
      tau_new <- 1 / tm$var - inv_cav
      nu_new  <- (tm$mean - off[i]) / tm$var - mu_cav / s2_cav
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
  # Site quality at the converged (or max-sweeps) state: recorded here because
  # this pass already recomputes every site's tilted moments at the final
  # cavities. A site whose mode search ran into its bracket, or whose
  # variance floored, means EP's Gaussian approximation at that observation is
  # not resolved by the quadrature -- `converged` below is downgraded rather
  # than reported TRUE over it (gcol33/tulpa#765).
  site_ok      <- rep(TRUE, n)
  site_floored <- rep(FALSE, n)
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
      tm  <- .ep_tilted_moments(muc + off[i], s2c, y[i], family, phi, nt[i], gh,
                                phi2 = phi2)
      if (is.null(tm) || !is.finite(tm$logZ)) return(NA_real_)
      site_ok[i]      <<- isTRUE(tm$site_converged)
      site_floored[i] <<- isTRUE(tm$floored)
      ci <- tm$logZ + 0.5 * log1p(tau[i] * s2c) -
        (muc / s2c + nu[i])^2 * s2 / 2 + muc^2 / (2 * s2c)
      if (!is.finite(ci)) return(NA_real_)
      csum <- csum + ci
    }
    lg + csum
  })
  n_site_not_converged <- sum(!site_ok)
  n_site_floored       <- sum(site_floored)
  converged <- converged && n_site_not_converged == 0L && n_site_floored == 0L

  .seed_scoped(control$seed)
  draws <- .ps_rmvnorm(n_draws, m, V)

  # The EP Gaussian N(m, V) is the posterior this fit reports; the draws are
  # samples from it for the draw-consuming accessors.
  fit <- list(
    means = m, cov = V, reported_posterior = "gaussian", draws = draws,
    param_names = pn, n_fixed = p, fixed_names = pn,
    log_marginal = log_marginal, converged = converged, n_sweeps = sweep,
    n_site_not_converged = n_site_not_converged, n_site_floored = n_site_floored,
    family = family, model_matrix = X,
    backend = "ep", inference_tier = 2L, inference_mode = "structured",
    draws_kind = "iid"
  )
  class(fit) <- c("tulpa_ep", "tulpa_fit")
  fit
}
