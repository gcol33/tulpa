# Nested-Laplace integration over a random-effect covariance Sigma.
#
# Bias-2 fix (the `Marginalize Derived Quantities` principle): rather than
# report Sigma at its mode (the plug-in MAP, biased low for skewed
# variance-component marginals at small G), integrate the Laplace marginal
# likelihood over a Sigma-grid and report weighted quantiles of Sigma and its
# derived scale / correlation parameters.
#
# Composition: the inner solve at each grid point is the correlated-RE Laplace
# fit (tulpa_laplace, which returns the Laplace log-marginal at a SUPPLIED
# Sigma); the grid is the nested_laplace + CCD recipe (here a rotated tensor
# grid centred at the marginal-likelihood mode); the summary reuses the same
# weighted-quantile machinery as the spatial/temporal nested-Laplace surface.

# log-Cholesky <-> matrix helpers ---------------------------------------------
# theta packs the lower Cholesky factor L (Sigma = L L') in column-major
# lower-triangular order: diagonal entries as log(L_ii) (positivity), strictly
# lower entries as raw L_ij. Length c(c+1)/2. Keeps Sigma positive definite for
# every theta in R^{c(c+1)/2}.
.re_logchol_to_L <- function(theta, c) {
  L <- matrix(0, c, c)
  idx <- 1L
  for (j in seq_len(c)) {
    for (i in j:c) {
      L[i, j] <- if (i == j) exp(theta[idx]) else theta[idx]
      idx <- idx + 1L
    }
  }
  L
}

.re_L_to_logchol <- function(L, c) {
  theta <- numeric(c * (c + 1L) / 2L)
  idx <- 1L
  for (j in seq_len(c)) {
    for (i in j:c) {
      theta[idx] <- if (i == j) log(max(L[i, j], 1e-8)) else L[i, j]
      idx <- idx + 1L
    }
  }
  theta
}

# Marginalized summary of a random-effect covariance from a set of Sigma
# matrices and weights. Shared by the grid integrator (tulpa_re_cov_nested,
# weighted grid cells) and the Gibbs sampler (tulpa_re_cov_gibbs, equal-weight
# posterior draws). `Marginalize Derived Quantities`: each derived value
# (sigma_i = sqrt(Sigma_ii), rho_ij = Sigma_ij / (sigma_i sigma_j), and the raw
# Sigma entries) is computed PER matrix, then weighted-quantiled -- never a
# transform of summarized components. With equal weights `.nl_wtd_quantile`
# reduces to the sample quantile, so the same code summarizes both.
.re_cov_derived_summary <- function(Sig_list, w, c_re) {
  derived <- list()
  for (i in seq_len(c_re)) {
    derived[[sprintf("sigma_%d", i)]] <-
      vapply(Sig_list, function(S) sqrt(S[i, i]), numeric(1))
  }
  if (c_re > 1L) {
    for (i in seq_len(c_re - 1L)) for (j in (i + 1L):c_re) {
      derived[[sprintf("rho_%d%d", i, j)]] <-
        vapply(Sig_list, function(S) S[i, j] / sqrt(S[i, i] * S[j, j]),
               numeric(1))
    }
  }
  for (i in seq_len(c_re)) for (j in i:c_re) {
    derived[[sprintf("Sigma_%d%d", i, j)]] <-
      vapply(Sig_list, function(S) S[i, j], numeric(1))
  }

  summarize <- function(x) {
    m   <- sum(w * x)
    sdv <- sqrt(max(0, sum(w * x^2) - m^2))
    q   <- .nl_wtd_quantile(x, w, c(0.025, 0.5, 0.975))
    c(mean = m, sd = sdv, median = q[2L], ci_lo = q[1L], ci_hi = q[3L])
  }
  post <- do.call(rbind, lapply(derived, summarize))
  posterior <- data.frame(parameter = rownames(post), post,
                          row.names = NULL, check.names = FALSE)
  Sigma_mean <- Reduce(`+`, Map(function(S, wi) S * wi, Sig_list, w))
  list(posterior = posterior, Sigma_mean = Sigma_mean)
}

#' Nested-Laplace integration over a random-effect covariance
#'
#' @description
#' For a single correlated random-effects term (e.g. `(1 + x | g)`), integrate
#' the Laplace marginal likelihood over the random-effect covariance `Sigma`
#' instead of fixing it at a point estimate. Reports weighted posterior
#' summaries (mean, SD, median, 2.5\%/97.5\%) of `Sigma` and its derived scale
#' (`sigma_i`) and correlation (`rho_ij`) parameters, marginalizing the joint
#' posterior over a `Sigma`-grid.
#'
#' This corrects the plug-in-MAP ("summary") bias: the mode of a skewed
#' variance-component marginal is biased low relative to its median, so the
#' headline summary should be the marginalized median, not the mode.
#'
#' @details
#' `Sigma` is parameterized by its lower Cholesky factor `L` (`Sigma = L L'`)
#' in log-Cholesky coordinates `theta`: the log-diagonal and the strictly-lower
#' entries of `L` (`c(c+1)/2` values for a `c`-coefficient term), which keeps
#' `Sigma` positive definite for every `theta`. The integration grid is a
#' tensor grid in whitened `theta`-space, centred at the marginal-likelihood
#' mode `theta_hat` and rotated/scaled by the Cholesky of the mode's posterior
#' covariance (`solve(Hessian)`), so points track the posterior ridge. Each
#' grid cell `k` contributes weight proportional to
#' `exp(log_marginal(Sigma_k) + log_prior_theta(theta_k))`.
#'
#' The default `log_prior_theta` is flat in `theta` (improper, but the finite
#' grid truncates the tails). Supply a function for an informative prior; note
#' it acts in the log-Cholesky parameterization described above.
#'
#' @param y,n_trials,X,family,phi Passed to [tulpa_laplace()] for the inner
#'   solve. `n_trials = NULL` defaults to 1 (binary / single-trial).
#' @param re_term A single random-effect term: a list with `idx` (1-based group
#'   index per observation), `n_groups`, `n_coefs` (`c`), and `Z` (the
#'   `n_obs x c` RE design, e.g. `cbind(1, x)` for `(1 + x | g)`). Any `L` /
#'   `cov` / `sigma` field is ignored -- `Sigma` is what this function
#'   integrates over.
#' @param n_per_axis Points per `theta` axis in the tensor grid (default 5).
#'   Grid size is `n_per_axis^(c(c+1)/2)`; each cell is one Laplace fit.
#' @param span Half-width of the grid in posterior standard deviations per
#'   whitened axis (default 3).
#' @param log_prior_theta Optional `function(theta)` returning a scalar log
#'   prior density in the log-Cholesky parameterization. Default flat (`0`).
#' @param max_iter,tol,n_threads Inner-solve controls (see [tulpa_laplace()]).
#'
#' @return A list with:
#'   - `posterior`: data frame with one row per parameter (`sigma_i`, `rho_ij`,
#'     `Sigma_ij`) and columns `mean`, `sd`, `median`, `ci_lo`, `ci_hi` -- the
#'     marginalized summaries.
#'   - `map`: the plug-in-mode summary (`Sigma`, `sigma`, `rho`) at `theta_hat`,
#'     for comparison with the marginalized `posterior`.
#'   - `Sigma_mean`: the weighted posterior mean of `Sigma` (a `c x c` matrix).
#'   - `theta_hat`, `theta_grid`, `weights`, `log_marginal`, `n_grid`, `n_coefs`.
#'
#' @seealso [tulpa_laplace()] for the inner solve; [tulpa_nested_laplace()] for
#'   the analogous outer integration over spatial / temporal prior
#'   hyperparameters.
#'
#' @export
tulpa_re_cov_nested <- function(y, n_trials = NULL, X, re_term,
                                family = "binomial", phi = 1.0,
                                n_per_axis = 5L, span = 3,
                                log_prior_theta = NULL,
                                max_iter = 100L, tol = 1e-8, n_threads = 1L) {

  if (is.null(re_term$n_coefs)) re_term$n_coefs <- 1L
  c_re <- as.integer(re_term$n_coefs)
  if (c_re < 1L) stop("`re_term$n_coefs` must be >= 1.", call. = FALSE)
  if (is.null(re_term$idx) || is.null(re_term$n_groups)) {
    stop("`re_term` must supply `idx` and `n_groups`.", call. = FALSE)
  }
  if (c_re > 1L && is.null(re_term$Z)) {
    stop("`re_term$Z` (the n_obs x n_coefs RE design) is required when ",
         "n_coefs > 1.", call. = FALSE)
  }
  if (is.null(n_trials)) n_trials <- rep(1L, length(y))
  if (is.null(log_prior_theta)) log_prior_theta <- function(theta) 0
  k <- c_re * (c_re + 1L) / 2L

  # Inner solve: Laplace log-marginal at Sigma = L L'. Failures at extreme grid
  # edges (non-finite / non-convergent) return -Inf so the cell gets zero
  # weight rather than aborting the integration.
  inner_logmarg <- function(L) {
    rt <- re_term
    rt$L <- L; rt$cov <- NULL; rt$sigma <- NULL
    val <- tryCatch(
      tulpa_laplace(
        y = y, n_trials = n_trials, X = X, re_list = list(rt),
        family = family, phi = phi, return_hessian = FALSE,
        max_iter = max_iter, tol = tol, n_threads = n_threads
      )$log_marginal,
      error = function(e) -Inf
    )
    if (length(val) != 1L || !is.finite(val)) -Inf else val
  }

  # --- pilot init: method-of-moments Sigma from a Sigma = I fit -------------
  L0 <- diag(c_re)
  pilot <- tryCatch(
    tulpa_laplace(
      y = y, n_trials = n_trials, X = X,
      re_list = list(utils::modifyList(re_term, list(L = L0, cov = NULL, sigma = NULL))),
      family = family, phi = phi, return_hessian = FALSE,
      max_iter = max_iter, tol = tol, n_threads = n_threads
    ),
    error = function(e) NULL
  )
  Sigma_init <- diag(c_re)
  if (!is.null(pilot) && !is.null(pilot$mode)) {
    p_fix <- ncol(X)
    re_vals <- pilot$mode[-seq_len(p_fix)]
    if (length(re_vals) == c_re * re_term$n_groups) {
      U <- matrix(re_vals, ncol = c_re, byrow = TRUE)   # n_groups x c
      S <- stats::cov(U)
      # clamp to PD with a floor on the variances
      if (all(is.finite(S))) {
        diag(S) <- pmax(diag(S), 1e-3)
        ev <- eigen(S, symmetric = TRUE, only.values = TRUE)$values
        if (min(ev) > 1e-8) Sigma_init <- S
      }
    }
  }
  theta0 <- .re_L_to_logchol(t(chol(Sigma_init)), c_re)

  # --- mode of g(theta) = log_marginal(Sigma(theta)) + log_prior ------------
  negg <- function(theta) {
    L <- .re_logchol_to_L(theta, c_re)
    -(inner_logmarg(L) + log_prior_theta(theta))
  }
  opt <- stats::optim(theta0, negg, method = "Nelder-Mead", hessian = TRUE,
                      control = list(maxit = 500L, reltol = 1e-8))
  theta_hat <- opt$par

  # Posterior covariance of theta ~ solve(Hessian of negg). Regularize to PD;
  # fall back to a diagonal scale if the numerical Hessian is unusable.
  Hn <- opt$hessian
  post_cov <- tryCatch({
    Hs <- (Hn + t(Hn)) / 2
    ev <- eigen(Hs, symmetric = TRUE, only.values = TRUE)$values
    if (min(ev) <= 1e-8) stop("non-PD Hessian")
    solve(Hs)
  }, error = function(e) diag(0.5^2, k))
  L_scale <- t(chol(post_cov))

  # --- rotated tensor grid in whitened theta-space --------------------------
  ax <- seq(-span, span, length.out = as.integer(n_per_axis))
  z <- as.matrix(expand.grid(rep(list(ax), k)))
  dimnames(z) <- NULL
  theta_grid <- ccd_to_theta(z, theta_hat, L_scale)   # n_grid x k
  ng <- nrow(theta_grid)

  # --- evaluate inner marginal + derived quantities per cell ----------------
  logm <- numeric(ng)
  Sig_entries <- vector("list", ng)
  for (i in seq_len(ng)) {
    th <- theta_grid[i, ]
    L  <- .re_logchol_to_L(th, c_re)
    logm[i] <- inner_logmarg(L) + log_prior_theta(th)
    Sig_entries[[i]] <- L %*% t(L)
  }
  w <- .nl_normalise_weights(logm)

  # --- derived quantities, marginalized over the grid -----------------------
  # Each derived value is computed PER cell and weighted-quantiled (never a
  # transform of summarized components); see `.re_cov_derived_summary`.
  summ <- .re_cov_derived_summary(Sig_entries, w, c_re)
  posterior <- summ$posterior

  # --- plug-in MAP summary (for comparison) and weighted-mean Sigma ---------
  L_hat <- .re_logchol_to_L(theta_hat, c_re)
  Sigma_hat <- L_hat %*% t(L_hat)
  sig_hat <- sqrt(diag(Sigma_hat))
  rho_hat <- if (c_re > 1L) {
    R <- Sigma_hat / outer(sig_hat, sig_hat); R[upper.tri(R)]
  } else numeric(0)

  list(
    posterior   = posterior,
    map         = list(Sigma = Sigma_hat, sigma = sig_hat, rho = rho_hat),
    Sigma_mean  = summ$Sigma_mean,
    theta_hat   = theta_hat,
    theta_grid  = theta_grid,
    weights     = w,
    log_marginal = logm,
    n_grid      = ng,
    n_coefs     = c_re
  )
}
