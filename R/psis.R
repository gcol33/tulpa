# =============================================================================
# psis.R -- Pareto-smoothed importance sampling (PSIS) and the Pareto-k-hat
# diagnostic, following Vehtari, Simpson, Gelman, Yao & Gabry (2024),
# "Pareto smoothed importance sampling", JMLR 25(72):1-58, and the
# generalized-Pareto fit of Zhang & Stephens (2009), Technometrics 51(3).
#
# Native implementation -- no `loo` / `posterior` dependency. The shape
# estimate `pareto_k` reproduces `loo::psis()` on a fixed log-ratio vector
# (tests/testthat/test-psis.R). PSIS is the approximation-accuracy gate for
# the nested-Laplace path: the inner-Laplace marginal posterior of the latent
# block hyperparameters is importance-sampled against the Gaussian proposal the
# nested integrator places its grid with, and k-hat measures whether that
# proposal can be corrected to the target. k-hat < 0.7 => reliable;
# k-hat >= 0.7 => the (skewed / heavy-tailed) hyperparameter posterior is
# misfit by the Gaussian grid and the nested result is not trustworthy.
# =============================================================================

# Minimum number of finite importance draws for a usable outer Pareto-k-hat.
# The GPD tail fit (Zhang-Stephens) below this is too noisy to report, so a
# diagnostic with fewer finite evaluations declines to NA. A `n_samples` budget
# under this floor can never clear it, so the IS cores short-circuit BEFORE
# paying any inner solve rather than evaluate the whole budget and discard it
# (gcol33/tulpa#51).
.PSIS_MIN_EVAL <- 25L

# Whitened-radius envelope for the outer Pareto-k importance draws
# (gcol33/tulpa#94). The outer integrator placed its grid to cover the
# hyperparameter posterior; a proposal draw far outside that node cloud is a
# deep extrapolation the integration never represented AND -- at EVA scale --
# exactly where the inner Newton stalls toward max_iter, so those draws
# dominate the diagnostic's wall time (~50x the grid integration) while
# carrying negligible target mass. They sit in the LEFT tail of the importance
# ratio (target ~ 0), immaterial to the right-tail GPD fit the k-hat reads, so
# excluding them bounds the diagnostic cost without moving the k-hat. The cap is
# `.K_DIAG_SUPPORT_RADIUS_MULT` times the largest whitened grid-node radius:
# a generous envelope (twice the covered radius) that retains essentially every
# draw in the well-behaved case -- where the proposal mass sits within the grid
# -- and skips only the deep-extrapolation stalls. Scope: the k-hat then
# certifies the Gaussian-proposal-over-grid approximation the draws are actually
# synthesised from; grid-width adequacy (target mass beyond the grid) is a
# separate concern.
.K_DIAG_SUPPORT_RADIUS_MULT <- 2.0

# Compute the whitened-radius cap from the integration grid. `u_grid` is the
# K x d grid in the proposal's own (unconstrained) coordinate, `u_hat` the
# proposal centre, `L` the lower-Cholesky factor of the proposal covariance
# (L L' = Sigma). Each grid node's whitened distance from u_hat is
# || L^{-1} (u_grid[k] - u_hat) ||, in the SAME metric as a draw's whitened
# radius ||z|| (the proposal draws U = u_hat + L z, so z = L^{-1}(U - u_hat)).
# Returns Inf (keep every draw) on a degenerate solve.
.nested_grid_radius_cap <- function(u_grid, u_hat, L) {
  cen <- sweep(u_grid, 2L, u_hat)
  w   <- tryCatch(forwardsolve(L, t(cen)), error = function(e) NULL)
  if (is.null(w)) return(Inf)
  r2  <- colSums(w * w)
  r2  <- r2[is.finite(r2)]
  if (length(r2) == 0L) return(Inf)
  sqrt(max(r2)) * .K_DIAG_SUPPORT_RADIUS_MULT
}

# Log-sum-exp, numerically stabilized.
.tulpa_logsumexp <- function(x) {
  m <- max(x)
  if (!is.finite(m)) return(m)
  m + log(sum(exp(x - m)))
}

# Quantile function of the generalized Pareto distribution (location 0):
# Q(p) = sigma/k * ((1 - p)^(-k) - 1), with the k -> 0 exponential limit.
.tulpa_qgpd <- function(p, k, sigma) {
  if (is.nan(sigma) || sigma <= 0) return(rep(NaN, length(p)))
  if (abs(k) < 1e-30) return(sigma * (-log1p(-p)))
  sigma * expm1(-k * log1p(-p)) / k
}

# Generalized-Pareto fit on positive exceedances `x`, by the Zhang & Stephens
# (2009) empirical-Bayes profile over the scale parameter, with the
# weakly-informative prior of Vehtari et al. (2024) shrinking the shape toward
# 1/2 (pseudo-count 10). Matches `loo:::gpdfit()`.
.tulpa_gpd_fit <- function(x, wip = TRUE, min_grid_pts = 30L) {
  x <- sort.int(x)
  N <- length(x)
  prior <- 3
  M <- min_grid_pts + floor(sqrt(N))
  jj <- seq_len(M)
  xstar <- x[floor(N / 4 + 0.5)]                       # 25% quantile
  theta <- 1 / x[N] + (1 - sqrt(M / (jj - 0.5))) / (prior * xstar)
  k_theta <- vapply(theta, function(t) mean(log1p(-t * x)), numeric(1))
  l_theta <- N * (log(-theta / k_theta) - k_theta - 1) # profile log-likelihood
  w_theta <- exp(l_theta - .tulpa_logsumexp(l_theta))  # softmax over the grid
  theta_hat <- sum(theta * w_theta)
  k <- mean(log1p(-theta_hat * x))                     # shape (heavy tail => k > 0)
  sigma <- -k / theta_hat
  if (wip) k <- (k * N + 0.5 * 10) / (N + 10)          # shrink toward 1/2
  if (is.na(k)) k <- Inf
  list(k = k, sigma = sigma)
}

#' Pareto-smoothed importance sampling
#'
#' Smooths a set of importance ratios by replacing the largest weights with the
#' order statistics of a generalized-Pareto fit to their upper tail, and
#' returns the Pareto shape diagnostic `pareto_k` together with the smoothed
#' (normalized) log weights and their importance-sampling effective sample
#' size. `pareto_k` estimates the number of finite moments of the raw weight
#' distribution: `< 0.5` is good, `0.5-0.7` acceptable, `>= 0.7` indicates the
#' proposal cannot be reliably corrected to the target.
#'
#' @param log_ratios Numeric vector of (unnormalized) log importance ratios
#'   `log p_target(x) - log q_proposal(x)` evaluated at draws `x ~ q`.
#' @return A list with `pareto_k` (the tail shape, `NA` if the sample is too
#'   small to fit), `is_ess` (importance-sampling effective sample size,
#'   `1 / sum(w^2)` on the normalized smoothed weights), and `log_weights`
#'   (the normalized smoothed log weights).
#' @references Vehtari, Simpson, Gelman, Yao & Gabry (2024). Pareto smoothed
#'   importance sampling. \emph{JMLR} 25(72):1-58.
#' @seealso [mcmc_diagnostics()] for the MCMC-chain counterpart.
#' @export
tulpa_psis <- function(log_ratios) {
  log_ratios <- log_ratios[is.finite(log_ratios)]
  S <- length(log_ratios)
  if (S < 5L) {
    return(list(pareto_k = NA_real_, is_ess = NA_real_, log_weights = numeric(0)))
  }

  lw <- log_ratios - max(log_ratios)                   # stabilize
  tail_len <- as.integer(ceiling(min(0.2 * S, 3 * sqrt(S))))
  k_hat <- NA_real_

  if (tail_len >= 5L && S >= 25L) {
    ord     <- order(lw)
    cut_idx <- S - tail_len
    cutoff  <- lw[ord[cut_idx]]                         # largest non-tail log weight
    tail_ord <- ord[(cut_idx + 1L):S]                  # tail indices, ascending lw
    exceed  <- exp(lw[tail_ord]) - exp(cutoff)         # exceedances over the threshold
    if (sum(exceed > 0) >= 5L) {
      fit   <- .tulpa_gpd_fit(exceed)
      k_hat <- fit$k
      pp    <- (seq_len(tail_len) - 0.5) / tail_len
      smoothed <- log(exp(cutoff) + .tulpa_qgpd(pp, fit$k, fit$sigma))
      smoothed <- pmin(smoothed, max(lw))              # cap at the largest raw weight
      lw[tail_ord] <- smoothed                         # ascending <- ascending
    }
  }

  lw <- lw - .tulpa_logsumexp(lw)                      # normalize
  w  <- exp(lw)
  list(pareto_k = k_hat, is_ess = 1 / sum(w^2), log_weights = lw)
}

# Outer (hyperparameter) Pareto-k-hat for a nested-Laplace fit. Monte-Carlo
# samples theta from the Gaussian proposal the integrator places its grid with
# (mean `theta_hat`, scale `L_scale` so theta = theta_hat + L_scale %*% z,
# z ~ N(0, I)), forms the log importance ratio log p_target(theta) - log q(theta)
# against the inner-marginal target, and PSIS-smooths it. The proposal's
# normalizing constant is common to every draw, so it drops under PSIS's
# self-normalization and only the quadratic 0.5 ||z||^2 enters. Each evaluation
# is one inner Laplace solve, so `n_samples` extra solves are paid -- the cost
# the `diagnose_k` switch controls.
#
# This path applies NO radius cap (unlike the grid/joint `.nested_is_pareto_k`):
# its inner solve is a dense `tulpa_laplace()` that converges in a handful of
# Newton steps even at far-radius draws, so the EVA-scale stalls the #94 cap
# bounds do not arise here. Evaluating every draw is therefore both affordable
# and the unbiased choice -- consistent in correctness with the grid/joint path,
# whose cap now folds the far tail back in whenever it would matter
# (gcol33/tulpa#100).
.nested_outer_pareto_k <- function(log_target, theta_hat, L_scale,
                                   n_samples = 200L) {
  n_samples <- as.integer(n_samples)
  if (n_samples < .PSIS_MIN_EVAL) {                    # cannot reach the floor
    return(list(pareto_k = NA_real_, is_ess = NA_real_, n_eval = 0L))
  }
  k <- length(theta_hat)
  Z <- matrix(stats::rnorm(n_samples * k), n_samples, k)
  lr <- vapply(seq_len(n_samples), function(s) {
    th <- as.numeric(theta_hat + L_scale %*% Z[s, ])
    lt <- log_target(th)
    if (!is.finite(lt)) return(-Inf)
    lt + 0.5 * sum(Z[s, ]^2)                           # - log q(theta), constants dropped
  }, numeric(1))

  n_eval <- sum(is.finite(lr))
  if (n_eval < .PSIS_MIN_EVAL) {
    return(list(pareto_k = NA_real_, is_ess = NA_real_, n_eval = n_eval))
  }
  ps <- tulpa_psis(lr)
  list(pareto_k = ps$pareto_k, is_ess = ps$is_ess, n_eval = n_eval)
}

# Outer Pareto-k-hat for a grid-integrated nested fit (the generic
# `tulpa_nested_laplace` path), where the inner marginal is evaluated by a
# BATCHED re-fit rather than a per-sample closure. Fits a Gaussian proposal to
# the hyperparameter posterior in the unconstrained (log) coordinate `u`, draws
# `n_samples`, re-fits the inner marginal at the back-transformed grid in ONE
# call, and PSIS-smooths log p(theta(u)) + log|d theta / d u| - log q(u). The
# integrator's unnormalized target is exp(log_marginal) in theta-space, so the
# log-Jacobian `sum(u)` (for theta = exp(u)) enters the u-space target; the
# proposal's normalizing constant is common to every draw and drops under PSIS.
#
# `theta_hat` / `L_scale` define the proposal N(theta_hat, L_scale L_scale')
# in the integrator's OWN coordinate space; `log_target_batched(U_matrix)`
# returns the integrator's unnormalized log posterior at the S x d sample in
# that same space (any change-of-variables Jacobian is the caller's job, so the
# diagnostic always matches whatever target the integrator actually weights).
# The proposal's normalizing constant is common to all draws and drops under
# PSIS, leaving the quadratic 0.5||z||^2.
.nested_is_pareto_k <- function(theta_hat, L_scale, log_target_batched,
                                n_samples = 200L, radius_cap = Inf) {
  d <- length(theta_hat)
  n_samples <- as.integer(n_samples)
  if (n_samples < .PSIS_MIN_EVAL) {                          # cannot reach the floor
    return(list(pareto_k = NA_real_, is_ess = NA_real_, n_eval = 0L))
  }
  Z <- matrix(stats::rnorm(n_samples * d), n_samples, d)
  U <- sweep(Z %*% t(L_scale), 2L, theta_hat, `+`)           # S x d ~ N(theta_hat, .)
  z2 <- rowSums(Z^2)                                         # squared whitened radius

  # Cost bound (gcol33/tulpa#94): draws far past the grid's coverage stall the
  # inner Newton toward max_iter and dominate the diagnostic's wall time, so by
  # default only draws within `radius_cap` of the proposal centre (whitened
  # metric) are evaluated. The discarded draws are NOT a guaranteed left tail of
  # the importance ratio, though: the log-ratio is lr = log p_target + 0.5||z||^2,
  # so a target HEAVIER than the Gaussian proposal makes lr RISE with z2 and the
  # far-radius draws carry the LARGEST ratios -- exactly the right-tail mass the
  # GPD k-hat reads. Dropping them would bias k-hat downward in the heavy-tail
  # regime the diagnostic exists to flag (gcol33/tulpa#100). The escalation step
  # below detects that regime and folds the dropped draws back in, so the cap
  # bounds cost only when it cannot bias the result. radius_cap = Inf keeps every
  # draw (the unrestricted re-cov path).
  in_supp <- if (is.finite(radius_cap)) z2 <= radius_cap * radius_cap
             else rep(TRUE, n_samples)
  n_in <- sum(in_supp)
  if (n_in < .PSIS_MIN_EVAL) {                               # too few within support
    return(list(pareto_k = NA_real_, is_ess = NA_real_, n_eval = n_in))
  }

  lr <- rep(-Inf, n_samples)
  lt <- log_target_batched(U[in_supp, , drop = FALSE])
  if (length(lt) != n_in) {
    return(list(pareto_k = NA_real_, is_ess = NA_real_, n_eval = 0L))
  }
  lr[in_supp] <- lt + 0.5 * z2[in_supp]                      # target - log q (up to const)

  # Heavy-tail escalation (gcol33/tulpa#100): if the importance log-ratio is
  # still rising at the cap boundary -- OLS slope of lr on z2 significantly
  # positive, i.e. the target is heavier-tailed than the proposal -- the radius
  # cap is clipping the right tail the k-hat depends on. Evaluate the dropped
  # draws and fold them in so the GPD fits the genuine uncapped tail. A flat or
  # negative slope (target no heavier than the proposal) leaves the cap in force,
  # keeping the well-behaved case at #94's bounded cost.
  if (is.finite(radius_cap) && n_in < n_samples) {
    fin <- in_supp & is.finite(lr)
    if (sum(fin) >= .PSIS_MIN_EVAL && .lr_rises_with_radius(z2[fin], lr[fin])) {
      out  <- !in_supp
      lt2  <- log_target_batched(U[out, , drop = FALSE])
      if (length(lt2) == sum(out)) lr[out] <- lt2 + 0.5 * z2[out]
    }
  }

  n_eval <- sum(is.finite(lr))
  if (n_eval < .PSIS_MIN_EVAL) {
    return(list(pareto_k = NA_real_, is_ess = NA_real_, n_eval = n_eval))
  }
  ps <- tulpa_psis(lr)
  list(pareto_k = ps$pareto_k, is_ess = ps$is_ess, n_eval = n_eval)
}

# One-sided test that the importance log-ratio RISES with the squared whitened
# radius `z2` AT THE CAP BOUNDARY -- the signal that the target is heavier-tailed
# than the Gaussian proposal there, so the radius cap is clipping the right tail
# the GPD k-hat reads (gcol33/tulpa#100). Because lr = log p_target + 0.5*z2, a
# Gaussian target c-times as wide as the proposal gives a constant slope
# 0.5*(1 - 1/c) (>0 when heavier); a genuinely heavy (e.g. Student-t) target is
# flat or falling in the bulk and only rises in the tail, so the GLOBAL slope is
# misleading. The slope is therefore fit on the OUTER HALF of the draws by `z2`
# (those nearest the cap), a local estimate of the boundary behaviour. Returns
# TRUE when that slope is more than `t_crit` standard errors above zero (a
# flat-slope good case fires only at the test's false-positive rate, preserving
# the cost bound).
.lr_rises_with_radius <- function(z2, lr, t_crit = 2.0) {
  keep <- z2 >= stats::median(z2)          # outer half: local slope at the cap
  x <- z2[keep]; y <- lr[keep]
  n <- length(x)
  if (n < 5L) return(FALSE)
  xb  <- mean(x)
  Sxx <- sum((x - xb)^2)
  if (!is.finite(Sxx) || Sxx <= 0) return(FALSE)
  yb    <- mean(y)
  slope <- sum((x - xb) * (y - yb)) / Sxx
  if (!is.finite(slope) || slope <= 0) return(FALSE)
  resid <- y - (yb + slope * (x - xb))
  s2    <- sum(resid^2) / (n - 2)
  se    <- sqrt(s2 / Sxx)
  if (!is.finite(se)) return(FALSE)
  if (se <= 0) return(TRUE)             # exact positive linear trend (e.g. a
                                        # Gaussian target wider than the proposal):
                                        # the strongest rising signal there is
  slope / se > t_crit
}

# Grid path whose integrator works in CONSTRAINED (positive) coordinates: fit
# the Gaussian proposal to the grid posterior in the unconstrained `u = log`
# coordinate and add the log-Jacobian `sum(u)` (theta = exp(u)) to the target.
# `u_grid` is the K x d log-scale grid, `weights` the K integration weights,
# `refit_log_marginal(theta_mat)` maps an S x d CONSTRAINED grid to its S inner
# log-marginals. Restricted by the caller to all-positive-scale axes, so there
# is no bounded-parameter (e.g. correlation) Jacobian to guess.
.nested_grid_pareto_k <- function(u_grid, weights, refit_log_marginal,
                                  n_samples = 200L) {
  u_hat <- as.numeric(crossprod(weights, u_grid))            # weighted mean
  cen   <- sweep(u_grid, 2L, u_hat)
  Su    <- crossprod(cen * weights, cen)                     # weighted covariance
  Su    <- (Su + t(Su)) / 2
  L <- tryCatch(t(chol(Su)), error = function(e) NULL)
  if (is.null(L)) return(list(pareto_k = NA_real_, is_ess = NA_real_, n_eval = 0L))

  lt <- function(U) {
    lm <- refit_log_marginal(exp(U))
    if (length(lm) != nrow(U)) return(rep(NA_real_, nrow(U)))
    lm + rowSums(U)                                          # + log|d theta / d u|
  }
  radius_cap <- .nested_grid_radius_cap(u_grid, u_hat, L)
  .nested_is_pareto_k(u_hat, L, lt, n_samples, radius_cap = radius_cap)
}
