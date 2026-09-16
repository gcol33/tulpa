# priorsense.R
# ------------------------------------------------------------------------------
# Power-scaling prior/likelihood sensitivity (Kallioinen, Paananen, Buerkner,
# Vehtari 2024). Raising the prior (or likelihood) to a power alpha and
# importance-reweighting the existing draws by exp((alpha-1) * log_component)
# shows how much the posterior moves; the sensitivity is the gradient of the
# cumulative Jensen-Shannon (CJS) distance between the base and power-scaled
# posteriors with respect to log2(alpha). No refits -- it reuses the native PSIS
# smoother (tulpa_psis). The CJS distance and the gradient reproduce the
# priorsense reference implementation (n-kall/priorsense: R/cjs.R, the CJS of
# Nguyen & Vreeken 2015; and R/powerscale_gradients.R for the gradient).
#
# Scope: the fixed-effect prior (a Gaussian `prior = list(mean, sd)` the fit
# used), the observation likelihood the fit's own linear predictor carries
# (fixed effects, formula random effects, the offset, n_trials, phi2, and any
# field folded into eta at fit time through .tulpa_eta_draws()), and -- when
# the fit stores per-draw hyperparameter log-prior values
# ($hyper_log_prior_draws, recorded at draw-synthesis time by the
# nested-Laplace mixture paths) -- the hyperparameter prior via the same
# reweighting. A nested-Laplace outer-grid mixture fit (fitted_eta + weights)
# has no fixed-effect-only decomposition of eta and is refused outright, as is
# any fit carrying an inline spatial/temporal field.
# ------------------------------------------------------------------------------

# Seed and default draw count for every RNG-consuming step this file takes,
# pinned to one value so .ps_fixed_draws()'s beta and .ps_component_loglik()'s
# eta -- two separate calls into .fixed_coef_draws() / .tulpa_eta_draws(), each
# resetting to this seed -- reproduce the identical draw sequence and stay
# row-aligned without threading beta through the log-likelihood call.
.PS_SEED    <- 285703L
.PS_NDRAWS  <- 4000L

# Cumulative Jensen-Shannon divergence CJS(P || Q) for two weighted ECDFs of the
# SAME sorted draws (the power-scaling case: x == y, only the weights differ).
# P, Q are the weighted ECDFs (cumsum of normalized weights). Follows priorsense
# R/cjs.R exactly: the log2 CJS integral plus the 0.5/ln2 (Q_int - P_int) term,
# normalized by the P_int + Q_int upper bound.
.cjs_identical <- function(x, wp, wq) {
  ord <- order(x)
  x   <- x[ord]; wp <- wp[ord]; wq <- wq[ord]
  n   <- length(x)
  if (n < 2L) return(NA_real_)
  binwidth <- diff(x)
  px <- cumsum(wp / sum(wp)); px <- px[-n]
  qx <- cumsum(wq / sum(wq)); qx <- qx[-n]
  px_int <- sum(px * binwidth); qx_int <- sum(qx * binwidth)
  cjs_pq <- sum(binwidth * (px * (log2(px) - log2(0.5 * px + 0.5 * qx))),
                na.rm = TRUE) + 0.5 / log(2) * (qx_int - px_int)
  cjs_qp <- sum(binwidth * (qx * (log2(qx) - log2(0.5 * qx + 0.5 * px))),
                na.rm = TRUE) + 0.5 / log(2) * (px_int - qx_int)
  bound <- px_int + qx_int
  if (!is.finite(bound) || bound <= 0) return(NA_real_)
  sqrt((cjs_pq + cjs_qp) / bound)
}

# Symmetric CJS distance between the base (uniform-weight) and power-scaled
# posteriors of one parameter, unsigned (max over x and -x, per priorsense).
.cjs_dist_ps <- function(x, w_base, w_scaled) {
  a <- .cjs_identical(x, w_base, w_scaled)
  b <- .cjs_identical(-x, w_base, w_scaled)
  max(a, b, na.rm = TRUE)
}

# PSIS-smoothed, normalized importance weights for log-ratios (alpha-1)*comp.
.ps_smoothed_weights <- function(log_ratios) {
  ps <- tulpa_psis(log_ratios)
  lw <- ps$log_weights
  w  <- exp(lw - max(lw))
  w / sum(w)
}

# Draw from N(mu, Sigma) via Cholesky, returning an [n x p] matrix named by mu.
.ps_rmvnorm <- function(n, mu, Sigma) {
  p <- length(mu)
  R <- tryCatch(chol(Sigma), error = function(e) {
    ev <- eigen(Sigma, symmetric = TRUE)
    ev$values[ev$values < 1e-10] <- 1e-10
    chol((ev$vectors %*% (ev$values * t(ev$vectors))))
  })
  Z <- matrix(stats::rnorm(n * p), n, p)
  out <- Z %*% R
  out <- sweep(out, 2L, mu, "+")
  colnames(out) <- names(mu)
  out
}

# Fixed-effect posterior draws [S x p] with columns ordered by `bnm`, from the
# same source .tulpa_eta_draws() reads: .fixed_coef_draws() -- a fit's genuine
# draws when present (genuine = TRUE), otherwise a synthesized draw from the
# (Laplace / Gaussian) fixed-effect posterior N(coef, vcov). Seeded so a
# synthesized draw here is bit-for-bit the one .ps_component_loglik()'s own
# .tulpa_eta_draws() call reproduces from the same seed.
.ps_fixed_draws <- function(fit, bnm) {
  .preserve_seed_in_frame()
  set.seed(.PS_SEED)
  cd <- .fixed_coef_draws(fit, .PS_NDRAWS)
  if (!all(bnm %in% colnames(cd$beta))) {
    stop("Power-scaling needs the fixed-effect posterior: the fit exposes ",
         "neither draws nor a usable coef()/vcov().", call. = FALSE)
  }
  list(B = cd$beta[, bnm, drop = FALSE], genuine = !is.null(.fixed_draws_mat(fit)))
}

# Per-draw total log-likelihood the fit's own linear predictor carries: fixed
# effects, formula random effects, the offset, n_trials and phi2 -- the same
# .tulpa_eta_draws() / .tulpa_eta_loglik() assembly .tulpa_loglik_parts() reads
# for waic() / loo() -- rather than the fixed-effect predictor alone. Seeded to
# reproduce, row for row, the fixed-effect draws .ps_fixed_draws() returns.
.ps_component_loglik <- function(fit) {
  eta <- .tulpa_eta_draws(fit, ndraws = .PS_NDRAWS, synth_seed = .PS_SEED)
  rowSums(.tulpa_eta_loglik(fit, eta))
}

# Per-draw fixed-effect Gaussian log-prior at prior = list(mean, sd).
.ps_component_logprior <- function(B, prior) {
  p  <- ncol(B)
  mu <- rep_len(prior$mean %||% 0, p)
  sd <- prior$sd
  if (is.null(sd)) {
    stop("Prior-component sensitivity needs the fixed-effect prior the fit ",
         "used: pass `prior = list(mean =, sd =)`.", call. = FALSE)
  }
  sd <- rep_len(sd, p)
  rowSums(vapply(seq_len(p), function(j)
    stats::dnorm(B[, j], mu[j], sd[j], log = TRUE), numeric(nrow(B))))
}

#' Power-scaling prior / likelihood sensitivity
#'
#' @description
#' Local power-scaling sensitivity (Kallioinen et al. 2024): how much each
#' fixed-effect posterior moves when the prior or the likelihood is raised to a
#' power `alpha` near 1. The existing draws are importance-reweighted by
#' `exp((alpha - 1) * log_component)` (PSIS-smoothed via [tulpa_psis()]; no
#' refits), and the sensitivity is the gradient of the cumulative
#' Jensen-Shannon distance between the base and power-scaled posteriors with
#' respect to `log2(alpha)`.
#'
#' Values above `threshold` flag sensitivity. High on **both** the prior and
#' likelihood components indicates potential prior-data conflict; high prior
#' with low likelihood indicates a strong prior / weak likelihood.
#'
#' When the fit recorded per-draw hyperparameter log-prior values at
#' draw-synthesis time (`$hyper_log_prior_draws`, stored by the nested-Laplace
#' mixture paths such as [tulpa_re_cov_nested()] and the `tulpa()` random-slope
#' redirect), a `hyperparameter` column reports the power-scaling sensitivity
#' of the hyperparameter prior by the same reweighting; `NA` otherwise.
#'
#' @param fit A `tulpa_fit` fitted through [tulpa()] (fixed-effect / GLMM;
#'   spatial / temporal-field fits and nested-Laplace outer-grid mixture fits,
#'   e.g. a `latent()` block or a random-slope redirect, are rejected).
#' @param prior The Gaussian fixed-effect prior the fit used, as
#'   `list(mean =, sd =)` (scalars recycled). Required for the prior component;
#'   omit to compute the likelihood component only.
#' @param lower_alpha,upper_alpha Power-scaling grid endpoints for the gradient
#'   (defaults 0.99 / 1.01, as in priorsense).
#' @param threshold Sensitivity flag threshold (default 0.05).
#'
#' @return A data frame with one row per fixed-effect parameter and columns
#'   `variable`, `prior`, `hyperparameter`, `likelihood`, `diagnosis`.
#'
#' @references Kallioinen, Paananen, Buerkner & Vehtari (2024). Detecting and
#'   diagnosing prior and likelihood sensitivity with power-scaling.
#'   \emph{Statistics and Computing} 34:57. Nguyen & Vreeken (2015).
#'   Non-parametric Jensen-Shannon divergence. ECML PKDD.
#' @seealso [tulpa_psis()], [tulpa_criteria()].
#' @examples
#' \donttest{
#' set.seed(1)
#' d <- data.frame(x = rnorm(150))
#' d$y <- rpois(150, exp(0.5 + 0.7 * d$x))
#' fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace",
#'              beta_prior = list(mean = 0, sd = 5))
#' tulpa_powerscale_sensitivity(fit, prior = list(mean = 0, sd = 5))
#' }
#' @export
tulpa_powerscale_sensitivity <- function(fit, prior = NULL,
                                         lower_alpha = 0.99, upper_alpha = 1.01,
                                         threshold = 0.05) {
  if (!inherits(fit, "tulpa_fit")) {
    stop("`fit` must be a tulpa_fit.", call. = FALSE)
  }
  if (!is.null(fit$spatial) || !is.null(fit$temporal) ||
      !is.null(fit$temporal_field) || !is.null(fit$spatial_fields) ||
      !is.null(fit$temporal_fields) || !is.null(fit$field_eta_contrib) ||
      identical(.tulpa_linpred_source(fit), "grid_mixture")) {
    stop("Power-scaling sensitivity is not supported for spatial / ",
         "temporal-field fits, or a nested-Laplace outer-grid mixture fit ",
         "(e.g. a `latent()` block or a random-slope redirect).",
         call. = FALSE)
  }
  if (is.null(fit$formula)) {
    stop("Power-scaling sensitivity needs the fit's engine formula, which ",
         "this fit's class does not carry (e.g. a tulpaObs fit).",
         call. = FALSE)
  }
  if (!is.character(fit$family) || length(fit$family) != 1L) {
    stop("Power-scaling sensitivity needs a single built-in family name; ",
         "this fit's family is not one.", call. = FALSE)
  }
  bnm <- names(stats::coef(fit))
  fd  <- .ps_fixed_draws(fit, bnm)
  B   <- fd$B
  S   <- nrow(B)
  w_base <- rep(1 / S, S)

  la <- as.numeric(lower_alpha); ua <- as.numeric(upper_alpha)
  grad_for <- function(comp) {
    wl <- .ps_smoothed_weights((la - 1) * comp)
    wu <- .ps_smoothed_weights((ua - 1) * comp)
    vapply(seq_len(ncol(B)), function(j) {
      x  <- B[, j]
      dl <- .cjs_dist_ps(x, w_base, wl)
      du <- .cjs_dist_ps(x, w_base, wu)
      # Mean of the one-sided gradients (priorsense powerscale_divergence_gradients).
      0.5 * (du / log2(ua) + (-1) * dl / log2(la))
    }, numeric(1))
  }

  lik  <- .ps_component_loglik(fit)
  lik_s <- grad_for(lik)

  if (!is.null(prior)) {
    prio  <- .ps_component_logprior(B, prior)
    prior_s <- grad_for(prio)
  } else {
    prior_s <- rep(NA_real_, ncol(B))
  }

  # Hyperparameter-prior component: available when the fit recorded the
  # per-draw hyperparameter log-prior at draw-synthesis time (the
  # nested-Laplace mixture paths). Requires the fit's own draws -- the
  # synthesized-Gaussian fallback has no row alignment with them.
  hyper_lp <- fit$hyper_log_prior_draws
  hyper_s  <- rep(NA_real_, ncol(B))
  if (!is.null(hyper_lp) && isTRUE(fd$genuine) && length(hyper_lp) == S &&
      all(is.finite(hyper_lp))) {
    hyper_s <- grad_for(hyper_lp)
  }

  diagnosis <- ifelse(
    !is.na(prior_s) & prior_s >= threshold & lik_s >= threshold,
    "potential prior-data conflict",
    ifelse(!is.na(prior_s) & prior_s >= threshold & lik_s < threshold,
           "potential strong prior / weak likelihood", "-"))

  data.frame(variable = bnm, prior = prior_s, hyperparameter = hyper_s,
             likelihood = lik_s, diagnosis = diagnosis,
             stringsAsFactors = FALSE, row.names = NULL)
}
