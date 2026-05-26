#' Per-site N-mixture marginal as a composable random-effect callback
#'
#' @description
#' Exposes the Royle (2004) N-mixture per-site marginal -- the latent abundance
#' \eqn{N_i} summed out in closed form -- as a reusable building block for
#' integrating grouped random effects over the abundance and/or detection
#' linear predictors. It is the bridge between the marginal kernel (which
#' already differentiates through both arms) and a random-effect integrator
#' such as [tulpa_re_aghq()].
#'
#' Unlike [tulpa_nmix_laplace()], which fixes the coefficients and returns a
#' point fit, this helper returns a *closure object* that evaluates the marginal
#' and its eta-level derivatives at arbitrary linear predictors. A random-effect
#' integrator perturbs the predictors by \eqn{Z b} per group and calls `eval()`
#' at each quadrature point; the per-site value, gradient and observed-
#' information block are everything the per-group integral needs.
#'
#' The abundance arm is per site (\eqn{\log\lambda_i = \eta^\lambda_i}) and the
#' detection arm is per visit (\eqn{\mathrm{logit}\,p_{ij} = \eta^p_{ij}}); the
#' two are coupled through the shared latent \eqn{N_i}. The per-site marginal
#' observed information in the eta coordinates
#' \eqn{(\eta^\lambda_i, \eta^p_{i1}, \dots, \eta^p_{iJ_i})} is
#' \deqn{B_i = \mathrm{diag}(I^\lambda_i, I^p_{ij}) - \mathrm{Var}(N_i\mid y_i)\,
#'       v_i v_i^\top, \qquad v_i = (-w_i, p_{i1}, \dots, p_{iJ_i}),}
#' where \eqn{I^\lambda_i}, \eqn{I^p_{ij}} are the complete-data Fisher diagonal,
#' \eqn{w_i} is the abundance score weight (`1` Poisson, \eqn{1-q_i} NB) and
#' \eqn{p_{ij} = \mathrm{plogis}(\eta^p_{ij})}. The off-diagonal
#' \eqn{\mathrm{Var}(N_i)\,w_i\,p_{ij}} is the abundance/detection coupling an
#' integrator placing random effects on both arms must carry (Louis 1982). This
#' is the eta-level form of the curvature [tulpa_nmix_laplace()] sandwiches with
#' the design matrices.
#'
#' The single- vs multi-arm `make_site` adapter that wires this into a specific
#' integrator is deliberately not built here -- this object is the arm-agnostic
#' foundation it sits on.
#'
#' @inheritParams tulpa_nmix_laplace
#'
#' @return An object of class `tulpa_nmix_marginal`: a list of the validated
#'   data plus closures
#'   * `eval(eta_lambda, eta_p, r = Inf)` -- evaluate at the abundance log
#'     predictor `eta_lambda` (length `n_sites`) and detection logit predictor
#'     `eta_p` (length `n_obs`, visit order matching `y`). Returns a list with
#'     `log_lik` (total), `log_lik_site`, `grad_eta_lambda`, `grad_eta_p`,
#'     `grad_theta`, the complete-data Fisher `info_eta_lambda` / `info_eta_p`,
#'     the abundance score weight `score_wt_lambda`, `p` (= `plogis(eta_p)`),
#'     `mean_N`, `var_N`, `boundary_weight`, and (NB) the dispersion-coupling
#'     pieces `info_theta` / `info_lambda_theta` / `cov_N_stheta` / `var_stheta`.
#'   * `eval_beta(beta_lambda, beta_p, r = Inf)` -- the same, with the predictors
#'     formed from the stored design matrices.
#'   * `obs_info_block(s, ev)` -- the \eqn{(1+J_s)\times(1+J_s)} per-site
#'     marginal observed-information matrix \eqn{B_s} (above) for site `s`, from
#'     an `eval()` / `eval_beta()` result `ev`. Coordinates are
#'     `(eta_lambda_s, eta_p over site s's visits in input order)`.
#'   Plus `n_sites`, `n_obs`, `p_lambda`, `p_p`, `mixture`, `K_max`,
#'   `obs_by_site` (per-site visit row indices).
#'
#' @references
#' Royle, J. A. (2004). N-mixture models for estimating population size from
#'   spatially replicated counts. *Biometrics* 60, 108-115.
#' Louis, T. A. (1982). Finding the observed information matrix when using the
#'   EM algorithm. *JRSS-B* 44, 226-233.
#'
#' @seealso [tulpa_nmix_laplace()] for the fixed-effects fit, [tulpa_re_aghq()]
#'   for the grouped random-effect integrator this feeds.
#' @export
tulpa_nmix_site_marginal <- function(y,
                                     site_idx,
                                     X_lambda,
                                     X_p,
                                     mixture = c("P", "NB"),
                                     K_max = NULL) {
  mixture <- match.arg(mixture)
  y        <- as.integer(y)
  site_idx <- as.integer(site_idx)
  if (!is.matrix(X_lambda)) stop("`X_lambda` must be a numeric matrix.", call. = FALSE)
  if (!is.matrix(X_p))      stop("`X_p` must be a numeric matrix.", call. = FALSE)
  if (length(y) != length(site_idx)) {
    stop("length(y) must equal length(site_idx).", call. = FALSE)
  }
  if (length(y) != nrow(X_p)) {
    stop("length(y) must equal nrow(X_p).", call. = FALSE)
  }
  if (any(y < 0L) || anyNA(y)) {
    stop("`y` must be nonnegative integers with no NA.", call. = FALSE)
  }
  n_sites <- nrow(X_lambda)
  n_obs   <- length(y)
  if (min(site_idx) < 1L || max(site_idx) > n_sites) {
    stop("`site_idx` values must lie in [1, nrow(X_lambda)].", call. = FALSE)
  }
  p_lambda <- ncol(X_lambda)
  p_p      <- ncol(X_p)

  if (is.null(K_max)) {
    K_max <- as.integer(max(y) + 100L)
  } else {
    K_max <- as.integer(K_max)
    if (K_max < max(y)) stop("`K_max` must be >= max(y).", call. = FALSE)
  }

  obs_by_site <- split(seq_len(n_obs), site_idx)
  # Re-key to a dense 1..n_sites list (sites with no visits get integer(0)).
  obs_by_site <- lapply(seq_len(n_sites), function(s) {
    o <- obs_by_site[[as.character(s)]]
    if (is.null(o)) integer(0) else o
  })

  resolve_r <- function(r) {
    if (identical(mixture, "P")) return(Inf)
    if (is.null(r) || !is.finite(r) || r <= 0) {
      stop("NB marginal requires a finite positive `r` (NB size).", call. = FALSE)
    }
    as.numeric(r)
  }

  eval_eta <- function(eta_lambda, eta_p, r = Inf) {
    eta_lambda <- as.numeric(eta_lambda)
    eta_p      <- as.numeric(eta_p)
    if (length(eta_lambda) != n_sites) {
      stop("length(eta_lambda) must equal n_sites.", call. = FALSE)
    }
    if (length(eta_p) != n_obs) {
      stop("length(eta_p) must equal n_obs.", call. = FALSE)
    }
    out <- cpp_nmix_total_log_lik(y, site_idx, eta_p, eta_lambda, K_max,
                                  r = resolve_r(r))
    out$p <- plogis(eta_p)
    out
  }

  eval_beta <- function(beta_lambda, beta_p, r = Inf) {
    beta_lambda <- as.numeric(beta_lambda)
    beta_p      <- as.numeric(beta_p)
    if (length(beta_lambda) != p_lambda) {
      stop("length(beta_lambda) must equal ncol(X_lambda).", call. = FALSE)
    }
    if (length(beta_p) != p_p) {
      stop("length(beta_p) must equal ncol(X_p).", call. = FALSE)
    }
    eval_eta(as.numeric(X_lambda %*% beta_lambda),
             as.numeric(X_p %*% beta_p), r = r)
  }

  # Per-site marginal observed-information block B_s from an eval() result.
  # Coordinates: (eta_lambda_s, eta_p over site s's visits in input order).
  # B_s = diag(I_lam, I_p) - var_N * v v', v = (-score_wt_lambda, p_visits).
  obs_info_block <- function(s, ev) {
    obs <- obs_by_site[[s]]
    J <- length(obs)
    d <- 1L + J
    B <- matrix(0, d, d)
    B[1, 1] <- ev$info_eta_lambda[s]
    if (J > 0L) {
      diag(B)[-1] <- ev$info_eta_p[obs]
      vN <- ev$var_N[s]
      v  <- c(-ev$score_wt_lambda[s], ev$p[obs])
      B  <- B - vN * tcrossprod(v)
    }
    B
  }

  structure(
    list(
      y = y, site_idx = site_idx, X_lambda = X_lambda, X_p = X_p,
      mixture = mixture, K_max = K_max,
      n_sites = n_sites, n_obs = n_obs, p_lambda = p_lambda, p_p = p_p,
      obs_by_site = obs_by_site,
      eval = eval_eta, eval_beta = eval_beta, obs_info_block = obs_info_block
    ),
    class = c("tulpa_nmix_marginal", "list")
  )
}

#' @export
print.tulpa_nmix_marginal <- function(x, ...) {
  cat(sprintf("tulpa N-mixture per-site marginal (mixture = %s)\n", x$mixture))
  cat(sprintf("  n_sites = %d   n_obs = %d   K_max = %d\n",
              x$n_sites, x$n_obs, x$K_max))
  cat(sprintf("  p_lambda = %d   p_p = %d\n", x$p_lambda, x$p_p))
  cat("  closures: eval(eta_lambda, eta_p, r), eval_beta(beta_lambda, beta_p, r), obs_info_block(s, ev)\n")
  invisible(x)
}
