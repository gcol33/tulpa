#' Community / multispecies N-mixture by Laplace
#'
#' @description
#' Fits the community (spAbundance `msNMix`) N-mixture model: a per-species
#' Royle (2004) N-mixture with Gaussian community hyperpriors on the per-species
#' abundance and detection coefficients,
#' \deqn{N_{s,i} \sim \mathrm{Poisson}(\lambda_{s,i}), \quad
#'       y_{s,i,j} | N \sim \mathrm{Binomial}(N_{s,i}, p_{s,i,j}),}
#' \deqn{\log \lambda_{s,i} = X_\lambda^{(i)} (\mu_\lambda + b^\lambda_s), \quad
#'       \mathrm{logit}\, p_{s,i,j} = X_p^{(ij)} (\mu_p + b^p_s),}
#' \deqn{b^\lambda_s \sim N(0, \Sigma_\lambda), \quad b^p_s \sim N(0, \Sigma_p).}
#'
#' The latent abundances integrate out per species-site in closed form (the
#' shared N-mixture kernel exposed by [tulpa_nmix_site_marginal()]); the
#' per-species coefficient deviations \eqn{b_s = (b^\lambda_s, b^p_s)} are the
#' random effects, integrated by the shared compiled AGHQ engine
#' ([tulpa_re_aghq()] at `n_quad = 1`, i.e. the joint Laplace). Each species'
#' marginal -- value, abundance/detection score, and the per-site observed-
#' information block carrying the \eqn{\mathrm{Var}(N_i\mid y_i)} abundance/
#' detection coupling -- is supplied as the per-group oracle, so there is one
#' marginal/quadrature/covariance implementation across the package. The
#' Gaussian community priors pin \eqn{(\mu_\lambda, \mu_p)} as fixed effects, so
#' no sum-to-zero constraint is needed; their standard errors come from the
#' marginal observed-information Hessian.
#'
#' Poisson abundance only (a global negative-binomial size is a planned
#' extension).
#'
#' @param y Integer vector of counts, one entry per observed visit (long form,
#'   all species stacked).
#' @param site_idx Integer vector (same length as `y`), 1-based site index.
#' @param species_idx Integer vector (same length as `y`), 1-based species index.
#' @param X_lambda Numeric matrix `[n_sites x p_lambda]` of abundance covariates
#'   (shared across species; one row per site).
#' @param X_p Numeric matrix `[n_obs x p_p]` of detection covariates (long form,
#'   row order matching `y`).
#' @param n_sites,n_species Integer counts.
#' @param mu_lambda_init,mu_p_init Optional warm starts for the community means.
#'   Default: the column means of independent per-species [tulpa_nmix_laplace()]
#'   fits.
#' @param Sigma_lambda_init,Sigma_p_init Optional warm starts for the community
#'   covariances. Default: the (ridge-regularized) sample covariance of the
#'   per-species coefficient estimates.
#' @param K_max Marginal-sum truncation (default `max(y) + 100`).
#' @param max_iter Optimizer iteration cap (default 200).
#' @param sigma_beta Weak Gaussian ridge SD on the community means (default 100,
#'   i.e. `tau = 1e-4`, matching the other Laplace paths); stabilizes a
#'   weakly-identified community mean without materially shifting it.
#' @param verbose Unused (kept for backward compatibility); the engine is silent.
#'
#' @return A list of class `tulpa_nmix_re_fit`: `mu_lambda`, `mu_p` (community
#'   means), `vcov` (their joint covariance from the AGHQ marginal Hessian,
#'   `(p_lambda + p_p)` square; marginalizes the community-covariance
#'   uncertainty rather than plugging in `Sigma`), `Sigma_lambda`, `Sigma_p`
#'   (community covariances), `b_lambda`, `b_p` (per-species BLUP deviations,
#'   `n_species` rows), `log_lik` (Laplace marginal), `converged`, `K_max`.
#'
#' @references
#' Royle, J. A. (2004). N-mixture models for estimating population size from
#'   spatially replicated counts. *Biometrics* 60, 108-115.
#' Doser, J. et al. (2023). spAbundance. `msNMix()`.
#'
#' @seealso [tulpa_nmix_laplace()] (single species), [tulpa_nmix_site_marginal()]
#'   (the per-species marginal primitive), [tulpa_re_aghq()] (the shared
#'   random-effect integrator).
#' @export
tulpa_nmix_laplace_re <- function(y, site_idx, species_idx,
                                  X_lambda, X_p, n_sites, n_species,
                                  mu_lambda_init = NULL, mu_p_init = NULL,
                                  Sigma_lambda_init = NULL, Sigma_p_init = NULL,
                                  K_max = NULL, max_iter = 200L,
                                  sigma_beta = 100, verbose = FALSE) {
  y        <- as.integer(y)
  site_idx <- as.integer(site_idx)
  species_idx <- as.integer(species_idx)
  if (!is.matrix(X_lambda)) stop("`X_lambda` must be a numeric matrix.", call. = FALSE)
  if (!is.matrix(X_p))      stop("`X_p` must be a numeric matrix.", call. = FALSE)
  n_obs <- length(y)
  if (length(site_idx) != n_obs || length(species_idx) != n_obs) {
    stop("`site_idx` and `species_idx` must have the same length as `y`.", call. = FALSE)
  }
  if (nrow(X_p) != n_obs) stop("nrow(X_p) must equal length(y).", call. = FALSE)
  if (nrow(X_lambda) != n_sites) stop("nrow(X_lambda) must equal n_sites.", call. = FALSE)
  p_lam <- ncol(X_lambda); p_p <- ncol(X_p)
  # Marginal-sum truncation. Default max(y) + 100 (matching tulpa_nmix_laplace):
  # it must cover the latent-N posterior, which the observed counts pull ABOVE
  # the prior-lambda tail, so a qpois(lambda) cap under-covers and truncates.
  # The lgamma cache makes a generous K_max cheap, so correctness wins.
  if (is.null(K_max)) K_max <- max(y) + 100L
  K_max <- as.integer(K_max)

  # ---- warm start: independent per-species fixed-effect fits ----
  if (is.null(mu_lambda_init) || is.null(Sigma_lambda_init) ||
      is.null(mu_p_init) || is.null(Sigma_p_init)) {
    B_lam <- matrix(NA_real_, n_species, p_lam)
    B_p   <- matrix(NA_real_, n_species, p_p)
    for (s in seq_len(n_species)) {
      sel <- species_idx == s
      # Warm start only seeds the community fit, so a boundary-weight warning on
      # a throwaway per-species init fit is not actionable -- suppress it (the
      # community fit's own K_max governs the final truncation).
      f <- tryCatch(
        suppressWarnings(
          tulpa_nmix_laplace(y = y[sel], site_idx = site_idx[sel],
                             X_lambda = X_lambda, X_p = X_p[sel, , drop = FALSE],
                             mixture = "P", K_max = K_max, verbose = FALSE)),
        error = function(e) NULL)
      if (!is.null(f)) {
        if (all(is.finite(f$beta_lambda))) B_lam[s, ] <- f$beta_lambda
        if (all(is.finite(f$beta_p)))      B_p[s, ]   <- f$beta_p
      }
    }
    ok_l <- stats::complete.cases(B_lam)
    ok_p <- stats::complete.cases(B_p)
    if (is.null(mu_lambda_init))
      mu_lambda_init <- if (any(ok_l)) colMeans(B_lam[ok_l, , drop = FALSE])
                        else c(log(mean(y) + 1), rep(0, p_lam - 1L))
    if (is.null(mu_p_init))
      mu_p_init <- if (any(ok_p)) colMeans(B_p[ok_p, , drop = FALSE]) else rep(0, p_p)
    if (is.null(Sigma_lambda_init)) Sigma_lambda_init <- .nmix_re_cov0(B_lam[ok_l, , drop = FALSE], p_lam)
    if (is.null(Sigma_p_init))      Sigma_p_init      <- .nmix_re_cov0(B_p[ok_p, , drop = FALSE], p_p)
  }

  # ---- per-species marginal oracle, integrated by the shared engine ----
  orc <- .nmix_re_oracle(y, site_idx, species_idx, X_lambda, X_p,
                         n_sites, n_species, p_lam, p_p, K_max)

  fit <- tulpa_re_aghq(
    theta0  = c(as.numeric(mu_lambda_init), as.numeric(mu_p_init)),
    re_terms = list(list(n_groups = n_species, n_coefs = p_lam, correlated = p_lam > 1L),
                    list(n_groups = n_species, n_coefs = p_p,   correlated = p_p   > 1L)),
    Sigma0  = list(as.matrix(Sigma_lambda_init), as.matrix(Sigma_p_init)),
    make_group = function(theta) list(
      grad_hess = function(g, b) orc$grad_hess(g, b + theta),
      node_ll   = function(g, B) orc$node_ll(g, sweep(B, 2, theta, `+`))),
    n_quad = 1L, theta_prior_sd = sigma_beta, maxit = as.integer(max_iter))

  if (is.null(fit)) {
    stop("Community N-mixture optimization failed (singular marginal Hessian ",
         "or non-finite optimum). Try a different warm start or K_max.", call. = FALSE)
  }

  mu <- fit$theta
  out <- list(
    mu_lambda    = mu[seq_len(p_lam)],
    mu_p         = mu[p_lam + seq_len(p_p)],
    vcov         = fit$theta_cov,
    Sigma_lambda = fit$Sigma_list[[1L]],
    Sigma_p      = fit$Sigma_list[[2L]],
    b_lambda     = fit$blup[[1L]],
    b_p          = fit$blup[[2L]],
    log_lik      = fit$log_marginal,
    converged    = fit$converged,
    K_max        = K_max,
    mixture      = "P"
  )
  class(out) <- c("tulpa_nmix_re_fit", "list")
  out
}

# Per-species community N-mixture oracle for the shared RE integrator. Each
# species' marginal (tulpa_nmix_site_marginal) is built once; the oracle
# evaluates it at the full coefficients coef = (mu + b_s), mapping the RE
# deviation through the abundance / detection designs. The b-space curvature is
# the design-sandwiched per-site observed-information block (with the latent-N
# coupling) -- the covariate generalization of the intercept-only assembly.
.nmix_re_oracle <- function(y, site_idx, species_idx, X_lambda, X_p,
                            n_sites, n_species, p_lam, p_p, K_max) {
  cl <- function(e) pmin(pmax(e, -30), 30)
  d  <- p_lam + p_p
  rows_by_sp <- split(seq_len(length(y)), species_idx)
  marg <- lapply(seq_len(n_species), function(s) {
    sel <- rows_by_sp[[as.character(s)]]
    if (is.null(sel)) sel <- integer(0)
    tulpa_nmix_site_marginal(
      y = y[sel], site_idx = site_idx[sel],
      X_lambda = X_lambda, X_p = X_p[sel, , drop = FALSE],
      mixture = "P", K_max = K_max)
  })

  eta_of <- function(m, coef) {
    list(lambda = cl(as.numeric(m$X_lambda %*% coef[seq_len(p_lam)])),
         p      = cl(as.numeric(m$X_p %*% coef[p_lam + seq_len(p_p)])))
  }

  list(
    grad_hess = function(s, coef) {
      m  <- marg[[s]]
      e  <- eta_of(m, coef)
      ev <- m$eval(e$lambda, e$p)
      grad <- c(as.numeric(crossprod(m$X_lambda, ev$grad_eta_lambda)),
                as.numeric(crossprod(m$X_p, ev$grad_eta_p)))
      # negH: marginal observed info (with the Var[N|y] abundance/detection
      # coupling), used by the Laplace marginal. fisher: complete-data Fisher
      # (block-diagonal, PSD), supplied for the mode-find Newton -- the observed
      # info can be indefinite away from the mode (latent-N coupling).
      negH <- matrix(0, d, d); fisher <- matrix(0, d, d)
      for (i in seq_len(m$n_sites)) {
        obs <- m$obs_by_site[[i]]; Ji <- length(obs)
        Zi  <- matrix(0, 1L + Ji, d)
        Zi[1L, seq_len(p_lam)] <- m$X_lambda[i, ]
        if (Ji > 0L) Zi[-1L, p_lam + seq_len(p_p)] <- m$X_p[obs, , drop = FALSE]
        negH   <- negH + crossprod(Zi, m$obs_info_block(i, ev) %*% Zi)
        Fdiag  <- c(ev$info_eta_lambda[i], if (Ji > 0L) ev$info_eta_p[obs] else numeric(0))
        fisher <- fisher + crossprod(Zi, Fdiag * Zi)
      }
      list(logL = ev$log_lik, grad = grad, negH = negH, fisher = fisher)
    },
    node_ll = function(s, COEF) {
      m <- marg[[s]]
      vapply(seq_len(nrow(COEF)), function(k) {
        e <- eta_of(m, COEF[k, ])
        m$eval(e$lambda, e$p)$log_lik
      }, numeric(1))
    })
}

# Method-of-moments community covariance seed: the sample covariance of the
# per-species coefficient estimates, ridge-regularized to PD.
.nmix_re_cov0 <- function(B, p) {
  if (is.null(B) || nrow(B) < 2L) return(diag(0.25, p))
  V <- stats::cov(B)
  V[!is.finite(V)] <- 0
  V + diag(max(1e-3, 1e-3 * mean(abs(diag(V)))), p)
}
