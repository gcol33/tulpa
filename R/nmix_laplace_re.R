#' Community / multispecies N-mixture by Laplace-EM
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
#' shared N-mixture kernel); the per-species coefficient deviations
#' \eqn{b_s = (b^\lambda_s, b^p_s)} are the random effects, integrated by a
#' Laplace-EM in C++ (`cpp_nmix_laplace_re`). The Gaussian community priors pin
#' \eqn{(\mu_\lambda, \mu_p)} as fixed effects, so no sum-to-zero constraint is
#' needed. Fixed-effect standard errors come from the marginal observed-
#' information Hessian (the b-block Schur complement, with the \eqn{\mathrm{Var}
#' [N|y]} rank-1 correction). This is the compiled counterpart of wrapping
#' [tulpa_nmix_site_marginal()] into [tulpa_re_aghq()]; it runs the same Laplace
#' marginal without the per-group R-interpreter overhead.
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
#' @param max_iter,tol EM iteration budget / `Sigma`-change tolerance.
#' @param inner_max,inner_tol Inner mode-finder (block-coordinate Newton) budget
#'   / step tolerance.
#' @param sigma_beta Weak ridge SD on the community means (default 100, i.e.
#'   `tau = 1e-4`, matching the other Laplace paths).
#' @param verbose Print per-iteration `Sigma` change.
#'
#' @return A list of class `tulpa_nmix_re_fit`: `mu_lambda`, `mu_p` (community
#'   means), `vcov` (their joint covariance, `(p_lambda + p_p)` square),
#'   `Sigma_lambda`, `Sigma_p` (community covariances), `b_lambda`, `b_p` (per-
#'   species BLUP deviations, `n_species` rows), `log_lik` (Laplace marginal),
#'   `converged`, `n_iter`, `K_max`.
#'
#' @references
#' Royle, J. A. (2004). N-mixture models for estimating population size from
#'   spatially replicated counts. *Biometrics* 60, 108-115.
#' Doser, J. et al. (2023). spAbundance. `msNMix()`.
#'
#' @seealso [tulpa_nmix_laplace()] (single species), [tulpa_nmix_site_marginal()]
#'   / [tulpa_re_aghq()] (the composable RE-integrator path).
#' @export
tulpa_nmix_laplace_re <- function(y, site_idx, species_idx,
                                  X_lambda, X_p, n_sites, n_species,
                                  mu_lambda_init = NULL, mu_p_init = NULL,
                                  Sigma_lambda_init = NULL, Sigma_p_init = NULL,
                                  K_max = NULL, max_iter = 100L, tol = 1e-6,
                                  inner_max = 50L, inner_tol = 1e-8,
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
      # Warm start only seeds the community EM, so a boundary-weight warning on
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

  raw <- cpp_nmix_laplace_re(
    y = y, site_idx = site_idx, species_idx = species_idx,
    X_lambda = X_lambda, X_p = X_p,
    n_sites = as.integer(n_sites), n_species = as.integer(n_species),
    Sigma_lambda_init = as.matrix(Sigma_lambda_init),
    Sigma_p_init = as.matrix(Sigma_p_init),
    mu_lambda_init = as.numeric(mu_lambda_init),
    mu_p_init = as.numeric(mu_p_init),
    K_max = K_max, max_iter = as.integer(max_iter), tol = as.numeric(tol),
    inner_max = as.integer(inner_max), inner_tol = as.numeric(inner_tol),
    sigma_beta = as.numeric(sigma_beta), verbose = isTRUE(verbose))
  raw$mixture <- "P"
  class(raw) <- c("tulpa_nmix_re_fit", "list")
  raw
}

# Method-of-moments community covariance seed: the sample covariance of the
# per-species coefficient estimates, ridge-regularized to PD.
.nmix_re_cov0 <- function(B, p) {
  if (is.null(B) || nrow(B) < 2L) return(diag(0.25, p))
  V <- stats::cov(B)
  V[!is.finite(V)] <- 0
  V + diag(max(1e-3, 1e-3 * mean(abs(diag(V)))), p)
}
