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
#' random effects. Both solvers assemble the per-species marginal in compiled
#' code via a native oracle (no per-group round trip into R). At `n_quad = 1`
#' (the default, the joint Laplace / glmer `nAGQ = 1`) the fit is a Laplace-EM
#' (block-coordinate Newton mode + closed-form covariance M-step + Schur-
#' complement SEs) -- the fast production path. A higher `n_quad` routes the same
#' native oracle through the shared compiled AGHQ engine ([tulpa_re_aghq()]),
#' replacing each species' Laplace integral with adaptive Gauss-Hermite
#' quadrature to reduce the small-cluster (few species) downward bias of the
#' community covariances `Sigma_lambda` / `Sigma_p` -- at a
#' `n_quad^(p_lambda + p_p)` per-species grid cost. Each species'
#' marginal -- value, abundance/detection score, and the per-site observed-
#' information block carrying the \eqn{\mathrm{Var}(N_i\mid y_i)} abundance/
#' detection coupling -- is supplied as the per-group oracle, so there is one
#' marginal/quadrature/covariance implementation across the package. The
#' Gaussian community priors pin \eqn{(\mu_\lambda, \mu_p)} as fixed effects, so
#' no sum-to-zero constraint is needed; their standard errors come from the
#' marginal observed-information Hessian.
#'
#' The abundance mixing distribution is Poisson (`mixture = "P"`) or negative
#' binomial (`mixture = "NB"`, a single global dispersion size `r` shared across
#' species, carried as an extra fixed effect and integrated jointly with the
#' community means). Poisson is the `r -> Inf` limit.
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
#' @param optimizer Outer optimize driver over the shared native oracle:
#'   `"em"` (default) is the fast Laplace-EM (block-coordinate Newton mode +
#'   closed-form covariance M-step + Schur SE), the exact `n_quad = 1` solver.
#'   `"joint_grad"` and `"joint_fd"` are the joint `(theta, log-Cholesky Sigma)`
#'   optimizers ([tulpa_re_aghq()]) and both do the `n_quad > 1` AGHQ debias of
#'   the community covariances. `"joint_grad"` (the fast debias path) supplies
#'   the analytic Fisher-identity gradient -- one group sweep per step, no
#'   per-coordinate objective re-solve -- and requires `n_quad > 1` (at
#'   `n_quad = 1` use the EM). `"joint_fd"` finite-differences the objective
#'   (slower; the FD sweep re-solves every per-species mode per coordinate) and
#'   is kept for correctness / architecture validation and as the `n_quad = 1`
#'   joint reference.
#' @param mixture Abundance mixing distribution: `"P"` (Poisson, default) or
#'   `"NB"` (negative binomial, one global dispersion size `r`). `"NB"` has no
#'   closed-form EM, so it defaults `optimizer` to `"joint_grad"` and `n_quad`
#'   to `5` when those are not supplied, and errors on `optimizer = "em"`.
#' @param r_init Initial negative-binomial size for the joint optimizer
#'   (`mixture = "NB"` only; default `10`, a moderate overdispersion start). The
#'   optimizer carries `log_r` as the `(p_lambda + p_p + 1)`-th fixed effect.
#' @param n_quad Quadrature points per random-effect dimension passed to
#'   [tulpa_re_aghq()] (default 1, the joint Laplace). A higher `n_quad`
#'   debiases the community covariances at a `n_quad^(p_lambda + p_p)`
#'   per-species grid cost, so keep it modest when the coefficient dimension is
#'   large.
#' @param lkj_eta LKJ shape regularizing each *correlated* community covariance
#'   block's correlation off the boundary (default 1, no penalty); passed
#'   through to [tulpa_re_aghq()]. Does not touch the marginal SDs.
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
#'   `n_species` rows), `log_lik` (AGHQ marginal), `converged`, `K_max`,
#'   `n_quad`, `lkj_eta`, and (when `mixture = "NB"`) `r`, the fitted global
#'   negative-binomial size (its `log_r` SE is in `vcov`).
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
                                  optimizer = c("em", "joint_fd", "joint_grad"),
                                  mixture = c("P", "NB"), r_init = 10,
                                  n_quad = 1L, lkj_eta = 1,
                                  sigma_beta = 100, verbose = FALSE) {
  mixture <- match.arg(mixture)
  # NB has no closed-form EM (that is the Poisson Laplace special case): the
  # global dispersion log_r is an extra fixed effect the joint optimizer carries.
  # So NB defaults to the analytic-gradient debias path, which (like any
  # joint_grad) needs n_quad > 1. Explicit optimizer = "em" with NB is an error.
  if (mixture == "NB" && missing(optimizer)) optimizer <- "joint_grad"
  if (mixture == "NB" && missing(n_quad))    n_quad <- 5L
  optimizer <- match.arg(optimizer)
  if (mixture == "NB" && optimizer == "em") {
    stop("mixture = \"NB\" needs a joint optimizer (\"joint_grad\" or ",
         "\"joint_fd\"); the EM is the Poisson Laplace solver.", call. = FALSE)
  }
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

  # One native oracle (NMixCommunityOracle) is the shared backend: it assembles
  # the per-species marginal / score / observed-info / complete-data Fisher in
  # C++, and BOTH optimize drivers consume it -- there is no second marginal
  # source. The driver is selected by `optimizer`.
  n_quad <- as.integer(n_quad)
  orc <- cpp_nmix_community_oracle(y, site_idx, species_idx, X_lambda, X_p,
                                   n_sites, n_species, K_max,
                                   nb = (mixture == "NB"))

  if (optimizer == "em") {
    # Default. EM is an outer driver over the shared oracle: block-coordinate
    # Newton mode (complete-data Fisher) + closed-form Sigma M-step + Schur SE.
    # It reaches the same n_quad = 1 (joint Laplace) stationary point as the
    # joint optimizer (agq_plan.md 4.3) without the FD-gradient objective sweep
    # that dominates joint_fd's runtime, so it is the production path. Quadrature
    # (n_quad > 1) is a joint-driver feature; the EM is Laplace (n_quad = 1).
    if (n_quad != 1L) {
      stop("optimizer = \"em\" is the n_quad = 1 (Laplace) solver; AGHQ ",
           "(n_quad > 1) needs optimizer = \"joint_fd\".", call. = FALSE)
    }
    em <- cpp_nmix_community_em(
      orc, mu_init = c(as.numeric(mu_lambda_init), as.numeric(mu_p_init)),
      Sigma_lambda_init = as.matrix(Sigma_lambda_init),
      Sigma_p_init      = as.matrix(Sigma_p_init),
      max_iter = as.integer(max_iter), tol = 1e-6,
      inner_max = 50L, inner_tol = 1e-8, sigma_beta = sigma_beta,
      verbose = isTRUE(verbose))
    out <- list(
      mu_lambda = as.numeric(em$mu_lambda), mu_p = as.numeric(em$mu_p),
      vcov = em$vcov, Sigma_lambda = em$Sigma_lambda, Sigma_p = em$Sigma_p,
      b_lambda = em$b_lambda, b_p = em$b_p, log_lik = em$log_lik,
      converged = isTRUE(em$converged), K_max = K_max,
      n_quad = 1L, lkj_eta = lkj_eta, optimizer = "em", mixture = "P")
    class(out) <- c("tulpa_nmix_re_fit", "list")
    return(out)
  }

  # Joint optimizers (joint_fd / joint_grad) drive the SAME native oracle through
  # the shared compiled AGHQ engine (tulpa_re_aghq) by quadrature, and both do the
  # n_quad > 1 variance-component debias the EM cannot.
  #   joint_grad: the analytic Fisher-identity gradient (cpp_aghq_objective_grad)
  #     -- one group sweep per optim step, no per-coordinate objective re-solve.
  #     This is the fast production debias path. It needs n_quad > 1: the analytic
  #     gradient omits the Laplace curvature term, so at n_quad = 1 it disagrees
  #     with the objective (use the EM, which is the exact n_quad = 1 solver).
  #   joint_fd: optim finite-differences the objective -- the FD sweep re-solves
  #     every per-species mode for each perturbed coordinate (~100% of the
  #     residual runtime), so it is slower; kept for correctness / architecture
  #     validation and as the n_quad = 1 joint reference.
  if (optimizer == "joint_grad" && n_quad <= 1L) {
    stop("optimizer = \"joint_grad\" is the analytic-gradient AGHQ debias path ",
         "(n_quad > 1); at n_quad = 1 the analytic gradient omits the Laplace ",
         "curvature term. Use optimizer = \"em\" (the exact n_quad = 1 solver).",
         call. = FALSE)
  }
  grad_mode <- if (optimizer == "joint_grad") "analytic" else "fd"
  # NB carries a global dispersion log_r as the (d+1)-th theta entry (the oracle
  # built with nb = TRUE exposes n_theta = d + 1); Poisson omits it.
  theta0 <- c(as.numeric(mu_lambda_init), as.numeric(mu_p_init))
  if (mixture == "NB") theta0 <- c(theta0, log(r_init))
  fit <- tulpa_re_aghq(
    theta0  = theta0,
    re_terms = list(list(n_groups = n_species, n_coefs = p_lam, correlated = p_lam > 1L),
                    list(n_groups = n_species, n_coefs = p_p,   correlated = p_p   > 1L)),
    Sigma0  = list(as.matrix(Sigma_lambda_init), as.matrix(Sigma_p_init)),
    oracle  = orc, gradient = grad_mode,
    n_quad = n_quad, lkj_eta = lkj_eta,
    theta_prior_sd = sigma_beta, maxit = as.integer(max_iter))

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
    n_quad       = fit$n_quad,
    lkj_eta      = fit$lkj_eta,
    optimizer    = optimizer,
    mixture      = mixture
  )
  # NB: the (d+1)-th theta entry is log_r; report the size r (its log-scale SE is
  # the corresponding marginal-Hessian diagonal in `vcov`).
  if (mixture == "NB") out$r <- exp(mu[p_lam + p_p + 1L])
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
