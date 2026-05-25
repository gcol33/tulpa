#' Spatial Royle (2004) N-mixture model via nested Laplace
#'
#' @description
#' Nested-Laplace fit of the spatial N-mixture model
#' \deqn{N_i \sim \mathrm{Poisson}(\lambda_i), \qquad
#'       y_{ij} | N_i \sim \mathrm{Binomial}(N_i, p_{ij}),}
#' with abundance linear predictor
#' \eqn{\log \lambda_i = X_\lambda^{(i)} \beta_\lambda + z_{u(i)}} where
#' \eqn{z \sim \mathrm{ICAR}(\tau)} is an intrinsic conditional autoregressive
#' field on the user-supplied adjacency graph, and detection linear predictor
#' \eqn{\mathrm{logit}\, p_{ij} = X_p^{(ij)} \beta_p}.
#'
#' The hyperparameter \eqn{\tau} (ICAR precision) is integrated out by an
#' outer grid: at each \eqn{\tau_k} the inner Newton finds the joint mode
#' of \eqn{(\beta_\lambda, \beta_p, z)} and the Laplace log-marginal
#' \eqn{\log p(y \mid \tau_k)} is accumulated. Posterior weights over
#' \eqn{\tau} normalise the grid.
#'
#' Inner Newton uses the marginal observed Fisher information matrix for
#' curvature (with the Var\eqn{[N \mid y_i]} rank-1 correction encoding
#' cross-arm coupling) and falls back to the complete-data Fisher block when
#' the observed-info matrix is not PSD. A small diagonal ridge keeps the
#' Cholesky stable through the (intercept, constant-\eqn{z}) structural null
#' direction; \eqn{z} is centered to sum zero after every step.
#'
#' @param y Integer vector of observed counts (long form, one entry per visit).
#' @param site_idx Integer vector, 1-based site index for each visit, same
#'   length as `y`.
#' @param map_site_to_unit Integer vector of length `n_sites`, 1-based spatial
#'   unit index for each site. Sites can share spatial units; the data
#'   contribution to `z[u]` aggregates across all sites that map to `u`.
#' @param X_lambda Numeric matrix `[n_sites x p_lambda]` of abundance covariates.
#' @param X_p Numeric matrix `[n_obs x p_p]` of detection covariates.
#' @param adj_row_ptr,adj_col_idx,n_neighbors CSR adjacency for the ICAR graph
#'   on the `n_spatial` units. `adj_row_ptr` has length `n_spatial + 1`,
#'   `adj_col_idx` lists 0-based neighbours, and `n_neighbors[s]` is the row
#'   degree of unit s.
#' @param n_spatial Number of spatial units.
#' @param tau_grid Optional numeric vector of \eqn{\tau} grid points. Defaults
#'   to `exp(seq(log(0.3), log(30), length.out = 9))`.
#' @param beta_lambda_init Optional warm start; default `c(log(mean(y)+0.1), 0, ...)`.
#' @param beta_p_init Optional warm start; default `rep(0, p_p)`.
#' @param z_init Optional warm start for the spatial field; default zeros.
#' @param K_max Truncation for the per-site marginal sum over N. Defaults to
#'   `max(y) + 100`. Returned `boundary_max` flags any grid point whose worst
#'   site puts non-trivial mass on `K_max` -- raise `K_max` if it exceeds 1e-4.
#' @param max_iter,tol Inner Newton iteration budget and gradient-norm tol.
#' @param verbose Print per-iteration and per-grid-point progress.
#'
#' @return A list of class `tulpa_nmix_spatial_fit`:
#'   * `tau_grid` -- input grid
#'   * `log_marginal` -- log marginal at each tau (up to a tau-independent constant)
#'   * `weights` -- normalised grid weights (sum to 1)
#'   * `tau_mean`, `tau_sd` -- posterior moments of tau
#'   * `modes` -- `[n_grid x (p_lambda + p_p + n_spatial)]` matrix of inner modes
#'   * `beta_lambda_mean`, `beta_p_mean` -- weighted-mean coefficient estimates
#'   * `z_mean` -- weighted-mean spatial field
#'   * `n_iter`, `converged`, `grad_norm`, `log_lik`, `boundary_max` -- per-grid diagnostics
#'   * `p_lambda`, `p_p`, `n_spatial`, `K_max` -- echoed dimensions
#'   * `call` -- matched call
#'
#' @references
#' Royle, J. A. (2004). N-mixture models for estimating population size from
#'   spatially replicated counts. *Biometrics* 60, 108-115.
#' Besag, J., York, J., Mollie, A. (1991). Bayesian image restoration with two
#'   applications in spatial statistics. *Ann. Inst. Statist. Math.* 43, 1-20.
#' Rue, H., Martino, S., Chopin, N. (2009). Approximate Bayesian inference for
#'   latent Gaussian models by using integrated nested Laplace approximations.
#'   *JRSS-B* 71, 319-392.
#'
#' @export
tulpa_nmix_laplace_icar <- function(y,
                                    site_idx,
                                    map_site_to_unit,
                                    X_lambda,
                                    X_p,
                                    adj_row_ptr,
                                    adj_col_idx,
                                    n_neighbors,
                                    n_spatial,
                                    tau_grid = NULL,
                                    beta_lambda_init = NULL,
                                    beta_p_init = NULL,
                                    z_init = NULL,
                                    K_max = NULL,
                                    max_iter = 100L,
                                    tol = 1e-6,
                                    verbose = FALSE) {
  y                <- as.integer(y)
  site_idx         <- as.integer(site_idx)
  map_site_to_unit <- as.integer(map_site_to_unit)
  if (!is.matrix(X_lambda)) stop("`X_lambda` must be a numeric matrix.", call. = FALSE)
  if (!is.matrix(X_p))      stop("`X_p` must be a numeric matrix.", call. = FALSE)
  n_sites <- nrow(X_lambda)
  n_obs   <- nrow(X_p)
  p_lam   <- ncol(X_lambda)
  p_p     <- ncol(X_p)
  if (length(y) != n_obs) stop("length(y) must equal nrow(X_p).", call. = FALSE)
  if (length(site_idx) != n_obs) stop("length(site_idx) must equal nrow(X_p).", call. = FALSE)
  if (length(map_site_to_unit) != n_sites) {
    stop("length(map_site_to_unit) must equal nrow(X_lambda).", call. = FALSE)
  }
  if (any(map_site_to_unit < 1L) || any(map_site_to_unit > n_spatial)) {
    stop("map_site_to_unit values must lie in [1, n_spatial].", call. = FALSE)
  }
  if (length(adj_row_ptr) != n_spatial + 1L) {
    stop("length(adj_row_ptr) must equal n_spatial + 1.", call. = FALSE)
  }
  if (length(n_neighbors) != n_spatial) {
    stop("length(n_neighbors) must equal n_spatial.", call. = FALSE)
  }
  if (is.null(tau_grid)) {
    tau_grid <- exp(seq(log(0.3), log(30), length.out = 9L))
  }
  if (is.null(beta_lambda_init)) {
    beta_lambda_init <- c(log(mean(y) + 0.1), rep(0, p_lam - 1L))
  }
  if (is.null(beta_p_init)) beta_p_init <- rep(0, p_p)
  if (length(beta_lambda_init) != p_lam) {
    stop("length(beta_lambda_init) must equal ncol(X_lambda).", call. = FALSE)
  }
  if (length(beta_p_init) != p_p) {
    stop("length(beta_p_init) must equal ncol(X_p).", call. = FALSE)
  }
  if (is.null(K_max)) {
    K_max <- as.integer(max(y) + 100L)
  } else {
    K_max <- as.integer(K_max)
    if (K_max < max(y)) stop("K_max must be >= max(y).", call. = FALSE)
  }
  if (!is.null(z_init) && length(z_init) != n_spatial) {
    stop("length(z_init) must equal n_spatial.", call. = FALSE)
  }

  fit <- cpp_nested_laplace_nmix_icar(
    y                  = y,
    site_idx           = site_idx,
    map_site_to_unit_R = map_site_to_unit,
    X_lambda_R         = X_lambda,
    X_p_R              = X_p,
    adj_row_ptr        = as.integer(adj_row_ptr),
    adj_col_idx        = as.integer(adj_col_idx),
    n_neighbors        = as.integer(n_neighbors),
    n_spatial          = as.integer(n_spatial),
    tau_grid           = as.numeric(tau_grid),
    beta_lambda_init   = as.numeric(beta_lambda_init),
    beta_p_init        = as.numeric(beta_p_init),
    z_init             = if (is.null(z_init)) NULL else as.numeric(z_init),
    K_max              = K_max,
    max_iter           = as.integer(max_iter),
    tol                = as.numeric(tol),
    verbose            = isTRUE(verbose)
  )

  # Normalise grid weights.
  lm <- fit$log_marginal
  finite_lm <- lm[is.finite(lm)]
  if (length(finite_lm) == 0L) {
    warning("All grid points returned non-finite log_marginal -- check tau_grid / data.",
            call. = FALSE)
    weights <- rep(NA_real_, length(lm))
  } else {
    m <- max(finite_lm)
    w <- exp(lm - m)
    w[!is.finite(w)] <- 0
    if (sum(w) == 0) {
      weights <- rep(NA_real_, length(lm))
    } else {
      weights <- w / sum(w)
    }
  }

  tau_mean <- sum(weights * tau_grid, na.rm = TRUE)
  tau_sd   <- sqrt(max(0, sum(weights * tau_grid^2, na.rm = TRUE) - tau_mean^2))

  # Posterior-mean coefficients (weighted across grid points).
  modes <- fit$modes
  beta_lambda_mean <- as.numeric(crossprod(weights, modes[, seq_len(p_lam), drop = FALSE]))
  beta_p_mean      <- as.numeric(crossprod(
    weights, modes[, p_lam + seq_len(p_p), drop = FALSE]
  ))
  z_mean <- as.numeric(crossprod(
    weights, modes[, p_lam + p_p + seq_len(n_spatial), drop = FALSE]
  ))

  nm_lam <- colnames(X_lambda)
  nm_p   <- colnames(X_p)
  if (is.null(nm_lam)) nm_lam <- paste0("lam_", seq_len(p_lam))
  if (is.null(nm_p))   nm_p   <- paste0("p_", seq_len(p_p))
  names(beta_lambda_mean) <- nm_lam
  names(beta_p_mean)      <- nm_p

  out <- c(fit, list(
    weights          = weights,
    tau_mean         = tau_mean,
    tau_sd           = tau_sd,
    beta_lambda_mean = beta_lambda_mean,
    beta_p_mean      = beta_p_mean,
    vcov             = .nmix_grid_vcov(fit$cov_blocks, modes, weights,
                                       p_lam, p_p, c(nm_lam, nm_p)),
    z_mean           = z_mean,
    n_sites          = n_sites,
    n_obs            = n_obs,
    prior_type       = "icar",
    call             = match.call()
  ))
  if (any(out$boundary_max > 1e-4, na.rm = TRUE)) {
    warning(sprintf(
      "Max posterior weight on N = K_max is %.2e at one or more grid points; raise K_max.",
      max(out$boundary_max, na.rm = TRUE)
    ), call. = FALSE)
  }
  class(out) <- c("tulpa_nmix_spatial_fit", "list")
  out
}

#' Proper CAR Royle (2004) N-mixture model via nested Laplace
#'
#' @description
#' Nested-Laplace fit of the spatial N-mixture model with a proper conditional
#' autoregressive prior on the abundance-arm spatial offset:
#' \deqn{N_i \sim \mathrm{Poisson}(\lambda_i), \qquad
#'       y_{ij} | N_i \sim \mathrm{Binomial}(N_i, p_{ij}),}
#' \eqn{\log \lambda_i = X_\lambda^{(i)} \beta_\lambda + z_{u(i)}}, where
#' \eqn{z \mid \tau, \rho \sim \mathrm{N}(0, [\tau (D - \rho W)]^{-1})}.
#' Both hyperparameters are integrated over an outer 2D grid; the inner
#' Newton step shares the kernel with [tulpa_nmix_laplace_icar()] (ICAR is
#' the \eqn{\rho = 1} limit).
#'
#' Unlike the ICAR fit, no sum-to-zero centering is applied -- \eqn{Q(\rho)}
#' is full rank for \eqn{\rho < 1}. The per-rho \eqn{\log |Q(\rho)|} is
#' precomputed once via a dense Cholesky on the \eqn{n_{\mathrm{spatial}}
#' \times n_{\mathrm{spatial}}} precision matrix.
#'
#' @inheritParams tulpa_nmix_laplace_icar
#' @param tau_grid Optional numeric vector of \eqn{\tau} grid points
#'   (defaults to `exp(seq(log(0.3), log(30), length.out = 7L))`).
#' @param rho_grid Optional numeric vector of \eqn{\rho} grid points in
#'   the valid eigenvalue interval. Defaults to a 5-point grid in
#'   \eqn{(0, 1)} -- callers that want eigenvalue-derived bounds should
#'   compute them via tulpa::spatial_car_proper(adjacency)`$`rho_bounds and
#'   pass an explicit grid in that interval.
#'
#' @return A list of class `tulpa_nmix_spatial_fit`:
#'   * `theta_grid` -- `[n_grid x 2]` matrix of (tau, rho) per grid point
#'   * `tau_grid`, `rho_grid` -- input axes
#'   * `log_det_Q_rho` -- precomputed log determinants per rho
#'   * `log_marginal`, `weights`, `tau_mean`, `tau_sd`, `rho_mean`, `rho_sd`
#'     -- as in [tulpa_nmix_laplace_icar()] but with the rho marginal added
#'   * other diagnostic fields and named coefficient means as in ICAR
#'
#' @references
#' Cressie, N. (1993). Statistics for Spatial Data. Wiley.
#' Rue, H., Held, L. (2005). Gaussian Markov Random Fields. CRC.
#'
#' @export
tulpa_nmix_laplace_car_proper <- function(y,
                                          site_idx,
                                          map_site_to_unit,
                                          X_lambda,
                                          X_p,
                                          adj_row_ptr,
                                          adj_col_idx,
                                          n_neighbors,
                                          n_spatial,
                                          tau_grid = NULL,
                                          rho_grid = NULL,
                                          beta_lambda_init = NULL,
                                          beta_p_init = NULL,
                                          z_init = NULL,
                                          K_max = NULL,
                                          max_iter = 100L,
                                          tol = 1e-6,
                                          verbose = FALSE) {
  y                <- as.integer(y)
  site_idx         <- as.integer(site_idx)
  map_site_to_unit <- as.integer(map_site_to_unit)
  if (!is.matrix(X_lambda)) stop("`X_lambda` must be a numeric matrix.", call. = FALSE)
  if (!is.matrix(X_p))      stop("`X_p` must be a numeric matrix.", call. = FALSE)
  n_sites <- nrow(X_lambda)
  n_obs   <- nrow(X_p)
  p_lam   <- ncol(X_lambda)
  p_p     <- ncol(X_p)
  if (length(y) != n_obs) stop("length(y) must equal nrow(X_p).", call. = FALSE)
  if (length(site_idx) != n_obs) stop("length(site_idx) must equal nrow(X_p).", call. = FALSE)
  if (length(map_site_to_unit) != n_sites) {
    stop("length(map_site_to_unit) must equal nrow(X_lambda).", call. = FALSE)
  }
  if (any(map_site_to_unit < 1L) || any(map_site_to_unit > n_spatial)) {
    stop("map_site_to_unit values must lie in [1, n_spatial].", call. = FALSE)
  }
  if (length(adj_row_ptr) != n_spatial + 1L) {
    stop("length(adj_row_ptr) must equal n_spatial + 1.", call. = FALSE)
  }
  if (length(n_neighbors) != n_spatial) {
    stop("length(n_neighbors) must equal n_spatial.", call. = FALSE)
  }
  if (is.null(tau_grid)) {
    tau_grid <- exp(seq(log(0.3), log(30), length.out = 7L))
  }
  if (is.null(rho_grid)) {
    rho_grid <- c(0.1, 0.3, 0.5, 0.75, 0.95)
  }
  if (any(rho_grid <= 0) || any(rho_grid >= 1)) {
    stop("rho_grid values must lie strictly in (0, 1) for the default ",
         "eigenvalue interval. Pass explicit bounds via spatial_car_proper().",
         call. = FALSE)
  }
  if (is.null(beta_lambda_init)) {
    beta_lambda_init <- c(log(mean(y) + 0.1), rep(0, p_lam - 1L))
  }
  if (is.null(beta_p_init)) beta_p_init <- rep(0, p_p)
  if (length(beta_lambda_init) != p_lam) {
    stop("length(beta_lambda_init) must equal ncol(X_lambda).", call. = FALSE)
  }
  if (length(beta_p_init) != p_p) {
    stop("length(beta_p_init) must equal ncol(X_p).", call. = FALSE)
  }
  if (is.null(K_max)) {
    K_max <- as.integer(max(y) + 100L)
  } else {
    K_max <- as.integer(K_max)
    if (K_max < max(y)) stop("K_max must be >= max(y).", call. = FALSE)
  }
  if (!is.null(z_init) && length(z_init) != n_spatial) {
    stop("length(z_init) must equal n_spatial.", call. = FALSE)
  }

  fit <- cpp_nested_laplace_nmix_car_proper(
    y                  = y,
    site_idx           = site_idx,
    map_site_to_unit_R = map_site_to_unit,
    X_lambda_R         = X_lambda,
    X_p_R              = X_p,
    adj_row_ptr        = as.integer(adj_row_ptr),
    adj_col_idx        = as.integer(adj_col_idx),
    n_neighbors        = as.integer(n_neighbors),
    n_spatial          = as.integer(n_spatial),
    tau_grid           = as.numeric(tau_grid),
    rho_grid           = as.numeric(rho_grid),
    beta_lambda_init   = as.numeric(beta_lambda_init),
    beta_p_init        = as.numeric(beta_p_init),
    z_init             = if (is.null(z_init)) NULL else as.numeric(z_init),
    K_max              = K_max,
    max_iter           = as.integer(max_iter),
    tol                = as.numeric(tol),
    verbose            = isTRUE(verbose)
  )

  # Normalise grid weights (joint over the (tau, rho) grid).
  lm <- fit$log_marginal
  finite_lm <- lm[is.finite(lm)]
  if (length(finite_lm) == 0L) {
    warning("All grid points returned non-finite log_marginal -- check grids / data.",
            call. = FALSE)
    weights <- rep(NA_real_, length(lm))
  } else {
    m <- max(finite_lm)
    w <- exp(lm - m)
    w[!is.finite(w)] <- 0
    if (sum(w) == 0) {
      weights <- rep(NA_real_, length(lm))
    } else {
      weights <- w / sum(w)
    }
  }

  tau_vec <- fit$theta_grid[, 1]
  rho_vec <- fit$theta_grid[, 2]
  tau_mean <- sum(weights * tau_vec, na.rm = TRUE)
  tau_sd   <- sqrt(max(0, sum(weights * tau_vec^2, na.rm = TRUE) - tau_mean^2))
  rho_mean <- sum(weights * rho_vec, na.rm = TRUE)
  rho_sd   <- sqrt(max(0, sum(weights * rho_vec^2, na.rm = TRUE) - rho_mean^2))

  modes <- fit$modes
  beta_lambda_mean <- as.numeric(crossprod(weights, modes[, seq_len(p_lam), drop = FALSE]))
  beta_p_mean      <- as.numeric(crossprod(
    weights, modes[, p_lam + seq_len(p_p), drop = FALSE]
  ))
  z_mean <- as.numeric(crossprod(
    weights, modes[, p_lam + p_p + seq_len(n_spatial), drop = FALSE]
  ))

  nm_lam <- colnames(X_lambda)
  nm_p   <- colnames(X_p)
  if (is.null(nm_lam)) nm_lam <- paste0("lam_", seq_len(p_lam))
  if (is.null(nm_p))   nm_p   <- paste0("p_", seq_len(p_p))
  names(beta_lambda_mean) <- nm_lam
  names(beta_p_mean)      <- nm_p

  out <- c(fit, list(
    weights          = weights,
    tau_mean         = tau_mean,
    tau_sd           = tau_sd,
    rho_mean         = rho_mean,
    rho_sd           = rho_sd,
    beta_lambda_mean = beta_lambda_mean,
    beta_p_mean      = beta_p_mean,
    vcov             = .nmix_grid_vcov(fit$cov_blocks, modes, weights,
                                       p_lam, p_p, c(nm_lam, nm_p)),
    z_mean           = z_mean,
    n_sites          = n_sites,
    n_obs            = n_obs,
    prior_type       = "car_proper",
    call             = match.call()
  ))
  if (any(out$boundary_max > 1e-4, na.rm = TRUE)) {
    warning(sprintf(
      "Max posterior weight on N = K_max is %.2e at one or more grid points; raise K_max.",
      max(out$boundary_max, na.rm = TRUE)
    ), call. = FALSE)
  }
  class(out) <- c("tulpa_nmix_spatial_fit", "list")
  out
}

#' BYM2 Royle (2004) N-mixture model via nested Laplace
#'
#' @description
#' Nested-Laplace fit of the spatial N-mixture model with a BYM2 (Riebler et al.
#' 2016) prior on the abundance-arm spatial offset. The offset decomposes as
#' \deqn{\phi_u = \sigma \left(\sqrt{\rho / s} \, v_u + \sqrt{1 - \rho} \, w_u\right),}
#' with \eqn{v \sim \mathrm{ICAR}} (unscaled, sum-to-zero) and
#' \eqn{w \sim \mathrm{N}(0, I)} iid. \eqn{s} is the geometric mean of the
#' non-zero eigenvalues of the ICAR precision \eqn{Q} (the Riebler scaling
#' factor); \eqn{\sigma} is then the joint marginal standard deviation of
#' \eqn{\phi}, and \eqn{\rho \in [0, 1]} is the spatial fraction of variance.
#'
#' The inner Newton works in the joint state
#' \eqn{x = (\beta_\lambda, \beta_p, v, w)} (dimension
#' \eqn{p_\lambda + p_p + 2 n_{\mathrm{spatial}}}). At the converged mode the
#' Laplace log-marginal is accumulated; the outer 2D grid integrates over
#' \eqn{(\sigma, \rho)}.
#'
#' @inheritParams tulpa_nmix_laplace_icar
#' @param sigma_grid Numeric vector of \eqn{\sigma} (joint sd) grid points.
#'   Defaults to `exp(seq(log(0.2), log(3), length.out = 5L))`.
#' @param rho_grid Numeric vector of spatial-fraction grid points in
#'   \eqn{[0, 1]}. Defaults to `c(0.05, 0.3, 0.5, 0.7, 0.95)`.
#' @param scale_factor Optional scalar Riebler scaling factor (geometric mean
#'   of non-zero eigenvalues of \eqn{Q}). If `NULL`, it is computed from the
#'   adjacency via dense eigendecomposition.
#' @param v_init,w_init Optional warm starts (each length `n_spatial`) for the
#'   two BYM2 latent components -- the structured (spatial) and unstructured
#'   (iid) effects. `NULL` (default) starts both at zero.
#'
#' @return A list of class `tulpa_nmix_spatial_fit`:
#'   * `theta_grid` -- `[n_grid x 2]` matrix of (sigma, rho) per grid point
#'   * `sigma_grid`, `rho_grid` -- input axes
#'   * `scale_factor` -- the Riebler scaling factor
#'   * `log_marginal`, `weights`, `sigma_mean`, `sigma_sd`, `rho_mean`, `rho_sd`
#'     -- posterior summaries of the joint hyperparameters
#'   * `modes` -- `[n_grid x (p_lambda + p_p + 2 n_spatial)]` per-grid modes;
#'     columns `(p_lambda + p_p + 1) .. (p_lambda + p_p + n_spatial)` are `v`,
#'     the next `n_spatial` columns are `w`
#'   * `beta_lambda_mean`, `beta_p_mean` -- weighted-mean coefficients
#'   * `v_mean`, `w_mean` -- weighted-mean ICAR and iid components
#'   * `phi_mean` -- weighted-mean total offset \eqn{\phi}
#'   * other diagnostic fields as in [tulpa_nmix_laplace_icar()]
#'
#' @references
#' Riebler, A., Sorbye, S. H., Simpson, D., Rue, H. (2016). An intuitive
#'   Bayesian spatial model for disease mapping that accounts for scaling.
#'   *Statistical Methods in Medical Research* 25, 1145-1165.
#' Morris, M., Wheeler-Martin, K., Simpson, D., Mooney, S. J., Gelman, A.,
#'   DiMaggio, C. (2019). Bayesian hierarchical spatial models: Implementing
#'   the BYM2 model in Stan. *Spatial and Spatio-temporal Epidemiology* 31.
#'
#' @export
tulpa_nmix_laplace_bym2 <- function(y,
                                    site_idx,
                                    map_site_to_unit,
                                    X_lambda,
                                    X_p,
                                    adj_row_ptr,
                                    adj_col_idx,
                                    n_neighbors,
                                    n_spatial,
                                    sigma_grid = NULL,
                                    rho_grid = NULL,
                                    scale_factor = NULL,
                                    beta_lambda_init = NULL,
                                    beta_p_init = NULL,
                                    v_init = NULL,
                                    w_init = NULL,
                                    K_max = NULL,
                                    max_iter = 100L,
                                    tol = 1e-6,
                                    verbose = FALSE) {
  y                <- as.integer(y)
  site_idx         <- as.integer(site_idx)
  map_site_to_unit <- as.integer(map_site_to_unit)
  if (!is.matrix(X_lambda)) stop("`X_lambda` must be a numeric matrix.", call. = FALSE)
  if (!is.matrix(X_p))      stop("`X_p` must be a numeric matrix.", call. = FALSE)
  n_sites <- nrow(X_lambda)
  n_obs   <- nrow(X_p)
  p_lam   <- ncol(X_lambda)
  p_p     <- ncol(X_p)
  if (length(y) != n_obs) stop("length(y) must equal nrow(X_p).", call. = FALSE)
  if (length(site_idx) != n_obs) stop("length(site_idx) must equal nrow(X_p).", call. = FALSE)
  if (length(map_site_to_unit) != n_sites) {
    stop("length(map_site_to_unit) must equal nrow(X_lambda).", call. = FALSE)
  }
  if (any(map_site_to_unit < 1L) || any(map_site_to_unit > n_spatial)) {
    stop("map_site_to_unit values must lie in [1, n_spatial].", call. = FALSE)
  }
  if (length(adj_row_ptr) != n_spatial + 1L) {
    stop("length(adj_row_ptr) must equal n_spatial + 1.", call. = FALSE)
  }
  if (length(n_neighbors) != n_spatial) {
    stop("length(n_neighbors) must equal n_spatial.", call. = FALSE)
  }
  if (is.null(sigma_grid)) {
    sigma_grid <- exp(seq(log(0.2), log(3), length.out = 5L))
  }
  if (is.null(rho_grid)) {
    rho_grid <- c(0.05, 0.3, 0.5, 0.7, 0.95)
  }
  if (any(sigma_grid <= 0)) stop("sigma_grid must be strictly positive.", call. = FALSE)
  if (any(rho_grid < 0) || any(rho_grid > 1)) {
    stop("rho_grid values must lie in [0, 1].", call. = FALSE)
  }
  if (is.null(scale_factor)) {
    # Build dense Q = D - W from CSR, compute geometric mean of non-zero
    # eigenvalues (Riebler scaling).
    W <- matrix(0, n_spatial, n_spatial)
    for (s in seq_len(n_spatial)) {
      a <- adj_row_ptr[s] + 1L
      b <- adj_row_ptr[s + 1L]
      if (b >= a) for (kk in a:b) {
        t <- adj_col_idx[kk] + 1L
        W[s, t] <- 1
      }
    }
    Q <- diag(as.numeric(n_neighbors)) - W
    eig <- eigen(Q, symmetric = TRUE, only.values = TRUE)$values
    nz  <- eig[abs(eig) > 1e-10]
    scale_factor <- exp(mean(log(nz)))
  }
  if (!is.numeric(scale_factor) || length(scale_factor) != 1L || scale_factor <= 0) {
    stop("scale_factor must be a positive scalar.", call. = FALSE)
  }
  if (is.null(beta_lambda_init)) {
    beta_lambda_init <- c(log(mean(y) + 0.1), rep(0, p_lam - 1L))
  }
  if (is.null(beta_p_init)) beta_p_init <- rep(0, p_p)
  if (length(beta_lambda_init) != p_lam) {
    stop("length(beta_lambda_init) must equal ncol(X_lambda).", call. = FALSE)
  }
  if (length(beta_p_init) != p_p) {
    stop("length(beta_p_init) must equal ncol(X_p).", call. = FALSE)
  }
  if (is.null(K_max)) {
    K_max <- as.integer(max(y) + 100L)
  } else {
    K_max <- as.integer(K_max)
    if (K_max < max(y)) stop("K_max must be >= max(y).", call. = FALSE)
  }
  if (!is.null(v_init) && length(v_init) != n_spatial) {
    stop("length(v_init) must equal n_spatial.", call. = FALSE)
  }
  if (!is.null(w_init) && length(w_init) != n_spatial) {
    stop("length(w_init) must equal n_spatial.", call. = FALSE)
  }

  fit <- cpp_nested_laplace_nmix_bym2(
    y                  = y,
    site_idx           = site_idx,
    map_site_to_unit_R = map_site_to_unit,
    X_lambda_R         = X_lambda,
    X_p_R              = X_p,
    adj_row_ptr        = as.integer(adj_row_ptr),
    adj_col_idx        = as.integer(adj_col_idx),
    n_neighbors        = as.integer(n_neighbors),
    n_spatial          = as.integer(n_spatial),
    sigma_grid         = as.numeric(sigma_grid),
    rho_grid           = as.numeric(rho_grid),
    scale_factor       = as.numeric(scale_factor),
    beta_lambda_init   = as.numeric(beta_lambda_init),
    beta_p_init        = as.numeric(beta_p_init),
    v_init             = if (is.null(v_init)) NULL else as.numeric(v_init),
    w_init             = if (is.null(w_init)) NULL else as.numeric(w_init),
    K_max              = K_max,
    max_iter           = as.integer(max_iter),
    tol                = as.numeric(tol),
    verbose            = isTRUE(verbose)
  )

  # Normalise grid weights (joint over (sigma, rho)).
  lm <- fit$log_marginal
  finite_lm <- lm[is.finite(lm)]
  if (length(finite_lm) == 0L) {
    warning("All grid points returned non-finite log_marginal -- check grids / data.",
            call. = FALSE)
    weights <- rep(NA_real_, length(lm))
  } else {
    m <- max(finite_lm)
    w <- exp(lm - m)
    w[!is.finite(w)] <- 0
    if (sum(w) == 0) {
      weights <- rep(NA_real_, length(lm))
    } else {
      weights <- w / sum(w)
    }
  }

  sigma_vec <- fit$theta_grid[, 1]
  rho_vec   <- fit$theta_grid[, 2]
  sigma_mean <- sum(weights * sigma_vec, na.rm = TRUE)
  sigma_sd   <- sqrt(max(0, sum(weights * sigma_vec^2, na.rm = TRUE) - sigma_mean^2))
  rho_mean   <- sum(weights * rho_vec, na.rm = TRUE)
  rho_sd     <- sqrt(max(0, sum(weights * rho_vec^2, na.rm = TRUE) - rho_mean^2))

  modes <- fit$modes
  beta_lambda_mean <- as.numeric(crossprod(weights, modes[, seq_len(p_lam), drop = FALSE]))
  beta_p_mean      <- as.numeric(crossprod(
    weights, modes[, p_lam + seq_len(p_p), drop = FALSE]
  ))
  v_idx <- p_lam + p_p + seq_len(n_spatial)
  w_idx <- p_lam + p_p + n_spatial + seq_len(n_spatial)
  v_mean <- as.numeric(crossprod(weights, modes[, v_idx, drop = FALSE]))
  w_mean <- as.numeric(crossprod(weights, modes[, w_idx, drop = FALSE]))

  # Weighted-mean total offset phi = sum_k w_k * sigma_k * (a_k v_k + b_k w_k).
  phi_grid <- matrix(0, nrow = nrow(modes), ncol = n_spatial)
  for (k in seq_len(nrow(modes))) {
    sg <- sigma_vec[k]; rg <- rho_vec[k]
    a_k <- sg * sqrt(rg / scale_factor)
    b_k <- sg * sqrt(1 - rg)
    phi_grid[k, ] <- a_k * modes[k, v_idx] + b_k * modes[k, w_idx]
  }
  phi_mean <- as.numeric(crossprod(weights, phi_grid))

  nm_lam <- colnames(X_lambda)
  nm_p   <- colnames(X_p)
  if (is.null(nm_lam)) nm_lam <- paste0("lam_", seq_len(p_lam))
  if (is.null(nm_p))   nm_p   <- paste0("p_", seq_len(p_p))
  names(beta_lambda_mean) <- nm_lam
  names(beta_p_mean)      <- nm_p

  out <- c(fit, list(
    weights          = weights,
    sigma_mean       = sigma_mean,
    sigma_sd         = sigma_sd,
    rho_mean         = rho_mean,
    rho_sd           = rho_sd,
    beta_lambda_mean = beta_lambda_mean,
    beta_p_mean      = beta_p_mean,
    vcov             = .nmix_grid_vcov(fit$cov_blocks, modes, weights,
                                       p_lam, p_p, c(nm_lam, nm_p)),
    v_mean           = v_mean,
    w_mean           = w_mean,
    phi_mean         = phi_mean,
    n_sites          = n_sites,
    n_obs            = n_obs,
    prior_type       = "bym2",
    call             = match.call()
  ))
  if (any(out$boundary_max > 1e-4, na.rm = TRUE)) {
    warning(sprintf(
      "Max posterior weight on N = K_max is %.2e at one or more grid points; raise K_max.",
      max(out$boundary_max, na.rm = TRUE)
    ), call. = FALSE)
  }
  class(out) <- c("tulpa_nmix_spatial_fit", "list")
  out
}

#' @export
print.tulpa_nmix_spatial_fit <- function(x, ...) {
  ptype <- x$prior_type %||% "icar"
  label <- switch(ptype,
                  icar       = "ICAR",
                  car_proper = "CAR(rho)",
                  bym2       = "BYM2",
                  toupper(ptype))
  cat(sprintf("tulpa spatial N-mixture (%s) nested-Laplace fit\n", label))
  cat(sprintf("  n_sites = %d   n_obs = %d   n_spatial = %d   K_max = %d\n",
              x$n_sites, x$n_obs, x$n_spatial, x$K_max))
  if (ptype == "icar") {
    cat(sprintf("  tau grid = [%.3g, %.3g] over %d points\n",
                min(x$tau_grid), max(x$tau_grid), x$n_grid))
    cat(sprintf("  tau_mean = %.3g   tau_sd = %.3g\n",
                x$tau_mean, x$tau_sd))
  } else if (ptype == "bym2") {
    cat(sprintf("  sigma grid = [%.3g, %.3g] x %d   rho grid = [%.3g, %.3g] x %d\n",
                min(x$sigma_grid), max(x$sigma_grid), length(x$sigma_grid),
                min(x$rho_grid), max(x$rho_grid), length(x$rho_grid)))
    cat(sprintf("  scale_factor = %.4g\n", x$scale_factor))
    cat(sprintf("  sigma_mean = %.3g   sigma_sd = %.3g\n",
                x$sigma_mean, x$sigma_sd))
    cat(sprintf("  rho_mean   = %.3g   rho_sd   = %.3g\n",
                x$rho_mean, x$rho_sd))
  } else {
    cat(sprintf("  tau grid = [%.3g, %.3g] x %d   rho grid = [%.3g, %.3g] x %d\n",
                min(x$tau_grid), max(x$tau_grid), length(x$tau_grid),
                min(x$rho_grid), max(x$rho_grid), length(x$rho_grid)))
    cat(sprintf("  tau_mean = %.3g   tau_sd = %.3g\n",
                x$tau_mean, x$tau_sd))
    if (!is.null(x$rho_mean)) {
      cat(sprintf("  rho_mean = %.3g   rho_sd = %.3g\n",
                  x$rho_mean, x$rho_sd))
    }
  }
  cat("\nabundance (log lambda):\n")
  print(x$beta_lambda_mean)
  cat("\ndetection (logit p):\n")
  print(x$beta_p_mean)
  invisible(x)
}

# Grid-integrated coefficient covariance via the law of total covariance:
#   V(beta) = sum_k w_k [ Cov_k + (m_k - mbar)(m_k - mbar)' ]
# where Cov_k = cov_blocks[[k]] is the within-grid Laplace covariance of
# beta = (beta_lambda, beta_p) at the k-th grid mode (the (beta) block of the
# joint H^{-1} returned by the C++ kernel), and the second term is the
# between-grid mode variance. The result is the marginal coefficient covariance
# integrating over the hyperparameter grid -- the right object for calibrated
# coefficient standard errors.
.nmix_grid_vcov <- function(cov_blocks, modes, weights, p_lam, p_p, nm) {
  p_beta <- p_lam + p_p
  beta_modes <- modes[, seq_len(p_beta), drop = FALSE]
  V <- matrix(NA_real_, p_beta, p_beta)
  ok <- is.finite(weights) & weights > 0
  if (any(ok)) {
    w <- weights; w[!ok] <- 0
    mbar <- as.numeric(crossprod(w, beta_modes))
    V <- matrix(0, p_beta, p_beta)
    for (k in which(ok)) {
      Ck <- cov_blocks[[k]]
      if (is.null(Ck) || anyNA(Ck)) next
      dk <- beta_modes[k, ] - mbar
      V <- V + w[k] * (as.matrix(Ck) + tcrossprod(dk))
    }
  }
  dimnames(V) <- list(nm, nm)
  V
}
