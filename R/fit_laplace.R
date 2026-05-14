#' Fit a model via Laplace approximation
#'
#' @description
#' General-purpose Laplace approximation for latent Gaussian models.
#' Finds the mode of the latent field (beta + random effects) and
#' returns the Laplace-approximated marginal likelihood.
#'
#' This is the public API for model packages (tulpaGlmm, tulpaObs, etc.)
#' to call tulpa's Laplace engine.
#'
#' @param y Response vector (integer for binomial/poisson/negbin, numeric for gaussian)
#' @param n_trials Trial sizes (integer vector, used for binomial only)
#' @param X Fixed-effects design matrix
#' @param re_list List of RE specifications. Each element is a list with:
#'   - `idx`: integer vector of group indices (1-based)
#'   - `n_groups`: number of groups
#'   - `sigma`: RE standard deviation
#'   - `Z`: optional slope design matrix (n_obs x n_coefs). If NULL, intercept-only.
#' @param family Character: `"binomial"`, `"poisson"`, `"neg_binomial_2"`, `"gaussian"`
#' @param phi Dispersion parameter (neg_binomial_2 and gamma only)
#' @param spatial Optional spatial specification (tulpa_spatial object)
#' @param max_iter Maximum Newton iterations (default 100)
#' @param tol Convergence tolerance (default 1e-6)
#' @param n_threads Number of threads (default 1)
#' @param return_hessian Logical: return the fixed-effect Hessian block? (default TRUE)
#'
#' @return A list with:
#'   - `mode`: full mode vector (beta, then RE values per term)
#'   - `log_marginal`: Laplace-approximated log-marginal likelihood
#'   - `n_iter`: number of Newton iterations
#'   - `converged`: logical
#'   - `log_det_Q`: log-determinant of the Hessian
#'   - `H_beta`: fixed-effect block of the Hessian (if return_hessian = TRUE)
#'
#' @export
tulpa_laplace <- function(y, n_trials, X,
                          re_list = list(),
                          family = "binomial",
                          phi = 1.0,
                          spatial = NULL,
                          weights = NULL,
                          offset = NULL,
                          max_iter = 100L, tol = 1e-6,
                          n_threads = 1L,
                          return_hessian = TRUE) {

  n_obs <- length(y)
  n_fixed <- ncol(X)

  # Validate
  stopifnot(is.numeric(y) || is.integer(y))
  stopifnot(is.matrix(X))
  stopifnot(nrow(X) == n_obs)

  if (identical(family, "beta")) {
    if (!is.numeric(phi) || length(phi) != 1L || !is.finite(phi) || phi <= 0) {
      stop("`phi` must be a positive scalar for family = 'beta'.", call. = FALSE)
    }
    yfin <- y[is.finite(y)]
    if (length(yfin) && (min(yfin) <= 0 || max(yfin) >= 1)) {
      stop("family = 'beta' requires y strictly in (0, 1); got range [",
           min(yfin), ", ", max(yfin),
           "]. Use cover(positive = 'beta') for hurdle handling of 0/1.",
           call. = FALSE)
    }
  }

  if (is.null(n_trials)) n_trials <- rep(1L, n_obs)

  # Route based on number of RE terms and spatial type
  if (!is.null(spatial)) {
    # Spatial path: use first RE term (single-block)
    if (length(re_list) == 0) {
      re_idx <- rep(1L, n_obs); n_re_groups <- 1L; sigma_re <- 1.0
    } else {
      re_idx <- re_list[[1]]$idx
      n_re_groups <- re_list[[1]]$n_groups
      sigma_re <- re_list[[1]]$sigma
    }
    result <- dispatch_laplace_spatial(
      y, n_trials, X, re_idx, n_re_groups, sigma_re,
      spatial, family, phi, max_iter, tol, n_threads
    )
  } else {
    # All non-spatial paths: use cpp_laplace_fit_multi_re
    # (handles single RE, multiple RE, slopes, weights, offset)
    if (length(re_list) == 0) {
      re_idx_list <- list(as.integer(rep(1L, n_obs)))
      re_ngroups <- 1L
      re_sigma_list <- list(1.0)
    } else {
      re_idx_list <- lapply(re_list, function(r) as.integer(r$idx))
      re_ngroups <- vapply(re_list, function(r) as.integer(r$n_groups), integer(1))
    }

    re_ncoefs <- vapply(
      if (length(re_list) > 0) re_list else list(list(n_coefs = 1L)),
      function(r) as.integer(r$n_coefs %||% 1L), integer(1)
    )

    if (length(re_list) > 0) {
      re_sigma_list <- lapply(re_list, function(r) {
        s <- r$sigma
        nc <- r$n_coefs %||% 1L
        if (length(s) == 1L && nc > 1L) rep(s, nc) else s
      })
    }

    re_Z_list <- lapply(
      if (length(re_list) > 0) re_list else list(list(n_coefs = 1L)),
      function(r) {
        nc <- r$n_coefs %||% 1L
        if (nc == 1L) return(NULL)
        r$Z
      }
    )

    has_slopes <- any(re_ncoefs > 1L)

    result <- cpp_laplace_fit_multi_re(
      y = as.numeric(y),
      n = as.integer(n_trials),
      X = X,
      re_idx_list = re_idx_list,
      re_ngroups = re_ngroups,
      re_sigma_list = re_sigma_list,
      family = family,
      phi = phi,
      max_iter = as.integer(max_iter),
      tol = tol,
      n_threads = as.integer(n_threads),
      re_Z_list = if (has_slopes) re_Z_list else NULL,
      re_ncoefs = if (has_slopes) re_ncoefs else NULL,
      weights = weights,
      offset = offset
    )
  }

  # SPDE / NNGP Laplace return mode = c(beta, spatial_effects) where the
  # spatial contribution to eta is A * spatial_effects (mesh-projected) or
  # w_at_obs (NNGP), not Z * u. The fixed-effect Hessian below assumes
  # eta = X*beta + Z*u and would under-weight observations by ignoring the
  # spatial term, so skip it for these spatial types. Proper Hessian /
  # uncertainty propagation for spatial fields is a separate item.
  is_spatial_field <- !is.null(spatial) &&
    (identical(spatial$type, "spde") || identical(spatial$type, "gp"))

  # Compute Hessian for fixed-effect block if requested
  if (return_hessian && !is.null(result$mode) && !is_spatial_field) {
    mode_vec <- result$mode
    beta <- mode_vec[seq_len(n_fixed)]
    re_vals <- mode_vec[-seq_len(n_fixed)]

    # Linear predictor at mode: eta = X*beta + sum_k Z_k * u_k
    eta <- as.numeric(X %*% beta)

    if (length(re_list) > 0) {
      # For multi-block: each term adds its RE contribution
      offset <- 0L
      for (k in seq_along(re_list)) {
        r <- re_list[[k]]
        u_k <- re_vals[offset + seq_len(r$n_groups)]
        eta <- eta + u_k[r$idx]
        offset <- offset + r$n_groups
      }
    }

    # GLM weights
    W <- glmm_weights(eta, family, n_trials, phi)

    # Apply observation weights
    if (!is.null(weights)) W <- W * weights

    # Fixed-effect precision block
    XtWX <- crossprod(X, W * X)

    if (length(re_list) > 0) {
      # Build combined Z and block-diagonal D^{-1} for Schur complement
      # Each term k contributes n_groups[k] * n_coefs[k] latent variables
      total_re <- sum(vapply(re_list, function(r) {
        as.integer(r$n_groups) * (r$n_coefs %||% 1L)
      }, integer(1)))
      Z_parts <- list()
      D_inv_diag <- numeric(total_re)
      offset <- 0L
      for (k in seq_along(re_list)) {
        r <- re_list[[k]]
        nc <- r$n_coefs %||% 1L
        sig <- r$sigma
        if (length(sig) < nc) sig <- rep(sig[1], nc)

        if (nc == 1L) {
          # Intercept-only: Z is n_obs x n_groups indicator matrix
          Z_parts[[k]] <- Matrix::sparseMatrix(
            i = seq_len(n_obs), j = r$idx,
            x = rep(1.0, n_obs), dims = c(n_obs, r$n_groups)
          )
          D_inv_diag[offset + seq_len(r$n_groups)] <- 1 / (sig[1]^2 + 1e-10)
          offset <- offset + r$n_groups
        } else {
          # Slopes: Z is n_obs x (n_groups * n_coefs)
          # Column layout: [g1_c1, g1_c2, ..., g2_c1, g2_c2, ...]
          # Build Z from the full Z matrix (intercept + slopes)
          Z_full <- r$Z  # n_obs x nc
          if (is.null(Z_full)) {
            Z_full <- matrix(1, nrow = n_obs, ncol = 1)
          }
          n_latent <- r$n_groups * nc
          # Build sparse Z: for obs i in group g, row i has Z_full[i,:] at cols (g-1)*nc+1 .. g*nc
          ii <- rep(seq_len(n_obs), each = nc)
          jj <- rep((r$idx - 1L) * nc, each = nc) + rep(seq_len(nc), n_obs)
          xx <- as.numeric(t(Z_full))
          Z_parts[[k]] <- Matrix::sparseMatrix(
            i = ii, j = jj, x = xx, dims = c(n_obs, n_latent)
          )
          # D_inv: per-coef sigma applied to each group's copy of that coef
          for (c_idx in seq_len(nc)) {
            cols <- seq(c_idx, n_latent, by = nc)
            D_inv_diag[offset + cols] <- 1 / (sig[c_idx]^2 + 1e-10)
          }
          offset <- offset + n_latent
        }
      }
      Z <- do.call(cbind, Z_parts)
      ZtWZ <- Matrix::crossprod(Z, W * Z)
      D_inv <- Matrix::Diagonal(total_re, D_inv_diag)
      ZtWZ_Dinv <- ZtWZ + D_inv
      XtWZ <- crossprod(X, W * Z)
      R <- chol(as.matrix(ZtWZ_Dinv))
      mid <- backsolve(R, forwardsolve(t(R), as.matrix(Matrix::t(XtWZ))))
      P_beta <- XtWX - XtWZ %*% mid
    } else {
      P_beta <- XtWX
    }

    result$H_beta <- as.matrix(P_beta)
  }

  result
}


#' Dispatch spatial Laplace to the correct C++ backend
#' @keywords internal
dispatch_laplace_spatial <- function(y, n_trials, X, re_idx, n_re_groups,
                                     sigma_re, spatial, family, phi,
                                     max_iter, tol, n_threads) {

  spatial_type <- spatial$type

  if (spatial_type %in% c("icar", "car")) {
    adj <- spatial$adjacency
    adj_sparse <- adjacency_to_csr_tulpa(adj)

    cpp_laplace_fit_spatial(
      y = as.numeric(y), n = as.integer(n_trials), X = X,
      re_idx = as.numeric(re_idx), n_re_groups = as.integer(n_re_groups),
      sigma_re = sigma_re,
      spatial_idx = as.integer(spatial$spatial_idx %||% seq_len(nrow(adj))),
      n_spatial_units = as.integer(nrow(adj)),
      adj_row_ptr = as.integer(adj_sparse$row_ptr),
      adj_col_idx = as.integer(adj_sparse$col_idx),
      n_neighbors = as.integer(adj_sparse$n_neighbors),
      tau_spatial = 1.0,
      family = family, phi = phi,
      max_iter = as.integer(max_iter), tol = tol,
      n_threads = as.integer(n_threads)
    )
  } else if (spatial_type == "bym2") {
    adj <- spatial$adjacency
    adj_sparse <- adjacency_to_csr_tulpa(adj)

    cpp_laplace_fit_bym2(
      y = as.numeric(y), n = as.integer(n_trials), X = X,
      re_idx = as.numeric(re_idx), n_re_groups = as.integer(n_re_groups),
      sigma_re = sigma_re,
      spatial_idx = as.integer(spatial$spatial_idx %||% seq_len(nrow(adj))),
      n_spatial_units = as.integer(nrow(adj)),
      adj_row_ptr = as.integer(adj_sparse$row_ptr),
      adj_col_idx = as.integer(adj_sparse$col_idx),
      n_neighbors = as.integer(adj_sparse$n_neighbors),
      sigma_spatial = 1.0, rho = 0.5,
      scale_factor = spatial$scale_factor %||% 1.0,
      family = family, phi = phi,
      max_iter = as.integer(max_iter), tol = tol,
      n_threads = as.integer(n_threads)
    )
  } else if (spatial_type == "spde") {
    # SPDE Laplace at fixed hyperparameters (uses spec's prior modes).
    # Nested integration over (range, sigma) is opt-in via fit_spde().
    if (!is.null(re_idx) && length(re_idx) > 0 && n_re_groups > 1L) {
      stop("SPDE Laplace does not yet support an additional iid RE block. ",
           "Drop the RE list or use HMC.", call. = FALSE)
    }
    laplace_spde_at(
      y = y, n_trials = n_trials, X = X, spatial = spatial,
      family = family, phi = phi,
      range = NULL, sigma = NULL,
      max_iter = max_iter, tol = tol, n_threads = n_threads
    )
  } else if (spatial_type == "gp") {
    # NNGP Laplace at fixed hyperparameters. Like SPDE: an additional iid RE
    # block is not currently supported in this branch — cpp_laplace_fit_gp's
    # n_re_groups > 0 path is exercised by HMC, but here we route only the
    # spatial-only case.
    if (!is.null(re_idx) && length(re_idx) > 0 && n_re_groups > 1L) {
      stop("NNGP Laplace does not yet support an additional iid RE block. ",
           "Drop the RE list or use HMC.", call. = FALSE)
    }
    laplace_gp_at(
      y = y, n_trials = n_trials, X = X, spatial = spatial,
      family = family, phi = phi,
      sigma2_gp = NULL, phi_gp = NULL,
      max_iter = max_iter, tol = tol, n_threads = n_threads
    )
  } else {
    stop(sprintf("Spatial type '%s' not yet supported in Laplace", spatial_type),
         call. = FALSE)
  }
}


#' SPDE Laplace at given hyperparameters
#'
#' Single-point Laplace approximation for an SPDE spatial field at a fixed
#' (range, sigma). Used by both `dispatch_laplace_spatial` (single-point
#' path) and `fit_spde` (single-point branch) so the call site stays a
#' single source of truth.
#'
#' @param y Response vector.
#' @param n_trials Trial sizes (binomial).
#' @param X Fixed-effects design matrix.
#' @param spatial A `tulpa_spatial` object of type `"spde"`.
#' @param family Distribution family.
#' @param phi Dispersion parameter (negbin / gamma only).
#' @param range Spatial range (NULL → use `spatial$prior_range[1]`).
#' @param sigma Marginal SD (NULL → use `spatial$prior_sigma[1]`).
#' @param max_iter Newton iterations.
#' @param tol Newton tolerance.
#' @param n_threads OpenMP threads.
#' @return The raw `cpp_laplace_fit_spde` result list (mode, log_det_Q,
#'   log_marginal, n_iter, converged), augmented with `range`, `sigma`,
#'   and the spatial spec for downstream prediction.
#' @keywords internal
laplace_spde_at <- function(y, n_trials, X, spatial,
                             family = "binomial", phi = 1.0,
                             range = NULL, sigma = NULL,
                             max_iter = 100L, tol = 1e-6, n_threads = 1L) {
  if (is.null(range)) range <- spatial$prior_range[1]
  if (is.null(sigma)) sigma <- spatial$prior_sigma[1]

  kappa <- sqrt(8 * spatial$nu) / range
  tau_spde <- 1.0 / (sqrt(4 * pi) * kappa * sigma)
  alpha <- as.integer(round(spatial$nu)) + 1L
  rat <- rational_spde_coefficients(spatial$nu)

  result <- cpp_laplace_fit_spde(
    y = as.numeric(y),
    n_trials = as.integer(n_trials %||% rep(1L, length(y))),
    X = as.matrix(X),
    A_x = spatial$A_x, A_i = spatial$A_i, A_p = spatial$A_p,
    n_obs = length(y), n_mesh = spatial$n_mesh,
    C0_diag = spatial$C0_diag,
    G1_x = spatial$G1_x, G1_i = spatial$G1_i, G1_p = spatial$G1_p,
    kappa = kappa, tau_spde = tau_spde,
    family = family, phi = phi, alpha = alpha,
    max_iter = as.integer(max_iter), tol = tol,
    n_threads = as.integer(n_threads),
    rational_poles = if (!rat$is_integer) rat$poles else NULL,
    rational_weights = if (!rat$is_integer) rat$weights else NULL
  )

  result$range <- range
  result$sigma <- sigma
  result$spatial <- spatial
  result
}


#' Map a spatial_gp covariance spec to the Laplace cov_type integer
#'
#' The Laplace NNGP kernel (`laplace_core.cpp`) supports three covariance
#' functions: 0 = exponential, 1 = Matern(nu=1.5), 2 = Matern(nu=2.5).
#' Anything else is rejected with a clear error rather than silently
#' falling back to a different covariance.
#'
#' @keywords internal
gp_cov_type_for_laplace <- function(spatial) {
  cov <- spatial$cov %||% "exponential"
  if (cov == "exponential") return(0L)
  if (cov == "matern") {
    nu <- spatial$nu %||% 1.5
    if (isTRUE(all.equal(nu, 1.5))) return(1L)
    if (isTRUE(all.equal(nu, 2.5))) return(2L)
    stop("NNGP Laplace supports Matern with nu in {1.5, 2.5}; ",
         "got nu = ", format(nu),
         ". Use HMC (which supports general nu) or set nu = 1.5 / 2.5.",
         call. = FALSE)
  }
  stop(sprintf(
    "NNGP Laplace supports cov in {'exponential','matern'}; got '%s'. ",
    cov),
    "Use HMC for gaussian / spherical covariances.",
    call. = FALSE)
}

#' NNGP Laplace at given hyperparameters
#'
#' Single-point Laplace approximation for a Matern/exponential GP spatial
#' field at fixed (sigma2_gp, phi_gp). Used by `dispatch_laplace_spatial`
#' when `spatial$type == "gp"`. The neighbor structure is read straight
#' off the validated spec — call `validate_gp(spatial, data)` first if
#' constructing manually.
#'
#' @param y Response vector.
#' @param n_trials Trial sizes (binomial).
#' @param X Fixed-effects design matrix.
#' @param spatial A `tulpa_gp` spec, validated (i.e., `neighbor_info` populated).
#' @param family Distribution family.
#' @param phi Dispersion parameter (negbin / gamma only).
#' @param sigma2_gp Marginal variance (NULL → 1.0).
#' @param phi_gp Range / decay parameter (NULL → 1.0).
#' @param max_iter Newton iterations.
#' @param tol Newton tolerance.
#' @param n_threads OpenMP threads.
#' @return The raw `cpp_laplace_fit_gp` result list, augmented with
#'   `sigma2_gp`, `phi_gp`, and the spatial spec.
#' @keywords internal
laplace_gp_at <- function(y, n_trials, X, spatial,
                           family = "binomial", phi = 1.0,
                           sigma2_gp = NULL, phi_gp = NULL,
                           max_iter = 100L, tol = 1e-6, n_threads = 1L) {
  if (is.null(spatial$neighbor_info)) {
    stop("spatial_gp() spec is unvalidated (neighbor_info is NULL). ",
         "Call validate_gp(spatial, data) first, or fit through tulpa() ",
         "which validates automatically.", call. = FALSE)
  }
  if (is.null(sigma2_gp)) sigma2_gp <- spatial$sigma2_gp %||% 1.0
  if (is.null(phi_gp))    phi_gp    <- spatial$phi_gp    %||% 1.0

  cov_type <- gp_cov_type_for_laplace(spatial)
  ni <- spatial$neighbor_info
  n_spatial <- spatial$n_spatial %||% nrow(spatial$unique_coords)
  nn <- spatial$nn %||% ncol(ni$nn_idx)

  # nn_order in cpp_laplace_fit_gp expects 0-based indexing, matching the
  # convention in test-gpu-nngp.R. spatial_gp() stores order_idx as 1-based.
  nn_order_0 <- as.integer((ni$nn_order %||% seq_len(n_spatial)) - 1L)

  # The spatial-only case has no separate iid RE block; pass dummy re_idx.
  re_idx <- rep(0.0, length(y))

  result <- cpp_laplace_fit_gp(
    y = as.numeric(y),
    n = as.integer(n_trials %||% rep(1L, length(y))),
    X = as.matrix(X),
    re_idx = re_idx,
    n_re_groups = 0L,
    sigma_re = 1.0,
    coords = as.matrix(spatial$unique_coords),
    nn_idx = as.matrix(ni$nn_idx),
    nn_dist = as.matrix(ni$nn_dist),
    nn_order = nn_order_0,
    n_spatial = as.integer(n_spatial),
    nn = as.integer(nn),
    sigma2_gp = sigma2_gp,
    phi_gp = phi_gp,
    cov_type = cov_type,
    family = family,
    phi = phi,
    max_iter = as.integer(max_iter),
    tol = tol,
    n_threads = as.integer(n_threads)
  )

  result$sigma2_gp <- sigma2_gp
  result$phi_gp <- phi_gp
  result$spatial <- spatial
  result
}


#' Convert adjacency matrix to CSR for tulpa C++
#' @keywords internal
adjacency_to_csr_tulpa <- function(adj) {
  if (inherits(adj, "sparseMatrix")) adj <- as.matrix(adj)
  n <- nrow(adj)
  row_ptr <- integer(n + 1)
  col_idx <- integer(0)
  n_neighbors <- integer(n)
  for (i in seq_len(n)) {
    neighbors <- which(adj[i, ] != 0)
    n_neighbors[i] <- length(neighbors)
    col_idx <- c(col_idx, neighbors - 1L)
    row_ptr[i + 1] <- row_ptr[i] + length(neighbors)
  }
  list(row_ptr = row_ptr, col_idx = col_idx, n_neighbors = n_neighbors)
}


#' Compute GLM working weights for Laplace Hessian
#' @keywords internal
glmm_weights <- function(eta, family, n_trials = NULL, phi = 1.0) {

  if (family == "binomial") {
    mu <- 1 / (1 + exp(-eta))
    mu <- pmin(pmax(mu, 1e-8), 1 - 1e-8)
    n <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
    w <- n * mu * (1 - mu)
  } else if (family == "poisson") {
    w <- pmax(exp(eta), 1e-8)
  } else if (family == "neg_binomial_2") {
    mu <- pmax(exp(eta), 1e-8)
    w <- mu * phi / (mu + phi)
  } else if (family == "gaussian") {
    w <- rep(1, length(eta))
  } else if (family == "beta") {
    mu <- 1 / (1 + exp(-eta))
    mu <- pmin(pmax(mu, 1e-7), 1 - 1e-7)
    dmu <- mu * (1 - mu)
    tg <- trigamma(mu * phi) + trigamma((1 - mu) * phi)
    w <- phi * phi * tg * dmu * dmu
  } else {
    w <- rep(1, length(eta))
  }

  as.numeric(w)
}


#' Fit via Polya-Gamma Gibbs sampler
#'
#' @description
#' Public API for PG Gibbs sampling. Used by model packages for
#' binomial and negative binomial GLMMs.
#'
#' @param y Response vector
#' @param n_trials Trial sizes (binomial)
#' @param X Design matrix
#' @param group Integer vector of group indices (1-based)
#' @param n_groups Number of groups
#' @param family Character: "binomial" or "neg_binomial_2"
#' @param iter Total iterations
#' @param warmup Warmup iterations
#' @param prior_beta_sd Prior SD for betas
#' @param prior_sigma_scale Prior scale for RE sigma
#' @param verbose Print progress
#' @param n_threads Number of threads
#'
#' @return List with beta draws, RE draws, sigma_re draws
#'
#' @export
tulpa_gibbs <- function(y, n_trials, X, group, n_groups,
                        family = "binomial",
                        iter = 2000L, warmup = 1000L,
                        prior_beta_sd = 10.0, prior_sigma_scale = 2.5,
                        verbose = TRUE, n_threads = 1L) {

  if (family == "binomial") {
    cpp_pg_binomial_gibbs(
      y = as.numeric(y), n = as.integer(n_trials), X = X,
      group = as.integer(group), n_groups = as.integer(n_groups),
      n_iter = as.integer(iter), n_warmup = as.integer(warmup),
      thin = 1L,
      prior_beta_sd = prior_beta_sd,
      prior_sigma_scale = prior_sigma_scale,
      store_eta = FALSE, verbose = verbose,
      n_threads = as.integer(n_threads)
    )
  } else if (family %in% c("neg_binomial_2", "negbin")) {
    cpp_pg_negbin_gibbs(
      y = as.numeric(y), X = X,
      group = as.integer(group), n_groups = as.integer(n_groups),
      n_iter = as.integer(iter), n_warmup = as.integer(warmup),
      thin = 1L,
      prior_beta_sd = prior_beta_sd,
      prior_sigma_scale = prior_sigma_scale,
      prior_r_shape = 1.0, prior_r_rate = 0.1, r_init = 5.0,
      store_eta = FALSE, verbose = verbose,
      n_threads = as.integer(n_threads)
    )
  } else {
    stop(sprintf("Gibbs not available for family '%s'", family), call. = FALSE)
  }
}
