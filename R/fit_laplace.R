#' Fit a model via Laplace approximation
#'
#' @description
#' General-purpose Laplace approximation for latent Gaussian models.
#' Finds the mode of the latent field (beta + random effects) and
#' returns the Laplace-approximated marginal likelihood.
#'
#' This is the public API for model packages (tulpaGlmm, tulpaOcc, etc.)
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

  # Compute Hessian for fixed-effect block if requested
  if (return_hessian && !is.null(result$mode)) {
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
  } else {
    stop(sprintf("Spatial type '%s' not yet supported in Laplace", spatial_type),
         call. = FALSE)
  }
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
