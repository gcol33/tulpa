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
#'   - `n_coefs`: coefficients per group (1 = intercept-only, >1 = random slopes)
#'   - `sigma`: per-coefficient RE standard deviation(s), a diagonal covariance
#'     (uncorrelated, lme4 `(x || g)`). A scalar is recycled to `n_coefs`.
#'   - `Z`: slope design matrix (n_obs x n_coefs) when `n_coefs > 1`; `NULL`
#'     means intercept-only.
#'   - `L` / `cov`: optional `n_coefs x n_coefs` covariance for a *correlated*
#'     term (lme4 `(1 + x | g)`) -- supply either a lower-triangular Cholesky
#'     factor `L` (covariance = `L L'`) or the covariance matrix `cov`. When
#'     present these take precedence over `sigma`; the off-diagonal enters both
#'     the joint Hessian (mode finding) and the marginal fixed-effect SE.
#' @param family Character: `"binomial"`, `"poisson"`, `"neg_binomial_2"`, `"gaussian"`
#' @param phi Dispersion parameter (neg_binomial_2 and gamma only)
#' @param spatial Optional spatial specification (tulpa_spatial object)
#' @param weights Optional observation weights (numeric vector, length `length(y)`).
#'   Scales each observation's likelihood contribution. `NULL` (default) uses 1.
#' @param offset Optional observation-level offset on the linear predictor
#'   (numeric vector, length `length(y)`). `NULL` (default) uses 0.
#' @param max_iter Maximum Newton iterations (default 100)
#' @param tol Convergence tolerance (default 1e-6)
#' @param n_threads Number of threads (default 1)
#' @param return_hessian Logical: return the fixed-effect Hessian block? (default TRUE)
#' @param beta_prior Optional Gaussian prior on the fixed effects. `NULL`
#'   (default) keeps the weak built-in prior `beta ~ N(0, 100^2)`. Otherwise a
#'   list with element `sd` (prior standard deviation, required) and optional
#'   `mean` (prior mean, default 0). Each may be a scalar (applied to every
#'   coefficient) or a length-`ncol(X)` vector. Adds
#'   `sum((beta - mean)^2 / (2 * sd^2))` to the negative log-posterior, so the
#'   mode is the penalized (MAP) estimate. A coefficient's `sd` may be `+Inf`,
#'   which sets its precision to 0 (no penalty on that coefficient). Not
#'   supported on the spatial path.
#'
#' @param return_re_cov If `TRUE`, additionally return per-group marginal
#'   posterior covariance blocks `Cov(u_g | y, Sigma)` -- one
#'   `n_coefs x n_coefs` matrix per (RE term, group), with the fixed effects
#'   and other groups marginalized out (each block is a diagonal block of the
#'   full inverse Hessian, not the inverse of a diagonal block). Used by the EM
#'   M-step for a full random-effect covariance. Non-spatial multi-RE path only.
#'
#' @return A list with:
#'   - `mode`: full mode vector (beta, then RE values per term)
#'   - `log_marginal`: Laplace-approximated log-marginal likelihood
#'   - `n_iter`: number of Newton iterations
#'   - `converged`: logical
#'   - `log_det_Q`: log-determinant of the Hessian
#'   - `H_beta`: fixed-effect block of the Hessian (if return_hessian = TRUE)
#'   - `cov_blocks`: list of per-group posterior covariance matrices, one per
#'     (RE term, group) in term-major then group order (if return_re_cov = TRUE)
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
                          return_hessian = TRUE,
                          beta_prior = NULL,
                          return_re_cov = FALSE) {

  n_obs <- length(y)
  n_fixed <- ncol(X)

  # Validate
  stopifnot(is.numeric(y) || is.integer(y))
  stopifnot(is.matrix(X))
  stopifnot(nrow(X) == n_obs)

  # Normalize the optional fixed-effect prior to length-p mean / sd vectors.
  bp <- .normalize_beta_prior(beta_prior, n_fixed)

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

  if (isTRUE(return_re_cov) && !is.null(spatial)) {
    stop("`return_re_cov` is only available on the non-spatial multi-RE path. ",
         "The spatial solvers do not expose per-group posterior covariance ",
         "blocks; drop `spatial` or `return_re_cov`.", call. = FALSE)
  }

  # Route based on number of RE terms and spatial type
  if (!is.null(spatial)) {
    if (!is.null(bp)) {
      stop("`beta_prior` is not supported on the spatial Laplace path. ",
           "The spatial solvers use the built-in weak fixed-effect prior; ",
           "drop `beta_prior`, or use NUTS for a custom fixed-effect prior ",
           "under a spatial field.", call. = FALSE)
    }
    # Spatial path: use first RE term (single-block)
    if (length(re_list) == 0) {
      # No formula-side RE: genuine no-RE fit (n_re_groups = 0). The areal
      # kernels (laplace_mode_spatial / _bym2 / _rsr) all guard their RE block
      # with `if (n_re_groups > 0)` and size the latent vector as
      # p + n_re_groups + n_spatial_units, so 0 means zero RE dims. Injecting a
      # 1-group sigma = 1 random intercept here (as we used to) adds a spurious
      # latent that confounds with the fixed intercept -- inert at the MAP only
      # when the intercept is unpenalised, but it perturbs the joint Hessian /
      # marginal likelihood and would bias the intercept under a beta_prior.
      # Matches the SPDE (fit_spde.R) and GP (laplace_gp_at) no-RE convention.
      re_idx <- rep(0L, n_obs); n_re_groups <- 0L; sigma_re <- 1.0
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
      # No random effects: fit a pure fixed-effects model (n_terms = 0 in the
      # kernel, so the latent vector is beta only). A dummy 1-group RE used to
      # be injected here, but a global random intercept (sigma = 1) confounds
      # with the fixed intercept once a `beta_prior` is active: the prior pulls
      # the penalised beta toward its mean while the unpenalised RE absorbs the
      # difference, biasing the fixed-effect MAP (the penalty's effective
      # precision on the intercept collapses to 1). An empty RE set removes the
      # spurious latent entirely so the MAP is the true penalised optimum.
      re_idx_list <- list()
      re_ngroups <- integer(0)
      re_sigma_list <- list()
    } else {
      re_idx_list <- lapply(re_list, function(r) as.integer(r$idx))
      re_ngroups <- vapply(re_list, function(r) as.integer(r$n_groups), integer(1))
    }

    re_ncoefs <- vapply(
      if (length(re_list) > 0) re_list else list(list(n_coefs = 1L)),
      function(r) as.integer(r$n_coefs %||% 1L), integer(1)
    )

    if (length(re_list) > 0) {
      # `pack` is the value the C++ kernel consumes: a length-n_coefs marginal-SD
      # vector (diagonal) or a packed lower-triangular Cholesky (correlated).
      re_sigma_list <- lapply(re_list, function(r) .re_cov_spec(r)$pack)
    }

    # Per-term RE design: an intercept-only term carries no Z (the kernel
    # defaults to a column of 1s); a slope term -- including a single random
    # slope `(0 + x | g)` with n_coefs == 1 -- carries its own design Z.
    re_Z_list <- lapply(
      if (length(re_list) > 0) re_list else list(list(n_coefs = 1L)),
      function(r) r$Z
    )

    # The kernel needs the Z / n_coefs metadata whenever any term has a
    # non-intercept design: a correlated/uncorrelated slope (n_coefs > 1) or a
    # single random slope that supplied its own Z.
    has_design <- any(re_ncoefs > 1L) ||
      any(vapply(re_Z_list, Negate(is.null), logical(1)))

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
      re_Z_list = if (has_design) re_Z_list else NULL,
      re_ncoefs = if (has_design) re_ncoefs else NULL,
      weights = weights,
      offset = offset,
      beta_prior_mean = if (is.null(bp)) NULL else bp$mean,
      beta_prior_sd   = if (is.null(bp)) NULL else bp$sd,
      return_re_cov   = isTRUE(return_re_cov)
    )
  }

  # SPDE / NNGP Laplace return mode = c(beta, spatial_effects). The
  # fixed-effect block of the joint Hessian is the *conditional* precision
  # P(beta | u^*) and under-states uncertainty. For SPDE the marginal block
  # is obtained by Schur complement on the joint Hessian (issue #16):
  #   H_beta^marg = X'WX - X'WA (A'WA + Q_spde)^{-1} A'WX
  # The dense Z-Schur branch below handles every non-spatial-field path;
  # the SPDE-specific marginal lives in .marginal_H_beta_spde() and is
  # invoked at the bottom of this function. NNGP marginal SE is still
  # outstanding (tracked under #16).
  is_spatial_field <- !is.null(spatial) &&
    (identical(spatial$type, "spde") || identical(spatial$type, "gp"))

  # Compute Hessian for fixed-effect block if requested
  if (return_hessian && !is.null(result$mode) && !is_spatial_field) {
    mode_vec <- result$mode
    beta <- mode_vec[seq_len(n_fixed)]
    re_vals <- mode_vec[-seq_len(n_fixed)]

    # Linear predictor at mode: eta = X*beta + sum_k Z_k %*% u_k. The mode
    # stores term k as [g1_c1, g1_c2, ..., g2_c1, ...] (n_groups * n_coefs),
    # so a slope term contributes Z_k[i, ] %*% u_{g(i)}, not just an intercept.
    eta <- as.numeric(X %*% beta)

    if (length(re_list) > 0) {
      offset <- 0L
      for (k in seq_along(re_list)) {
        r  <- re_list[[k]]
        nc <- r$n_coefs %||% 1L
        # Z is the term's design: a supplied Z (slopes, incl. a single
        # `(0 + x | g)`) or the intercept indicator (column of 1s) when absent.
        Zk <- r$Z %||% matrix(1, n_obs, 1L)
        u_k <- re_vals[offset + seq_len(r$n_groups * nc)]
        u_mat <- matrix(u_k, ncol = nc, byrow = TRUE)   # row g = (c1, ..., cnc)
        eta <- eta + rowSums(Zk * u_mat[r$idx, , drop = FALSE])
        offset <- offset + r$n_groups * nc
      }
    }

    # GLM weights
    W <- glmm_weights(eta, family, n_trials, phi)

    # Apply observation weights
    if (!is.null(weights)) W <- W * weights

    # Fixed-effect precision block
    XtWX <- crossprod(X, W * X)

    if (length(re_list) > 0) {
      # Build combined Z and the RE precision D^{-1} for the Schur complement.
      # D^{-1} is block-diagonal in the latent layout [g1_c1, g1_c2, g2_c1, ...]:
      # one n_coefs x n_coefs precision block per group -- full (off-diagonal
      # included) for a correlated term, diagonal otherwise. Each term k
      # contributes n_groups[k] * n_coefs[k] latent variables.
      Z_parts <- list()
      Dinv_blocks <- list()
      for (k in seq_along(re_list)) {
        r <- re_list[[k]]
        nc <- r$n_coefs %||% 1L
        Qk <- .re_cov_spec(r)$Q  # nc x nc RE precision (Sigma^{-1})

        if (nc == 1L && is.null(r$Z)) {
          # Intercept-only: Z is n_obs x n_groups indicator matrix
          Z_parts[[k]] <- Matrix::sparseMatrix(
            i = seq_len(n_obs), j = r$idx,
            x = rep(1.0, n_obs), dims = c(n_obs, r$n_groups)
          )
        } else {
          # Slopes (incl. a single random slope `(0 + x | g)`, n_coefs == 1 with
          # a supplied Z): Z is n_obs x (n_groups * n_coefs), column layout
          # [g1_c1, g1_c2, ..., g2_c1, g2_c2, ...].
          Z_full <- r$Z %||% matrix(1, nrow = n_obs, ncol = 1)
          n_latent <- r$n_groups * nc
          ii <- rep(seq_len(n_obs), each = nc)
          jj <- rep((r$idx - 1L) * nc, each = nc) + rep(seq_len(nc), n_obs)
          xx <- as.numeric(t(Z_full))
          Z_parts[[k]] <- Matrix::sparseMatrix(
            i = ii, j = jj, x = xx, dims = c(n_obs, n_latent)
          )
        }
        # One precision block per group, same column order as Z. The block is
        # the full Q_k for a correlated term, so the off-diagonal propagates
        # into the marginal fixed-effect SE rather than being dropped.
        Dinv_blocks[[k]] <- Matrix::bdiag(
          rep(list(Matrix::Matrix(Qk, sparse = TRUE)), r$n_groups)
        )
      }
      Z <- do.call(cbind, Z_parts)
      ZtWZ <- Matrix::crossprod(Z, W * Z)
      D_inv <- Matrix::bdiag(Dinv_blocks)
      ZtWZ_Dinv <- ZtWZ + D_inv
      XtWZ <- crossprod(X, W * Z)
      R <- chol(as.matrix(ZtWZ_Dinv))
      mid <- backsolve(R, forwardsolve(t(R), as.matrix(Matrix::t(XtWZ))))
      P_beta <- XtWX - XtWZ %*% mid
    } else {
      P_beta <- XtWX
    }

    # Add the fixed-effect prior precision so H_beta is the negative-log-
    # POSTERIOR curvature (matching the penalty the mode-finding kernel added),
    # not just the likelihood information. Without this the Laplace SE would
    # ignore the prior. `sd = Inf` contributes 0 (no penalty on that coef).
    if (!is.null(bp)) {
      pen_prec <- ifelse(is.finite(bp$sd), 1 / (bp$sd^2), 0)
      diag(P_beta) <- diag(P_beta) + pen_prec[seq_len(n_fixed)]
    }

    result$H_beta <- as.matrix(P_beta)
  }

  # Marginal H_beta for spatial-field Laplace via Schur on the joint Hessian.
  # See .marginal_H_beta_spde() / .marginal_H_beta_gp() and issue #16.
  if (return_hessian && !is.null(result$mode) && !is.null(spatial)) {
    if (identical(spatial$type, "spde")) {
      range_val <- result$range %||% spatial$prior_range[1]
      sigma_val <- result$sigma %||% spatial$prior_sigma[1]
      result$H_beta <- tryCatch(
        .marginal_H_beta_spde(
          mode = result$mode, X = X, spatial = spatial,
          family = family, phi = phi,
          n_trials = n_trials, weights = weights,
          range_val = range_val, sigma_val = sigma_val
        ),
        error = function(e) {
          warning("Marginal H_beta (SPDE Schur) failed: ", conditionMessage(e),
                  ". Returning H_beta = NULL.", call. = FALSE)
          NULL
        }
      )
    } else if (identical(spatial$type, "gp")) {
      sigma2_val <- result$sigma2_gp %||% spatial$sigma2_gp %||% 1.0
      phi_gp_val <- result$phi_gp    %||% spatial$phi_gp    %||% 1.0
      result$H_beta <- tryCatch(
        .marginal_H_beta_gp(
          mode = result$mode, X = X, spatial = spatial,
          family = family, phi = phi,
          n_trials = n_trials, weights = weights,
          sigma2_gp = sigma2_val, phi_gp = phi_gp_val
        ),
        error = function(e) {
          warning("Marginal H_beta (NNGP Schur) failed: ", conditionMessage(e),
                  ". Returning H_beta = NULL.", call. = FALSE)
          NULL
        }
      )
    }
  }

  result
}


#' Normalize an optional fixed-effect Gaussian prior
#'
#' Validates `beta_prior` and recycles scalar `mean` / `sd` to length `p`.
#' Returns `NULL` (use the built-in weak prior) or `list(mean, sd)` with both
#' vectors of length `p`. Shared by [tulpa_laplace()] and the EM driver so the
#' validation rules live in one place.
#'
#' @param beta_prior `NULL`, or a list with `sd` (required) and optional `mean`.
#' @param p Number of fixed effects (`ncol(X)`).
#' @keywords internal
.normalize_beta_prior <- function(beta_prior, p) {
  if (is.null(beta_prior)) return(NULL)
  if (!is.list(beta_prior)) {
    stop("`beta_prior` must be NULL or a list with `sd` (and optional `mean`); ",
         "got ", class(beta_prior)[1], ".", call. = FALSE)
  }
  if (is.null(beta_prior$sd)) {
    stop("`beta_prior` must supply `sd` (prior standard deviation on the ",
         "fixed effects).", call. = FALSE)
  }

  recycle <- function(v, nm) {
    v <- as.numeric(v)
    if (length(v) == 1L) return(rep(v, p))
    if (length(v) == p)  return(v)
    stop(sprintf("`beta_prior$%s` must have length 1 or %d (ncol(X)); got %d.",
                 nm, p, length(v)), call. = FALSE)
  }

  sd   <- recycle(beta_prior$sd, "sd")
  mean <- recycle(if (is.null(beta_prior$mean)) 0 else beta_prior$mean, "mean")

  # `sd = +Inf` is allowed and means "no penalty on that coefficient"
  # (precision 1 / sd^2 = 0). The C++ layer maps +Inf -> tau = 0.
  if (any(is.na(sd)) || any(sd <= 0)) {
    stop("`beta_prior$sd` must be positive (Inf allowed = no penalty).",
         call. = FALSE)
  }
  if (any(!is.finite(mean))) {
    stop("`beta_prior$mean` must be finite.", call. = FALSE)
  }

  list(mean = mean, sd = sd)
}


#' Interpret an RE term's covariance specification
#'
#' Maps one `re_list` element to the two representations the Laplace path needs:
#' `pack`, the value the C++ kernel consumes (a length-`n_coefs` marginal-SD
#' vector for a diagonal / uncorrelated term, or a packed lower-triangular
#' Cholesky of length `n_coefs (n_coefs + 1) / 2` for a correlated one), and
#' `Q`, the `n_coefs x n_coefs` RE precision `Sigma^{-1}` used to build the
#' marginal fixed-effect SE. A correlated term is signalled by `r$L` (a lower-
#' triangular Cholesky factor, `Sigma = L L'`) or `r$cov` (the covariance
#' matrix); when present these take precedence over `r$sigma`.
#'
#' @param r One element of a `re_list` (see [tulpa_laplace()]).
#' @return `list(pack, Q, diagonal)`.
#' @keywords internal
.re_cov_spec <- function(r) {
  nc <- r$n_coefs %||% 1L

  if (nc > 1L && (!is.null(r$L) || !is.null(r$cov))) {
    L <- if (!is.null(r$L)) as.matrix(r$L) else t(chol(as.matrix(r$cov)))
    if (nrow(L) != nc || ncol(L) != nc) {
      stop(sprintf("RE term `L`/`cov` must be %d x %d (n_coefs); got %d x %d.",
                   nc, nc, nrow(L), ncol(L)), call. = FALSE)
    }
    # Column-major lower-triangular packing, unpacked by the C++ kernel as
    # L[r, c] = pack[idx] over columns c = 1..nc, rows r = c..nc.
    pack <- L[lower.tri(L, diag = TRUE)]
    # Q = Sigma^{-1} = (L L')^{-1}. t(L) is the upper factor with
    # (t(L))' t(L) = L L' = Sigma, so chol2inv(t(L)) = Sigma^{-1}.
    Q <- chol2inv(t(L))
    return(list(pack = pack, Q = Q, diagonal = FALSE))
  }

  # Diagonal (uncorrelated) covariance: per-coefficient marginal SD.
  sig <- r$sigma
  if (length(sig) == 1L && nc > 1L) sig <- rep(sig, nc)
  list(pack = sig, Q = diag(1 / (sig^2 + 1e-10), nc), diagonal = TRUE)
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
#' @param range Spatial range (NULL -> use `spatial$prior_range[1]`).
#' @param sigma Marginal SD (NULL -> use `spatial$prior_sigma[1]`).
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
    rational_poles_nullable = if (!rat$is_integer) rat$poles else NULL,
    rational_weights_nullable = if (!rat$is_integer) rat$weights else NULL
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
#' @param sigma2_gp Marginal variance (NULL -> 1.0).
#' @param phi_gp Range / decay parameter (NULL -> 1.0).
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

# Neighbor-list form of an adjacency matrix for the Polya-Gamma Gibbs spatial
# samplers. They take `adj_list` as an R list whose j-th element is the 1-based
# neighbor indices of unit j (the C++ subtracts 1 internally), plus the
# `n_neighbors` count -- in contrast to the CSR row_ptr/col_idx form
# adjacency_to_csr_tulpa() builds for the Laplace kernels.
adjacency_to_list_tulpa <- function(adj) {
  if (inherits(adj, "sparseMatrix")) adj <- as.matrix(adj)
  n <- nrow(adj)
  adj_list <- lapply(seq_len(n), function(i) as.integer(which(adj[i, ] != 0)))
  list(adj_list = adj_list,
       n_neighbors = vapply(adj_list, length, integer(1)))
}

# Shared `cpp_pg_binomial_gibbs_*` argument block (the universal GLMM inputs
# every spatial / temporal Polya-Gamma sampler takes: response, design, the iid
# RE block, iteration controls and the fixed-effect / RE-scale priors). Factored
# out so dispatch_gibbs_spatial() and dispatch_gibbs_temporal() assemble it once
# (principle #5) and the per-sampler do.call only adds the structure inputs.
.pg_gibbs_common_args <- function(y, n_trials, X, re_group, n_re_groups,
                                  iter, warmup, thin,
                                  prior_beta_sd, prior_sigma_re_scale,
                                  verbose, n_threads) {
  list(
    y = as.numeric(y), n = as.integer(n_trials), X = X,
    re_group = as.integer(re_group), n_re_groups = as.integer(n_re_groups),
    n_iter = as.integer(iter), n_warmup = as.integer(warmup),
    thin = as.integer(thin),
    prior_beta_sd = prior_beta_sd, prior_sigma_re_scale = prior_sigma_re_scale,
    store_eta = FALSE, verbose = verbose, n_threads = as.integer(n_threads)
  )
}

# Pull the (nn_idx, nn_dist, nn_order, nn) tuple a Polya-Gamma NNGP Gibbs
# sampler consumes out of a validated GP spec's `neighbor_info`. nn_order is
# stored 1-based (ordered-position -> original location index); the C++ kernels
# index 0-based, so subtract 1 -- the same convention as laplace_gp_at(). nn_idx
# / nn_dist are passed through (1-based ordered-position neighbour ids).
.gp_gibbs_nn_inputs <- function(neighbor_info, n_spatial) {
  list(
    nn_idx   = as.matrix(neighbor_info$nn_idx),
    nn_dist  = as.matrix(neighbor_info$nn_dist),
    nn_order = as.integer((neighbor_info$nn_order %||% seq_len(n_spatial)) - 1L),
    nn       = as.integer(ncol(neighbor_info$nn_idx))
  )
}

# The continuous-field Polya-Gamma Gibbs samplers (gp / multiscale_gp) carry NO
# observation->location index: they map observation row i straight to spatial
# location i (`gp_contrib[i] = w[i]` for `i < n_spatial`). So they require one
# observation per unique location, in coordinate order. Surface that constraint
# loudly rather than silently fitting the trailing observations with no field.
# (Flagged for the post-recovery-net refactor that would add an obs->loc map;
# see dev_notes/plan_gibbs_spatial_frontdoor.md.)
.gp_gibbs_require_one_obs_per_loc <- function(spatial, n_obs, label) {
  n_spatial  <- spatial$n_spatial %||% nrow(spatial$unique_coords)
  obs_to_loc <- as.integer(spatial$obs_to_loc %||% seq_len(n_obs))
  if (n_spatial != n_obs || !identical(obs_to_loc, seq_len(n_obs))) {
    stop("The binomial ", label, " Gibbs sampler maps observation i directly to ",
         "location i (it carries no observation->location index), so it needs ",
         "one observation per unique location in coordinate order. Got ", n_obs,
         " observation(s) for ", n_spatial, " location(s). Use mode = 'nested' ",
         "or 'laplace' for repeated-location designs.", call. = FALSE)
  }
  as.integer(n_spatial)
}

# Unit-level RSR projector P_perp = I - Q Q', orthogonalising the spatial field
# against the (unit-aggregated) fixed-effect design so the field cannot absorb
# covariate signal (Reich et al. 2006). X is per observation; collapse it to one
# row per spatial unit (mean over each unit's observations -- the identity for
# the canonical one-observation-per-unit areal design) before projecting.
.rsr_unit_projection <- function(X, spatial_group, n_units) {
  X <- as.matrix(X)
  X_unit <- matrix(0.0, n_units, ncol(X))
  cnt    <- integer(n_units)
  for (i in seq_len(nrow(X))) {
    u <- spatial_group[i]
    X_unit[u, ] <- X_unit[u, ] + X[i, ]
    cnt[u]      <- cnt[u] + 1L
  }
  pos <- cnt > 0
  X_unit[pos, ] <- X_unit[pos, ] / cnt[pos]
  compute_rsr_projection(X_unit)
}

#' Dispatch a spatial Polya-Gamma Gibbs fit to the correct sampler
#'
#' The Gibbs analogue of [dispatch_laplace_spatial()]: routes on `spatial$type`
#' to the matching `cpp_pg_binomial_gibbs_<structure>` sampler, building the
#' neighbour-list / coordinate inputs each one needs. Binomial only (the
#' Polya-Gamma augmentation is for the binomial/logit likelihood).
#' @keywords internal
dispatch_gibbs_spatial <- function(y, n_trials, X, re_group, n_re_groups,
                                   spatial, family,
                                   iter, warmup, thin = 1L,
                                   prior_beta_sd = 10.0,
                                   prior_sigma_re_scale = 2.5,
                                   verbose = FALSE, n_threads = 1L) {
  if (!identical(family, "binomial")) {
    stop("Spatial Gibbs supports family = 'binomial' only; got '", family,
         "'. Use mode = 'laplace' for other families under a spatial field.",
         call. = FALSE)
  }
  # RSR keeps the underlying areal $type (icar/car) and flags $rsr; normalise it
  # to the "rsr" route so the projection is applied rather than a plain areal fit.
  spatial_type <- tolower(spatial$type %||% "")
  if (isTRUE(spatial$rsr)) spatial_type <- "rsr"

  common <- .pg_gibbs_common_args(
    y, n_trials, X, re_group, n_re_groups, iter, warmup, thin,
    prior_beta_sd, prior_sigma_re_scale, verbose, n_threads
  )

  # Areal samplers (icar / bym2 / rsr) share the neighbour-list block.
  if (spatial_type %in% c("icar", "bym2", "rsr")) {
    adj <- spatial$adjacency
    if (is.null(adj)) {
      stop("Spatial Gibbs (", spatial_type, ") needs `spatial$adjacency`.",
           call. = FALSE)
    }
    al <- adjacency_to_list_tulpa(adj)
    areal <- list(
      spatial_group   = as.integer(spatial$spatial_idx %||% seq_len(nrow(adj))),
      n_spatial_units = as.integer(nrow(adj)),
      adj_list        = al$adj_list,
      n_neighbors     = al$n_neighbors
    )
  }

  if (spatial_type == "icar") {
    do.call(cpp_pg_binomial_gibbs_spatial, c(common, areal, list(
      prior_tau_shape = spatial$prior_tau_shape %||% 1.0,
      prior_tau_rate  = spatial$prior_tau_rate  %||% 0.01
    )))
  } else if (spatial_type == "bym2") {
    do.call(cpp_pg_binomial_gibbs_bym2, c(common, areal, list(
      scale_factor              = spatial$scale_factor %||% 1.0,
      prior_sigma_spatial_scale = spatial$prior_sigma_spatial_scale %||% 2.5,
      prior_rho_alpha           = spatial$prior_rho_alpha %||% 0.5,
      prior_rho_beta            = spatial$prior_rho_beta  %||% 0.5
    )))
  } else if (spatial_type == "rsr") {
    # Restricted spatial regression: an ICAR field projected orthogonal to the
    # covariates each sweep. Reuse a caller-supplied projector if validate_rsr()
    # built one, else assemble the unit-level P_perp from the model design.
    rsr_n  <- areal$n_spatial_units
    P_perp <- spatial$rsr_projection %||%
      .rsr_unit_projection(X, areal$spatial_group, rsr_n)
    if (nrow(P_perp) != rsr_n || ncol(P_perp) != rsr_n) {
      stop("RSR projection matrix is ", nrow(P_perp), "x", ncol(P_perp),
           " but must be ", rsr_n, "x", rsr_n, " (one row/col per spatial unit).",
           call. = FALSE)
    }
    do.call(cpp_pg_binomial_gibbs_rsr, c(common, areal, list(
      # Row-major flatten for the C++ `rsr_projection[s * rsr_n + k]` indexing;
      # P_perp is symmetric so t() is a no-op but keeps the convention explicit.
      rsr_projection  = as.numeric(t(P_perp)),
      rsr_n           = as.integer(rsr_n),
      prior_tau_shape = spatial$prior_tau_shape %||% 1.0,
      prior_tau_rate  = spatial$prior_tau_rate  %||% 0.01
    )))
  } else if (spatial_type %in% c("gp", "nngp")) {
    if (is.null(spatial$neighbor_info)) {
      stop("Spatial Gibbs (", spatial_type, ") needs a validated spatial_gp() ",
           "spec (neighbor_info is NULL). Call validate_gp(spatial, data) first, ",
           "or fit through tulpa() which validates automatically.", call. = FALSE)
    }
    n_spatial <- .gp_gibbs_require_one_obs_per_loc(spatial, length(y), spatial_type)
    nn_in     <- .gp_gibbs_nn_inputs(spatial$neighbor_info, n_spatial)
    do.call(cpp_pg_binomial_gibbs_gp, c(common, list(
      coords         = as.matrix(spatial$unique_coords),
      nn_idx         = nn_in$nn_idx,
      nn_dist        = nn_in$nn_dist,
      nn_order       = nn_in$nn_order,
      n_spatial      = n_spatial,
      nn             = nn_in$nn,
      sigma2_gp_init = spatial$sigma2_gp %||% 1.0,
      phi_gp_init    = spatial$phi_gp %||% 1.0,
      cov_type       = gp_cov_type_for_laplace(spatial)
    )))
  } else if (spatial_type %in% c("multiscale", "multiscale_gp")) {
    # Both scales reuse the shared NNGP kriging conditional
    # (tulpa::pg_nngp_conditional), so cov_type (exponential / Matern 3/2 / 5/2)
    # is honoured exactly as in the single-scale sampler.
    if (is.null(spatial$neighbor_info_local) ||
        is.null(spatial$neighbor_info_regional)) {
      stop("Spatial Gibbs (multiscale GP) needs a validated spatial_multiscale() ",
           "spec (neighbor_info_local / _regional are NULL). Call ",
           "validate_gp(spatial, data) first, or fit through tulpa().",
           call. = FALSE)
    }
    n_spatial <- .gp_gibbs_require_one_obs_per_loc(spatial, length(y), "multiscale GP")
    loc <- .gp_gibbs_nn_inputs(spatial$neighbor_info_local, n_spatial)
    reg <- .gp_gibbs_nn_inputs(spatial$neighbor_info_regional, n_spatial)
    rl  <- spatial$range_local    %||% c(0.01, 1)
    rr  <- spatial$range_regional %||% c(1, 10)
    do.call(cpp_pg_binomial_gibbs_multiscale_gp, c(common, list(
      coords               = as.matrix(spatial$unique_coords),
      nn_idx_local         = loc$nn_idx,
      nn_dist_local        = loc$nn_dist,
      nn_order_local       = loc$nn_order,
      nn_local             = loc$nn,
      nn_idx_regional      = reg$nn_idx,
      nn_dist_regional     = reg$nn_dist,
      nn_order_regional    = reg$nn_order,
      nn_regional          = reg$nn,
      n_spatial            = n_spatial,
      sigma2_local_init    = 1.0,
      phi_local_init       = mean(rl),
      sigma2_regional_init = 1.0,
      phi_regional_init    = mean(rr),
      cov_type             = gp_cov_type_for_laplace(spatial),
      prior_phi_local_lower    = rl[1], prior_phi_local_upper    = rl[2],
      prior_phi_regional_lower = rr[1], prior_phi_regional_upper = rr[2]
    )))
  } else {
    stop("Spatial Gibbs not wired for type '", spatial_type, "'. Supported: ",
         "icar, bym2, rsr, gp/nngp, multiscale_gp. ",
         "See dev_notes/plan_gibbs_spatial_frontdoor.md.", call. = FALSE)
  }
}


#' Dispatch a temporal Polya-Gamma Gibbs fit
#'
#' The temporal analogue of [dispatch_gibbs_spatial()]: maps a validated
#' [temporal_multiscale()] spec onto the multiscale temporal Polya-Gamma
#' sampler (`cpp_pg_binomial_gibbs_temporal`), which composes an additive
#' RW1 trend + cyclic-RW1 seasonal + AR1/IID short-term decomposition. Binomial
#' only. The C++ kernel implements an RW1 trend (rw2 is rejected here rather than
#' silently downgraded).
#' @keywords internal
dispatch_gibbs_temporal <- function(y, n_trials, X, re_group, n_re_groups,
                                    temporal, family,
                                    iter, warmup, thin = 1L,
                                    prior_beta_sd = 10.0,
                                    prior_sigma_re_scale = 2.5,
                                    verbose = FALSE, n_threads = 1L) {
  if (!identical(family, "binomial")) {
    stop("Temporal Gibbs supports family = 'binomial' only; got '", family,
         "'. Use mode = 'laplace' for other families under a temporal field.",
         call. = FALSE)
  }
  if (is.null(temporal$time_index) || is.null(temporal$n_times)) {
    stop("Temporal Gibbs needs a validated temporal_multiscale() spec ",
         "(time_index / n_times are NULL). Call ",
         "validate_temporal_multiscale(temporal, data) first, or fit through ",
         "tulpa().", call. = FALSE)
  }

  trend_type <- switch(temporal$trend %||% "none",
    none = 0L, rw1 = 1L,
    rw2  = stop("The temporal Gibbs sampler implements an RW1 trend; rw2 is not ",
                "wired. Use trend = 'rw1', or mode = 'laplace' for an RW2 trend.",
                call. = FALSE),
    stop("Unknown temporal trend '", temporal$trend, "'.", call. = FALSE))
  short_type <- switch(temporal$short_term %||% "none",
    none = 0L, ar1 = 1L, iid = 2L,
    stop("Unknown temporal short_term '", temporal$short_term, "'.", call. = FALSE))

  common <- .pg_gibbs_common_args(
    y, n_trials, X, re_group, n_re_groups, iter, warmup, thin,
    prior_beta_sd, prior_sigma_re_scale, verbose, n_threads
  )
  do.call(cpp_pg_binomial_gibbs_temporal, c(common, list(
    time_idx        = as.integer(temporal$time_index),
    n_times         = as.integer(temporal$n_times),
    seasonal_period = as.integer(temporal$seasonal %||% 0L),
    trend_type      = trend_type,
    short_type      = short_type
  )))
}


#' Compute GLM working weights for Laplace Hessian
#'
#' Thin wrapper over the family-ops registry ([family_weight()]) so the weight
#' formulas live in exactly one place (`R/family_loglik.R`). Unknown families
#' fall back to unit weights, preserving the historical permissive behaviour.
#' @keywords internal
glmm_weights <- function(eta, family, n_trials = NULL, phi = 1.0) {
  if (is.null(.FAMILY_OPS[[family]])) return(rep(1, length(eta)))
  as.numeric(family_weight(eta, family, n_trials, phi))
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
#' @param spatial Optional spatial spec. When supplied the fit routes to the
#'   matching spatial Polya-Gamma Gibbs sampler via [dispatch_gibbs_spatial()]
#'   (binomial only); `group`/`n_groups` are the iid random-effect block carried
#'   alongside the field. Supported `type`s:
#'   * areal -- `"icar"`, `"bym2"`, `"rsr"`: a list with `type`, `adjacency` and
#'     a 1-based `spatial_idx` per observation (e.g.
#'     `list(type = "icar", adjacency = W, spatial_idx = unit)`). `"rsr"` reuses
#'     `spatial$rsr_projection` if present, else builds the unit-level projector
#'     from the design.
#'   * continuous -- `"gp"`/`"nngp"` (a validated [spatial_gp()] spec) and
#'     `"multiscale_gp"` (a validated [spatial_multiscale()] spec). These
#'     samplers carry no observation->location map, so they require one
#'     observation per unique location in coordinate order.
#' @param temporal Optional temporal spec: a validated [temporal_multiscale()]
#'   object. Routes to the multiscale temporal Polya-Gamma sampler via
#'   [dispatch_gibbs_temporal()] (binomial only; RW1 trend + cyclic seasonal +
#'   AR1/IID short-term). Cannot be combined with `spatial`.
#' @param verbose Print progress
#' @param n_threads Number of threads
#'
#' @return List with beta draws, RE draws, sigma_re draws (plus the spatial
#'   field draws when `spatial` is supplied)
#'
#' @export
tulpa_gibbs <- function(y, n_trials, X, group, n_groups,
                        family = "binomial",
                        iter = 2000L, warmup = 1000L,
                        prior_beta_sd = 10.0, prior_sigma_scale = 2.5,
                        spatial = NULL, temporal = NULL,
                        verbose = TRUE, n_threads = 1L) {

  # Spatial / temporal field present: route to the matching Polya-Gamma Gibbs
  # sampler (the Gibbs analogue of tulpa_laplace(spatial = ...)). `group` /
  # `n_groups` become the iid random-effect block carried alongside the field.
  # There is no joint spatial+temporal Gibbs sampler -- reject rather than
  # silently dropping one field.
  if (!is.null(spatial) && !is.null(temporal)) {
    stop("Combined spatial + temporal Gibbs is not available; supply only one ",
         "of `spatial` / `temporal`.", call. = FALSE)
  }
  if (!is.null(spatial)) {
    return(dispatch_gibbs_spatial(
      y = y, n_trials = if (is.null(n_trials)) rep(1L, length(y)) else n_trials,
      X = X, re_group = group, n_re_groups = n_groups,
      spatial = spatial, family = family,
      iter = iter, warmup = warmup,
      prior_beta_sd = prior_beta_sd, prior_sigma_re_scale = prior_sigma_scale,
      verbose = verbose, n_threads = n_threads
    ))
  }
  if (!is.null(temporal)) {
    return(dispatch_gibbs_temporal(
      y = y, n_trials = if (is.null(n_trials)) rep(1L, length(y)) else n_trials,
      X = X, re_group = group, n_re_groups = n_groups,
      temporal = temporal, family = family,
      iter = iter, warmup = warmup,
      prior_beta_sd = prior_beta_sd, prior_sigma_re_scale = prior_sigma_scale,
      verbose = verbose, n_threads = n_threads
    ))
  }

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
