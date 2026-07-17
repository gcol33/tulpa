#' Fit a model via Laplace approximation
#'
#' @description
#' General-purpose Laplace approximation for latent Gaussian models.
#' Finds the mode of the latent field (beta + random effects) and
#' returns the Laplace-approximated marginal likelihood.
#'
#' This is the public API for model packages (tulpaGlmm, tulpaObs, etc.)
#' to call tulpa's Laplace engine. As the low-level engine entry point (not a
#' front-door fitter), it keeps its numerical controls (`max_iter`, `tol`,
#' `n_threads`, `return_hessian`) inline in the signature rather than in a
#' `control` list, so callers assembling many solves pass them positionally.
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
#' @param phi Dispersion parameter. For `gaussian` / `lognormal` this is the
#'   residual VARIANCE (matching the R-side family registry and [tulpa()]);
#'   the SD-parameterized compiled kernels receive `sqrt(phi)` internally.
#'   For `neg_binomial_2` the size, `beta` the precision, `t` the scale.
#' @param phi2 Optional second dispersion: the Student-t degrees of freedom
#'   (`family = "t"`; default 4 when `NULL`). Non-spatial path only.
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
#' @examples
#' set.seed(1)
#' n <- 200L
#' X <- cbind(1, rnorm(n))
#' eta <- X %*% c(-0.3, 0.8)
#' y <- rbinom(n, 1, plogis(eta))
#' fit <- tulpa_laplace(y, rep(1L, n), X, family = "binomial")
#' fit$mode          # posterior mode of the fixed effects
#' @export
tulpa_laplace <- function(y, n_trials, X,
                          re_list = list(),
                          family = "binomial",
                          phi = 1.0,
                          phi2 = NULL,
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

  # tulpa_laplace's `phi` is the residual VARIANCE for gaussian / lognormal
  # (the convention of the R-side registry, which builds H_beta below); the
  # compiled kernels parameterize by the residual SD. Convert once here so the
  # mode-finding and the Hessian describe the same model (the kernel used to
  # receive the variance where it expected the SD).
  phi_kernel <- if (family %in% c("gaussian", "lognormal")) sqrt(phi) else phi
  if (!is.null(phi2)) {
    .phi2_or_stop(family, phi2)
    if (!is.null(spatial)) {
      stop("`phi2` is not threaded through the spatial Laplace kernels; drop ",
           "`phi2` or `spatial`.", call. = FALSE)
    }
  }

  # Validate
  stopifnot(is.numeric(y) || is.integer(y))
  stopifnot(is.matrix(X))
  stopifnot(nrow(X) == n_obs)

  # Normalize the optional fixed-effect prior to length-p mean / sd vectors.
  bp <- .normalize_beta_prior(beta_prior, n_fixed)

  .validate_family_phi(family, phi)
  if (identical(family, "beta")) {
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
      spatial, family, phi_kernel, max_iter, tol, n_threads, offset = offset
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
      phi = phi_kernel,
      max_iter = as.integer(max_iter),
      tol = tol,
      n_threads = as.integer(n_threads),
      re_Z_list = if (has_design) re_Z_list else NULL,
      re_ncoefs = if (has_design) re_ncoefs else NULL,
      weights = weights,
      offset = offset,
      beta_prior_mean = if (is.null(bp)) NULL else bp$mean,
      beta_prior_sd   = if (is.null(bp)) NULL else bp$sd,
      return_re_cov   = isTRUE(return_re_cov),
      phi2 = phi2 %||% NA_real_
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
    spatial$type %in% c("spde", "gp", "car_proper", "hsgp")

  # Compute Hessian for fixed-effect block if requested
  if (return_hessian && !is.null(result$mode) && !is_spatial_field) {
    mode_vec <- result$mode
    beta <- mode_vec[seq_len(n_fixed)]
    re_vals <- mode_vec[-seq_len(n_fixed)]

    # Linear predictor at mode: eta = X*beta + offset + sum_k Z_k %*% u_k.
    # The observation offset must enter here: for non-gaussian families the
    # GLM weight W depends on eta, so omitting it curves H_beta at the wrong
    # linear predictor. The mode stores term k as
    # [g1_c1, g1_c2, ..., g2_c1, ...] (n_groups * n_coefs), so a slope term
    # contributes Z_k[i, ] %*% u_{g(i)}, not just an intercept.
    eta <- as.numeric(X %*% beta) + (offset %||% 0)

    if (length(re_list) > 0) {
      lat_off <- 0L
      for (k in seq_along(re_list)) {
        r  <- re_list[[k]]
        nc <- r$n_coefs %||% 1L
        # Z is the term's design: a supplied Z (slopes, incl. a single
        # `(0 + x | g)`) or the intercept indicator (column of 1s) when absent.
        Zk <- r$Z %||% matrix(1, n_obs, 1L)
        u_k <- re_vals[lat_off + seq_len(r$n_groups * nc)]
        u_mat <- matrix(u_k, ncol = nc, byrow = TRUE)   # row g = (c1, ..., cnc)
        eta <- eta + rowSums(Zk * u_mat[r$idx, , drop = FALSE])
        lat_off <- lat_off + r$n_groups * nc
      }
    }

    # GLM weights
    W <- glmm_weights(eta, family, n_trials, phi, phi2)

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
          n_trials = n_trials, weights = weights, offset = offset,
          range_val = range_val, sigma_val = sigma_val,
          re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re
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
          n_trials = n_trials, weights = weights, offset = offset,
          sigma2_gp = sigma2_val, phi_gp = phi_gp_val,
          re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re
        ),
        error = function(e) {
          warning("Marginal H_beta (NNGP Schur) failed: ", conditionMessage(e),
                  ". Returning H_beta = NULL.", call. = FALSE)
          NULL
        }
      )
    } else if (identical(spatial$type, "car_proper")) {
      result$H_beta <- tryCatch(
        .marginal_H_beta_car_proper(
          mode = result$mode, X = X, spatial = spatial,
          family = family, phi = phi,
          n_trials = n_trials, weights = weights, offset = offset,
          tau = result$tau %||% 1.0, rho = result$rho %||% 0.5,
          re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re
        ),
        error = function(e) {
          warning("Marginal H_beta (CAR_proper Schur) failed: ",
                  conditionMessage(e), ". Returning H_beta = NULL.", call. = FALSE)
          NULL
        }
      )
    } else if (identical(spatial$type, "hsgp")) {
      result$H_beta <- tryCatch(
        .marginal_H_beta_hsgp(
          mode = result$mode, X = X, spatial = spatial,
          family = family, phi = phi,
          n_trials = n_trials, weights = weights, offset = offset,
          phi_basis = result$phi_basis, lambda_eig = result$lambda_eig,
          sigma2 = result$sigma2 %||% 1.0,
          lengthscale = result$lengthscale %||% 1.0,
          re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re
        ),
        error = function(e) {
          warning("Marginal H_beta (HSGP Schur) failed: ", conditionMessage(e),
                  ". Returning H_beta = NULL.", call. = FALSE)
          NULL
        }
      )
    }
  }

  .finalize_fit(result, backend = "laplace",
                n_fixed = n_fixed, fixed_names = colnames(X))
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
                                     max_iter, tol, n_threads, offset = NULL) {

  spatial_type <- spatial$type
  off_arg <- if (is.null(offset)) NULL else as.numeric(offset)

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
      n_threads = as.integer(n_threads),
      offset_nullable = off_arg
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
      n_threads = as.integer(n_threads),
      offset_nullable = off_arg
    )
  } else if (spatial_type == "spde") {
    # SPDE Laplace at fixed hyperparameters (uses spec's prior modes), with an
    # optional iid RE block threaded into the [beta, re, w_mesh] latent layout.
    # Nested integration over (range, sigma) is opt-in via fit_spde().
    laplace_spde_at(
      y = y, n_trials = n_trials, X = X, spatial = spatial,
      family = family, phi = phi,
      range = NULL, sigma = NULL,
      re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
      max_iter = max_iter, tol = tol, n_threads = n_threads,
      offset = offset
    )
  } else if (spatial_type == "gp") {
    # NNGP Laplace at fixed hyperparameters, with an optional iid RE block
    # threaded into cpp_laplace_fit_gp's [beta, re, w_gp] latent layout.
    laplace_gp_at(
      y = y, n_trials = n_trials, X = X, spatial = spatial,
      family = family, phi = phi,
      sigma2_gp = NULL, phi_gp = NULL,
      re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
      max_iter = max_iter, tol = tol, n_threads = n_threads,
      offset = offset
    )
  } else if (spatial_type == "car_proper") {
    # Proper-CAR Laplace at a fixed (tau, rho), the conditional counterpart of
    # the nested CAR_proper integrator (same make_car_proper_latent_blocks
    # factory). tau / rho default from the spec; recorded on the result.
    laplace_car_proper_at(
      y = y, n_trials = n_trials, X = X, spatial = spatial,
      family = family, phi = phi, tau = NULL, rho = NULL,
      re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
      max_iter = max_iter, tol = tol, n_threads = n_threads,
      offset = offset
    )
  } else if (spatial_type == "hsgp") {
    # HSGP Laplace at a fixed (sigma2, lengthscale), the conditional counterpart
    # of the nested HSGP integrator (same make_hsgp_block basis). Defaults from
    # the spec; recorded on the result.
    laplace_hsgp_at(
      y = y, n_trials = n_trials, X = X, spatial = spatial,
      family = family, phi = phi, sigma2 = NULL, lengthscale = NULL,
      re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
      max_iter = max_iter, tol = tol, n_threads = n_threads,
      offset = offset
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
                             re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                             max_iter = 100L, tol = 1e-6, n_threads = 1L,
                             offset = NULL) {
  if (is.null(range)) range <- spatial$prior_range[1]
  if (is.null(sigma)) sigma <- spatial$prior_sigma[1]

  # Fractional nu: the operator-based rational SPDE (Bolin & Kirchner 2020) with
  # BRASIL coefficients (gcol33/tulpa#71). Assembled in R (the validated oracle)
  # and solved by the precomputed C++ fit; the integer branch below is the exact
  # FEM construction. Both branches return the same [beta (p), re (n_re_groups),
  # mesh (n_mesh)] mode layout, so the beta / spatial_effects split is shared.
  if (.spde_nu_is_fractional(spatial$nu)) {
    result <- .spde_laplace_fractional_at(
      y = y, n_trials = n_trials, X = X, spatial = spatial,
      family = family, phi = phi, range = range, sigma = sigma,
      re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
      max_iter = max_iter, tol = tol, n_threads = n_threads, offset = offset,
      order = spatial$rational_order %||% 2L
    )
  } else {
    kappa <- sqrt(8 * spatial$nu) / range
    tau_spde <- 1.0 / (sqrt(4 * pi) * kappa * sigma)
    alpha <- as.integer(round(spatial$nu)) + 1L
    rat <- rational_spde_coefficients(spatial$nu)

    # iid RE block laid out between beta and the mesh field. No RE -> zero dims.
    if (is.null(re_idx)) re_idx <- rep(0L, length(y))

    result <- cpp_laplace_fit_spde(
      y = as.numeric(y),
      n_trials = as.integer(n_trials %||% rep(1L, length(y))),
      X = as.matrix(X),
      re_idx = as.numeric(re_idx),
      n_re_groups = as.integer(n_re_groups),
      sigma_re = sigma_re,
      A_x = spatial$A_x, A_i = spatial$A_i, A_p = spatial$A_p,
      n_obs = length(y), n_mesh = spatial$n_mesh,
      C0_diag = spatial$C0_diag,
      G1_x = spatial$G1_x, G1_i = spatial$G1_i, G1_p = spatial$G1_p,
      kappa = kappa, tau_spde = tau_spde,
      family = family, phi = phi, alpha = alpha,
      max_iter = as.integer(max_iter), tol = tol,
      n_threads = as.integer(n_threads),
      rational_poles_nullable = if (!rat$is_integer) rat$poles else NULL,
      rational_weights_nullable = if (!rat$is_integer) rat$weights else NULL,
      offset_nullable = if (is.null(offset)) NULL else as.numeric(offset)
    )

    result$range <- range
    result$sigma <- sigma
    result$spatial <- spatial
  }

  # Split the mode into fixed effects and the mesh field so every caller (the
  # fit_spde single-point branch and the nested CCD refit) reads beta /
  # spatial_effects directly instead of re-slicing the mode.
  p <- ncol(X)
  mesh_start <- p + as.integer(n_re_groups)
  result$beta <- result$mode[seq_len(p)]
  result$spatial_effects <-
    result$mode[(mesh_start + 1L):(mesh_start + spatial$n_mesh)]
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
         "got nu = ", format(nu), ". Set nu = 1.5 or 2.5.",
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
#' off the validated spec -- call `validate_gp(spatial, data)` first if
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
                           re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                           max_iter = 100L, tol = 1e-6, n_threads = 1L,
                           offset = NULL) {
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

  # iid RE block laid out between beta and the GP field (cpp_laplace_fit_gp's
  # n_re_groups > 0 path). No RE -> zero dims, dummy re_idx.
  if (is.null(re_idx)) re_idx <- rep(0L, length(y))

  result <- cpp_laplace_fit_gp(
    y = as.numeric(y),
    n = as.integer(n_trials %||% rep(1L, length(y))),
    X = as.matrix(X),
    re_idx = as.numeric(re_idx),
    n_re_groups = as.integer(n_re_groups),
    sigma_re = sigma_re,
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
    n_threads = as.integer(n_threads),
    offset_nullable = if (is.null(offset)) NULL else as.numeric(offset),
    # Per-obs 1-based location index so repeated coordinates share one field
    # node (n_spatial unique locations < N). NULL keeps the identity map.
    obs_to_loc_nullable = if (is.null(spatial$obs_to_loc)) NULL else
      as.integer(spatial$obs_to_loc)
  )

  result$sigma2_gp <- sigma2_gp
  result$phi_gp <- phi_gp
  result$spatial <- spatial
  result
}


#' Proper-CAR Laplace at given hyperparameters
#'
#' Single-point Laplace for a proper-CAR areal field at a fixed `(tau, rho)`,
#' the conditional counterpart of the nested CAR_proper integrator. Reuses the
#' shared `make_car_proper_latent_blocks` factory + dense spec solver via
#' `cpp_laplace_fit_car_proper`, so the mode + log-marginal equal the nested
#' kernel at that one grid cell. `tau` / `rho` default to the spec's fields (or
#' `tau = 1`, `rho` = midpoint of the eigenvalue-derived `rho_bounds`) and are
#' recorded on the result.
#' @keywords internal
laplace_car_proper_at <- function(y, n_trials, X, spatial,
                                  family = "binomial", phi = 1.0,
                                  tau = NULL, rho = NULL,
                                  re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                                  max_iter = 100L, tol = 1e-6, n_threads = 1L,
                                  offset = NULL) {
  adj <- spatial$adjacency
  if (is.null(adj)) stop("car_proper spec needs an `adjacency`.", call. = FALSE)
  n_units <- nrow(as.matrix(adj))
  csr <- adjacency_to_csr_tulpa(adj)
  # Exact extraction: `spatial$rho` would partial-match `rho_bounds` (length 2).
  if (is.null(tau)) tau <- spatial[["tau"]] %||% 1.0
  if (is.null(rho)) rho <- spatial[["rho"]] %||% mean(spatial[["rho_bounds"]] %||% c(0, 1))
  if (is.null(re_idx)) re_idx <- rep(0L, length(y))

  result <- cpp_laplace_fit_car_proper(
    y = as.numeric(y), n = as.integer(n_trials %||% rep(1L, length(y))),
    X = as.matrix(X), re_idx = as.numeric(re_idx),
    n_re_groups = as.integer(n_re_groups), sigma_re = sigma_re,
    spatial_idx = as.integer(spatial$spatial_idx %||% seq_len(n_units)),
    n_spatial_units = as.integer(n_units),
    adj_row_ptr = as.integer(csr$row_ptr), adj_col_idx = as.integer(csr$col_idx),
    n_neighbors = as.integer(csr$n_neighbors),
    tau_spatial = as.numeric(tau), rho = as.numeric(rho),
    family = family, phi = phi,
    max_iter = as.integer(max_iter), tol = tol, n_threads = as.integer(n_threads),
    offset_nullable = if (is.null(offset)) NULL else as.numeric(offset)
  )
  result$tau <- tau
  result$rho <- rho
  result$spatial <- spatial
  result
}


#' HSGP Laplace at given hyperparameters
#'
#' Single-point Laplace for a Hilbert-space GP field at a fixed
#' `(sigma2, lengthscale)`, the conditional counterpart of the nested HSGP
#' integrator. Reuses the shared `make_hsgp_block` factory + dense spec solver
#' (DENSE_BASIS scatter) via `cpp_laplace_fit_hsgp`, so the mode equals the
#' nested kernel at that one grid cell. `sigma2` / `lengthscale` default to the
#' spec's fields (or `1`) and are recorded on the result.
#' @keywords internal
laplace_hsgp_at <- function(y, n_trials, X, spatial,
                            family = "binomial", phi = 1.0,
                            sigma2 = NULL, lengthscale = NULL,
                            re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                            max_iter = 100L, tol = 1e-6, n_threads = 1L,
                            offset = NULL) {
  cm <- spatial$coords_matrix
  if (is.null(cm)) {
    stop("spatial_gp(approx='hsgp') spec is unvalidated (coords_matrix is NULL). Call ",
         "validate_hsgp(spatial, data) first, or fit through tulpa() which ",
         "validates automatically.", call. = FALSE)
  }
  # Exact extraction: `spatial$c` would partial-match `coords_matrix`.
  basis <- cpp_hsgp_basis_2d(as.matrix(cm), as.integer(spatial[["m"]]),
                             as.numeric(spatial[["c"]]))
  if (is.null(sigma2))      sigma2      <- spatial[["sigma2"]]      %||% 1.0
  if (is.null(lengthscale)) lengthscale <- spatial[["lengthscale"]] %||% 1.0
  if (is.null(re_idx)) re_idx <- rep(0L, length(y))

  result <- cpp_laplace_fit_hsgp(
    y = as.numeric(y), n = as.integer(n_trials %||% rep(1L, length(y))),
    X = as.matrix(X), re_idx = as.numeric(re_idx),
    n_re_groups = as.integer(n_re_groups), sigma_re = sigma_re,
    phi_basis = basis$phi_basis, lambda_eig = basis$lambda_eig,
    sigma2 = as.numeric(sigma2), lengthscale = as.numeric(lengthscale),
    family = family, phi = phi,
    max_iter = as.integer(max_iter), tol = tol, n_threads = as.integer(n_threads),
    offset_nullable = if (is.null(offset)) NULL else as.numeric(offset)
  )
  result$sigma2      <- sigma2
  result$lengthscale <- lengthscale
  result$phi_basis   <- basis$phi_basis
  result$lambda_eig  <- basis$lambda_eig
  result$spatial     <- spatial
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
