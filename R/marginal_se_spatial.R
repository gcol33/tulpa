#' Marginal fixed-effect Hessian for spatial-field Laplace fits
#'
#' Closes the SE gap surfaced in issue #16: the single-arm Laplace path
#' previously returned `H_beta = NULL` for SPDE and NNGP because the raw
#' fixed-effect block of the joint Hessian gives the *conditional*
#' precision on \eqn{\beta \mid u^*}, which under-states uncertainty.
#'
#' The correct marginal precision comes from a Schur complement on the
#' joint Hessian at the mode:
#'
#' \deqn{H_\beta^{\mathrm{marg}} = X'WX - X'WZ (Z'WZ + Q_u)^{-1} Z'WX}
#'
#' where \eqn{Z} is the obs->latent map (SPDE: projection matrix A;
#' NNGP: indicator from obs to unique-location field) and \eqn{Q_u}
#' is the spatial precision at the fitted hyperparameters.
#'
#' These helpers rebuild the spatial precision in R from the spec and
#' fitted hyperparameters, then solve via sparse Cholesky. The shape
#' matches what `cpp_laplace_fit_spde` / `cpp_laplace_fit_gp` use
#' internally -- Q construction here is a 1:1 port of `spde_qbuilder.h`
#' (orphan ridge included).
#'
#' @keywords internal
#' @name marginal_se_spatial
NULL


# --- SPDE precision rebuild (mirrors src/spde_qbuilder.h) -----------------

#' @keywords internal
.spde_precision_Q <- function(spatial, kappa, tau_spde) {
  n_mesh <- spatial$n_mesh
  C0     <- as.numeric(spatial$C0_diag)
  G1     <- as(spatial$G, "CsparseMatrix")

  c0_eps   <- 1e-15
  is_orph  <- C0 <= c0_eps
  c0_inv   <- ifelse(C0 > c0_eps, 1 / C0, 0)
  Cmat     <- Matrix::Diagonal(n_mesh, C0)
  CinvDiag <- Matrix::Diagonal(n_mesh, c0_inv)

  rat <- rational_spde_coefficients(spatial$nu)
  k2   <- kappa * kappa
  tau2 <- tau_spde * tau_spde

  if (isTRUE(rat$is_integer)) {
    alpha <- as.integer(round(rat$alpha))
    if (alpha == 1L) {
      eps_ridge <- 1e-2
      Q <- tau2 * ((k2 + eps_ridge) * Cmat + G1)
    } else {
      # alpha == 2 (the default, nu = 1): Q = tau² * (κ⁴C + 2κ²G + GC⁻¹G)
      k4  <- k2 * k2
      G2  <- G1 %*% CinvDiag %*% G1
      Q   <- tau2 * (k4 * Cmat + 2 * k2 * G1 + G2)
    }
  } else {
    # Fractional alpha — weighted sum of shifted alpha=2 terms
    G2 <- G1 %*% CinvDiag %*% G1
    Q  <- Matrix::sparseMatrix(
      i = integer(0), j = integer(0), x = numeric(0),
      dims = c(n_mesh, n_mesh)
    )
    for (j in seq_along(rat$poles)) {
      k2s <- k2 + rat$poles[j]
      k4s <- k2s * k2s
      Q <- Q + tau2 * rat$weights[j] * (k4s * Cmat + 2 * k2s * G1 + G2)
    }
  }

  if (any(is_orph)) {
    idx <- which(is_orph)
    Q <- Q + Matrix::sparseMatrix(
      i = idx, j = idx, x = 1.0, dims = c(n_mesh, n_mesh)
    )
  }

  Matrix::forceSymmetric(Q)
}


# --- Schur complement for SPDE marginal H_beta ----------------------------

# --- NNGP precision rebuild (mirrors src/gpu_nngp_laplace.h logic) --------

#' @keywords internal
.nngp_cov_fn <- function(cov_type, sigma2, phi_gp) {
  if (cov_type == 0L) {
    function(d) sigma2 * exp(-d / phi_gp)
  } else if (cov_type == 1L) {
    function(d) {
      x <- sqrt(3) * d / phi_gp
      sigma2 * (1 + x) * exp(-x)
    }
  } else if (cov_type == 2L) {
    function(d) {
      x <- sqrt(5) * d / phi_gp
      sigma2 * (1 + x + x * x / 3) * exp(-x)
    }
  } else {
    stop("Unknown cov_type ", cov_type, call. = FALSE)
  }
}

#' Build NNGP precision Λ = (I - A)' D⁻¹ (I - A) at given hyperparameters.
#'
#' Mirrors the algorithm in `src/gpu_nngp_laplace.h::batch_nngp_scatter` plus
#' `apply_nngp_full_prior_dense` — for each Vecchia row i (in NNGP order):
#'   - resolve conditioning set N(i) via `nn_idx` / `nn_order`
#'   - form the n_nb × n_nb cov matrix C and the n_nb cov vector c
#'   - solve C alpha = c
#'   - v_i = sigma² - c' alpha
#'   - β_i is 1 at i and -alpha_k at each neighbor → Λ += β_i β_i' / v_i
#'
#' Identifiers follow the C++ side: `nn_idx[i, k]` is a 1-based NNGP-order
#' index, `nn_order[j]` maps NNGP-order j (1-based) to obs idx (1-based).
#'
#' @keywords internal
.nngp_precision_Q <- function(spatial, sigma2_gp, phi_gp) {
  ni        <- spatial$neighbor_info
  n_spatial <- spatial$n_spatial %||% nrow(spatial$unique_coords)
  nn        <- spatial$nn %||% ncol(ni$nn_idx)
  cov_type  <- gp_cov_type_for_laplace(spatial)
  cov_fn    <- .nngp_cov_fn(cov_type, sigma2_gp, phi_gp)

  nn_idx   <- ni$nn_idx
  nn_dist  <- ni$nn_dist
  nn_order <- ni$nn_order %||% seq_len(n_spatial)
  coords   <- as.matrix(spatial$unique_coords)

  # Triplet accumulator (pre-size: each row contributes (n_nb+1)^2 entries)
  cap <- sum(rowSums(nn_idx > 0) + 1L)^2
  ii <- integer(cap); jj <- integer(cap); xx <- numeric(cap); pos <- 0L

  for (i_nngp in seq_len(n_spatial)) {
    obs_focal <- nn_order[i_nngp]
    if (obs_focal < 1L || obs_focal > n_spatial) next

    active <- nn_idx[i_nngp, ] > 0
    n_nb   <- sum(active)
    if (n_nb == 0L) {
      pos <- pos + 1L
      ii[pos] <- obs_focal; jj[pos] <- obs_focal; xx[pos] <- 1 / sigma2_gp
      next
    }

    nb_nngp <- nn_idx[i_nngp, active]
    nb_obs  <- nn_order[nb_nngp]

    c_vec <- cov_fn(nn_dist[i_nngp, seq_len(n_nb)])

    if (n_nb == 1L) {
      C_mat <- matrix(sigma2_gp, 1, 1)
    } else {
      coords_nb <- coords[nb_obs, , drop = FALSE]
      D <- as.matrix(dist(coords_nb))
      C_mat <- cov_fn(D)
      diag(C_mat) <- sigma2_gp
    }

    alpha_vec <- as.numeric(solve(C_mat, c_vec))
    v_i       <- max(1e-10, sigma2_gp - sum(c_vec * alpha_vec))
    tau_i     <- 1 / v_i

    idxs  <- c(obs_focal, nb_obs)
    coefs <- c(1, -alpha_vec)
    m     <- length(idxs)
    outer_xx <- tau_i * tcrossprod(coefs)

    new_n <- m * m
    rng   <- pos + seq_len(new_n)
    ii[rng] <- rep(idxs, each  = m)
    jj[rng] <- rep(idxs, times = m)
    xx[rng] <- as.numeric(outer_xx)
    pos     <- pos + new_n
  }

  if (pos < length(ii)) {
    ii <- ii[seq_len(pos)]; jj <- jj[seq_len(pos)]; xx <- xx[seq_len(pos)]
  }

  Q <- Matrix::sparseMatrix(
    i = ii, j = jj, x = xx, dims = c(n_spatial, n_spatial),
    use.last.ij = FALSE  # sum duplicates
  )
  Matrix::forceSymmetric(Q)
}


# --- Schur complement for NNGP marginal H_beta ----------------------------

#' Build a length-n_obs → length-n_unique indicator design matrix Z.
#'
#' NNGP latent w lives on unique coordinates; each observation maps to one
#' unique-coord slot via `spatial_idx`. So Z[i, j] = 1 iff observation i
#' belongs to unique location j.
#'
#' @keywords internal
.nngp_design_Z <- function(spatial, n_obs) {
  obs_to_unique <- spatial$spatial_idx %||% seq_len(n_obs)
  Matrix::sparseMatrix(
    i = seq_len(n_obs),
    j = as.integer(obs_to_unique),
    x = 1.0,
    dims = c(n_obs, spatial$n_spatial %||% nrow(spatial$unique_coords))
  )
}


#' @keywords internal
.marginal_H_beta_gp <- function(mode, X, spatial, family, phi,
                                n_trials, weights = NULL,
                                sigma2_gp, phi_gp) {
  p         <- ncol(X)
  n_obs     <- nrow(X)
  n_spatial <- spatial$n_spatial %||% nrow(spatial$unique_coords)

  beta <- mode[seq_len(p)]
  w    <- mode[p + seq_len(n_spatial)]

  Z <- .nngp_design_Z(spatial, n_obs)

  eta <- as.numeric(X %*% beta) + as.numeric(Z %*% w)
  W   <- glmm_weights(eta, family, n_trials, phi)
  if (!is.null(weights)) W <- W * weights

  Q_u <- .nngp_precision_Q(spatial, sigma2_gp, phi_gp)

  WX   <- W * X
  WZ   <- as(W * Z, "CsparseMatrix")
  XtWX <- crossprod(X, WX)
  XtWZ <- as.matrix(Matrix::crossprod(X, WZ))
  ZtWZ <- Matrix::crossprod(Z, WZ)

  P_uu <- Matrix::forceSymmetric(ZtWZ + Q_u)

  R   <- Matrix::Cholesky(P_uu, LDL = FALSE, perm = TRUE)
  sol <- Matrix::solve(R, t(XtWZ), system = "A")
  H   <- as.matrix(XtWX) - XtWZ %*% as.matrix(sol)
  H   <- (H + t(H)) / 2

  H
}


#' @keywords internal
.marginal_H_beta_spde <- function(mode, X, spatial, family, phi,
                                  n_trials, weights = NULL,
                                  range_val, sigma_val) {
  p      <- ncol(X)
  n_mesh <- spatial$n_mesh
  beta   <- mode[seq_len(p)]
  w      <- mode[p + seq_len(n_mesh)]

  A <- as(spatial$A, "CsparseMatrix")

  # eta at the mode: X β + A w
  eta <- as.numeric(X %*% beta) + as.numeric(A %*% w)
  W   <- glmm_weights(eta, family, n_trials, phi)
  if (!is.null(weights)) W <- W * weights

  # Q_spde at the (range, sigma) used in the fit
  kappa    <- sqrt(8 * spatial$nu) / range_val
  tau_spde <- 1 / (sqrt(4 * pi) * kappa * sigma_val)
  Q_u      <- .spde_precision_Q(spatial, kappa, tau_spde)

  # Joint Hessian blocks at the mode
  WX   <- W * X
  WA   <- as(W * A, "CsparseMatrix")
  XtWX <- crossprod(X, WX)
  XtWA <- as.matrix(Matrix::crossprod(X, WA))   # p x n_mesh
  AtWA <- Matrix::crossprod(A, WA)              # n_mesh x n_mesh

  P_uu <- Matrix::forceSymmetric(AtWA + Q_u)

  # Schur: H_β^marg = X'WX - (X'WA) P_uu^{-1} (X'WA)'
  R   <- Matrix::Cholesky(P_uu, LDL = FALSE, perm = TRUE)
  sol <- Matrix::solve(R, t(XtWA), system = "A")   # n_mesh x p
  H   <- as.matrix(XtWX) - XtWA %*% as.matrix(sol)
  H   <- (H + t(H)) / 2

  H
}
