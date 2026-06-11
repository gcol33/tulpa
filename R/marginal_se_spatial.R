#' Marginal fixed-effect Hessian for spatial-field Laplace fits
#'
#' Closes the SE gap surfaced in: the single-arm Laplace path
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
    # Fractional alpha: the operator-based rational field (gcol33/tulpa#71) does
    # NOT have a shifted-alpha=2 precision; its precision is Q = Pl' C^{-1} Pl
    # from the rational roots (see .spde_rational_assemble). This analytic
    # marginal-SE path covers only the integer construction; fractional-nu field
    # SEs go through the assembly-based fit, not this rebuild.
    stop("Analytic SPDE marginal SEs are integer-nu only (gcol33/tulpa#71; #85). ",
         "For fractional nu, the rational field SEs are not yet wired through ",
         "this path.", call. = FALSE)
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

#' Build NNGP precision Lambda = (I - A)' D^-1 (I - A) at given hyperparameters.
#'
#' Mirrors the algorithm in `src/gpu_nngp_laplace.h::batch_nngp_scatter` plus
#' `apply_nngp_full_prior_dense` — for each Vecchia row i (in NNGP order):
#'   - resolve conditioning set N(i) via `nn_idx` / `nn_order`
#'   - form the n_nb x n_nb cov matrix C and the n_nb cov vector c
#'   - solve C alpha = c
#'   - v_i = sigma² - c' alpha
#'   - beta_i is 1 at i and -alpha_k at each neighbor -> Lambda += beta_i beta_i' / v_i
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

#' Build a length-n_obs -> length-n_unique indicator design matrix Z.
#'
#' NNGP latent w lives on unique coordinates; each observation maps to one
#' unique-coord slot via `spatial_idx`. So `Z[i, j]` = 1 iff observation i
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


#' n_obs x n_re_groups indicator design for an iid RE block.
#'
#' `re_idx` is the 1-based per-observation group index. Observations whose
#' group falls outside `1:n_re_groups` contribute no RE column, matching the
#' `g >= 0 && g < n_re_groups` guard in the C++ kernels.
#'
#' @keywords internal
.re_design <- function(re_idx, n_re_groups, n_obs) {
  if (n_re_groups <= 0L) {
    return(Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
                                dims = c(n_obs, 0L)))
  }
  g  <- as.integer(re_idx)
  ok <- which(g >= 1L & g <= n_re_groups)
  Matrix::sparseMatrix(i = ok, j = g[ok], x = 1.0,
                       dims = c(n_obs, as.integer(n_re_groups)))
}


#' Schur complement of the latent block out of the joint Hessian.
#'
#' Given the fixed-effect design `X`, the combined latent design `D` (the iid
#' RE indicator block stacked with the spatial-field design), the latent prior
#' precision `Q_latent`, and the GLM weights `W`, returns the marginal fixed-
#' effect precision `H_beta = X'WX - (X'WD) (D'WD + Q_latent)^{-1} (X'WD)'`.
#' Shared by the SPDE and NNGP marginal-SE paths.
#'
#' @keywords internal
.schur_H_beta <- function(X, D, Q_latent, W) {
  D    <- as(D, "CsparseMatrix")
  WX   <- W * X
  WD   <- as(W * D, "CsparseMatrix")
  XtWX <- crossprod(X, WX)
  XtWD <- as.matrix(Matrix::crossprod(X, WD))
  DtWD <- Matrix::crossprod(D, WD)

  P_uu <- Matrix::forceSymmetric(DtWD + Q_latent)
  R    <- Matrix::Cholesky(P_uu, LDL = FALSE, perm = TRUE)
  sol  <- Matrix::solve(R, t(XtWD), system = "A")
  H    <- as.matrix(XtWX) - XtWD %*% as.matrix(sol)
  (H + t(H)) / 2
}


# Both marginal functions take the optional iid RE block so an SPDE / NNGP fit
# carrying one slices its [beta, re, field] mode correctly and Schurs the RE
# columns out of the joint Hessian alongside the field (issue #74). With
# n_re_groups = 0 the RE design is empty and the result is the field-only Schur.

#' @keywords internal
.marginal_H_beta_gp <- function(mode, X, spatial, family, phi,
                                n_trials, weights = NULL,
                                sigma2_gp, phi_gp,
                                re_idx = NULL, n_re_groups = 0L,
                                sigma_re = 1.0) {
  p         <- ncol(X)
  n_obs     <- nrow(X)
  n_spatial <- spatial$n_spatial %||% nrow(spatial$unique_coords)

  beta <- mode[seq_len(p)]
  u    <- mode[p + seq_len(n_re_groups + n_spatial)]

  D_re    <- .re_design(re_idx, n_re_groups, n_obs)
  Z_field <- .nngp_design_Z(spatial, n_obs)
  D       <- cbind(D_re, Z_field)

  eta <- as.numeric(X %*% beta) + as.numeric(D %*% u)
  W   <- glmm_weights(eta, family, n_trials, phi)
  if (!is.null(weights)) W <- W * weights

  tau_re   <- 1 / (sigma_re^2 + 1e-10)
  Q_latent <- Matrix::bdiag(
    Matrix::Diagonal(n_re_groups, x = tau_re),
    .nngp_precision_Q(spatial, sigma2_gp, phi_gp)
  )

  .schur_H_beta(X, D, Q_latent, W)
}


#' @keywords internal
.marginal_H_beta_spde <- function(mode, X, spatial, family, phi,
                                  n_trials, weights = NULL,
                                  range_val, sigma_val,
                                  re_idx = NULL, n_re_groups = 0L,
                                  sigma_re = 1.0) {
  p      <- ncol(X)
  n_obs  <- nrow(X)
  n_mesh <- spatial$n_mesh
  beta   <- mode[seq_len(p)]
  tau_re <- 1 / (sigma_re^2 + 1e-10)

  # Fractional nu: the operator-based rational field (gcol33/tulpa#71; #85). The
  # latent is the auxiliary weight vector x ~ N(0, Q_rat^{-1}) on the non-orphan
  # submesh, with obs -> latent map A_eff = A Pr; the field is u = Pr x. The
  # marginal-beta Schur is parameterisation-invariant, so it runs on (A_eff,
  # Q_rat) exactly as the integer path runs on (A, Q_spde). eta at the mode reads
  # off the field-space mesh block (A w = A_eff x), so W needs no x.
  if (.spde_nu_is_fractional(spatial$nu)) {
    order <- spatial$rational_order %||% 2L
    asm   <- .spde_assemble_at(spatial, range_val, sigma_val, order = order)
    A_full     <- as(spatial$A, "CsparseMatrix")
    D_re       <- .re_design(re_idx, n_re_groups, n_obs)
    u_re       <- if (n_re_groups > 0L) mode[p + seq_len(n_re_groups)] else numeric(0)
    field_full <- mode[p + n_re_groups + seq_len(n_mesh)]
    eta <- as.numeric(X %*% beta) +
           (if (n_re_groups > 0L) as.numeric(D_re %*% u_re) else 0) +
           as.numeric(A_full %*% field_full)
    W <- glmm_weights(eta, family, n_trials, phi)
    if (!is.null(weights)) W <- W * weights
    Q_latent <- Matrix::bdiag(Matrix::Diagonal(n_re_groups, x = tau_re), asm$Q)
    D        <- cbind(D_re, asm$A_eff)
    return(.schur_H_beta(X, D, Q_latent, W))
  }

  u      <- mode[p + seq_len(n_re_groups + n_mesh)]

  A       <- as(spatial$A, "CsparseMatrix")
  D_re    <- .re_design(re_idx, n_re_groups, n_obs)
  D       <- cbind(D_re, A)

  # eta at the mode: X β + D_re u_re + A w
  eta <- as.numeric(X %*% beta) + as.numeric(D %*% u)
  W   <- glmm_weights(eta, family, n_trials, phi)
  if (!is.null(weights)) W <- W * weights

  # Q_spde at the (range, sigma) used in the fit
  kappa    <- sqrt(8 * spatial$nu) / range_val
  tau_spde <- 1 / (sqrt(4 * pi) * kappa * sigma_val)
  Q_latent <- Matrix::bdiag(
    Matrix::Diagonal(n_re_groups, x = tau_re),
    .spde_precision_Q(spatial, kappa, tau_spde)
  )

  .schur_H_beta(X, D, Q_latent, W)
}


# --- Proper-CAR precision + marginal H_beta -------------------------------

#' Proper-CAR precision Q = tau * (D - rho W) at fixed (tau, rho).
#'
#' D is the diagonal degree matrix (neighbour counts) and W the adjacency, so
#' Q has the same nonzero structure as the ICAR precision with rho scaling the
#' off-diagonals. Matches `tulpa::add_car_proper_prior` / the nested CAR_proper
#' precision builder.
#'
#' @keywords internal
.car_proper_precision_Q <- function(spatial, tau, rho) {
  W   <- as(Matrix::Matrix(as.matrix(spatial$adjacency), sparse = TRUE),
            "CsparseMatrix")
  deg <- Matrix::rowSums(W)
  Matrix::forceSymmetric(tau * (Matrix::Diagonal(nrow(W), deg) - rho * W))
}

#' @keywords internal
.marginal_H_beta_car_proper <- function(mode, X, spatial, family, phi,
                                        n_trials, weights = NULL,
                                        tau, rho,
                                        re_idx = NULL, n_re_groups = 0L,
                                        sigma_re = 1.0) {
  p       <- ncol(X)
  n_obs   <- nrow(X)
  n_units <- nrow(as.matrix(spatial$adjacency))

  beta <- mode[seq_len(p)]
  u    <- mode[p + seq_len(n_re_groups + n_units)]

  D_re    <- .re_design(re_idx, n_re_groups, n_obs)
  Z_field <- .nngp_design_Z(spatial, n_obs)   # obs -> areal unit indicator
  D       <- cbind(D_re, Z_field)

  eta <- as.numeric(X %*% beta) + as.numeric(D %*% u)
  W   <- glmm_weights(eta, family, n_trials, phi)
  if (!is.null(weights)) W <- W * weights

  tau_re   <- 1 / (sigma_re^2 + 1e-10)
  Q_latent <- Matrix::bdiag(
    Matrix::Diagonal(n_re_groups, x = tau_re),
    .car_proper_precision_Q(spatial, tau, rho)
  )

  .schur_H_beta(X, D, Q_latent, W)
}


# --- HSGP marginal H_beta -------------------------------------------------

#' @keywords internal
.marginal_H_beta_hsgp <- function(mode, X, spatial, family, phi,
                                  n_trials, weights = NULL,
                                  phi_basis, lambda_eig, sigma2, lengthscale,
                                  re_idx = NULL, n_re_groups = 0L,
                                  sigma_re = 1.0) {
  p     <- ncol(X)
  n_obs <- nrow(X)
  M     <- ncol(phi_basis)

  beta <- mode[seq_len(p)]
  u    <- mode[p + seq_len(n_re_groups + M)]

  # The latent coefficients carry an N(0, I) prior; the spectral density
  # sqrt(S_j) is folded into the design (matching make_hsgp_block.basis_eval /
  # .prep: S_j = sigma2 * sqrt(2 pi) * ell * exp(-0.5 ell^2 lambda_j)).
  S    <- sigma2 * sqrt(2 * pi) * lengthscale *
          exp(-0.5 * lengthscale^2 * lambda_eig)
  PhiS <- sweep(phi_basis, 2, sqrt(pmax(S, 0)), `*`)

  D_re <- .re_design(re_idx, n_re_groups, n_obs)
  D    <- cbind(D_re, Matrix::Matrix(PhiS, sparse = TRUE))

  eta <- as.numeric(X %*% beta) + as.numeric(D %*% u)
  W   <- glmm_weights(eta, family, n_trials, phi)
  if (!is.null(weights)) W <- W * weights

  tau_re   <- 1 / (sigma_re^2 + 1e-10)
  Q_latent <- Matrix::bdiag(
    Matrix::Diagonal(n_re_groups, x = tau_re),
    Matrix::Diagonal(M, x = 1.0)
  )

  .schur_H_beta(X, D, Q_latent, W)
}
