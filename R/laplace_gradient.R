# ============================================================================
# Exact gradient of the joint-field Laplace log-marginal with respect to the
# random-effect covariance coordinates.
#
# The outer objective the EB and nested paths optimize is
#
#     m(theta) = l(x_hat) - 0.5 (x_hat - mu)' P(theta) (x_hat - mu)
#                + 0.5 log|P| - 0.5 log|H|,     H = A' W A + P
#
# with the mode x_hat(theta) and the weights W both moving with theta. The
# stationarity of the inner solve kills the mode's contribution everywhere
# except log|H|, which depends on x_hat through W. Collecting the surviving
# terms (dev_notes/laplace_exact_gradient.md) gives, per RE block,
#
#     dm/dtheta_j = 0.5 tr( dSigma/dtheta_j .
#                           [ Omega (R + V - C) Omega - G Omega ] )
#
# the Fisher identity evaluated at the second moment R + V - C:
#
#     R = sum_g b_g b_g'          mode outer products
#     V = sum_g [H^-1]_gg         posterior covariance of the same coordinates
#     C = sym( sum_g b_g u_g' )   the term that carries dW/dtheta,
#                                 u = H^-1 A' (dw/deta * s),
#                                 s_i = (A H^-1 A')_ii
#
# R + V is the Laplace posterior second moment. -C is the piece the Fisher
# identity alone omits, and it is not small: dropping it leaves the gradient
# wrong by 9 to 58 percent on the cases in dev_notes/proto_exact_gradient.R.
# It vanishes exactly when the curvature is constant in eta (gaussian, and any
# family whose working weight does not move with the linear predictor), which
# is why a Gaussian response never needed it.
#
# `dw/deta` is the eta-derivative of the weight the Newton system ACTUALLY uses
# (laplace_family_curvature.h), not the third derivative of the log density --
# for the families whose H carries an expected/working weight those differ, and
# it is the former that makes this the gradient of the objective being
# optimized.
# ============================================================================


# Symmetric joint posterior precision from the lower-triangle CSC the kernel
# emits under `return_joint_hessian`. Returns a dsCMatrix and drops the raw
# triplet so callers see one object rather than four parallel vectors.
.laplace_joint_hessian <- function(result) {
  need <- c("H_joint_p", "H_joint_i", "H_joint_x", "H_joint_n")
  if (!all(need %in% names(result))) return(NULL)
  n <- as.integer(result$H_joint_n)
  if (length(n) != 1L || is.na(n) || n < 1L) return(NULL)
  # stype = -1: only the lower triangle is stored, so declare the symmetry
  # rather than materializing the mirror image.
  Matrix::sparseMatrix(
    i = as.integer(result$H_joint_i) + 1L,
    p = as.integer(result$H_joint_p),
    x = as.numeric(result$H_joint_x),
    dims = c(n, n),
    symmetric = TRUE,
    index1 = TRUE
  )
}


# Combined random-effect design in the latent column order the kernel uses:
# terms in `re_list` order, and within a term [g1_c1, g1_c2, g2_c1, ...].
# Extracted from the marginal-H_beta Schur block in tulpa_laplace(), which
# needs the same matrix -- one construction, so the two cannot drift.
.re_design_matrix <- function(re_list, n_obs) {
  if (length(re_list) == 0L) return(NULL)
  parts <- lapply(re_list, function(r) {
    nc <- r$n_coefs %||% 1L
    if (nc == 1L && is.null(r$Z)) {
      Matrix::sparseMatrix(
        i = seq_len(n_obs), j = r$idx,
        x = rep(1.0, n_obs), dims = c(n_obs, r$n_groups)
      )
    } else {
      Z_full <- r$Z %||% matrix(1, nrow = n_obs, ncol = 1)
      ii <- rep(seq_len(n_obs), each = nc)
      jj <- rep((r$idx - 1L) * nc, each = nc) + rep(seq_len(nc), n_obs)
      Matrix::sparseMatrix(
        i = ii, j = jj, x = as.numeric(t(Z_full)),
        dims = c(n_obs, r$n_groups * nc)
      )
    }
  })
  do.call(cbind, parts)
}


# dSigma/dtheta_j for one block, in the same coordinate order as
# .re_cov_theta_to_L_list: column-major lower triangle, log L_ii on the
# diagonal and raw L_ij below, or log sigma_i for a diagonal block.
.re_block_dSigma <- function(L, nc, full) {
  emit <- function(dL) dL %*% t(L) + L %*% t(dL)
  if (!isTRUE(full)) {
    return(lapply(seq_len(nc), function(i) {
      dL <- matrix(0, nc, nc); dL[i, i] <- L[i, i]; emit(dL)
    }))
  }
  out <- list()
  for (j in seq_len(nc)) {
    for (i in j:nc) {
      dL <- matrix(0, nc, nc)
      dL[i, j] <- if (i == j) L[i, i] else 1
      out[[length(out) + 1L]] <- emit(dL)
    }
  }
  out
}


# Exact gradient of the Laplace log-marginal w.r.t. the stacked log-Cholesky
# theta, given a fit produced with `return_joint_hessian = TRUE`.
#
# Returns NULL rather than a wrong number whenever an ingredient is missing or
# unusable: no joint Hessian, a singular H, a family without an exact curvature
# derivative, or a layout that does not line up with the mode. The caller then
# falls back to the derivative-free path instead of optimizing a fiction.
.laplace_exact_re_grad <- function(fit, y, X, n_trials, offset, weights,
                                   re_list, layout, L_list, family, phi,
                                   phi2 = NA_real_, want_jacobian = FALSE) {
  H <- fit$H_joint
  x <- fit$mode
  if (is.null(H) || is.null(x)) return(NULL)
  if (!isTRUE(cpp_family_has_curvature_derivative(family))) return(NULL)

  n_obs <- length(y)
  p_fix <- ncol(X)
  n_x   <- length(x)
  Z <- .re_design_matrix(re_list, n_obs)
  if (is.null(Z)) return(NULL)
  A <- cbind(Matrix::Matrix(X, sparse = TRUE), Z)
  if (ncol(A) != n_x) return(NULL)

  eta <- as.numeric(A %*% x)
  if (!is.null(offset)) eta <- eta + as.numeric(offset)

  # H^-1 in full. The alternative is one back-solve per observation for s and
  # per group for V; at n_x below a few thousand the explicit inverse is the
  # same order and serves every downstream piece from one factorization.
  Hinv <- tryCatch(as.matrix(Matrix::solve(H)), error = function(e) NULL)
  if (is.null(Hinv) || any(!is.finite(Hinv))) return(NULL)

  dw <- cpp_family_curvature_deta_vec(as.numeric(y), as.integer(n_trials),
                                      eta, family, phi, phi2)
  if (!is.null(weights)) dw <- dw * as.numeric(weights)
  if (any(!is.finite(dw))) return(NULL)

  # s_i = (A H^-1 A')_ii, the posterior variance of the linear predictor.
  AH <- as.matrix(A %*% Hinv)
  s  <- rowSums(AH * as.matrix(A))

  # One solve carries the whole dW/dtheta channel, independently of how many
  # hyperparameter coordinates there are.
  u <- as.numeric(Hinv %*% as.numeric(Matrix::crossprod(A, dw * s)))

  grad <- numeric(0)
  # Exact mode Jacobian dx_hat/dtheta = -H^-1 (dP/dtheta) x_hat, assembled
  # column by column as the blocks are walked. Each column is a matrix-vector
  # product against the H^-1 already formed, so the whole Jacobian costs no
  # further factorization -- against 2k full inner Laplace re-solves to
  # finite-difference the mode.
  J <- if (isTRUE(want_jacobian)) matrix(0, n_x, 0) else NULL
  pos  <- p_fix
  for (m in seq_along(layout)) {
    bl <- layout[[m]]
    nc <- bl$nc
    G  <- bl$n_groups
    idx <- pos + seq_len(G * nc)
    if (max(idx) > n_x) return(NULL)
    pos <- pos + G * nc

    # Group-major, coef-minor: column g of each is group g's nc-vector.
    b_mat <- matrix(x[idx], nrow = nc)
    u_mat <- matrix(u[idx], nrow = nc)

    Rm <- tcrossprod(b_mat)                       # sum_g b_g b_g'
    Vm <- matrix(0, nc, nc)
    for (g in seq_len(G)) {
      sl <- idx[(g - 1L) * nc + seq_len(nc)]
      Vm <- Vm + Hinv[sl, sl, drop = FALSE]
    }
    Cm <- tcrossprod(b_mat, u_mat)                # sum_g b_g u_g'
    Cm <- (Cm + t(Cm)) / 2

    Lm  <- L_list[[m]]
    Sig <- Lm %*% t(Lm)
    Om  <- tryCatch(chol2inv(chol(Sig)), error = function(e) NULL)
    if (is.null(Om)) return(NULL)

    # -Cm, not +Cm: the term is written in terms of dw/deta (the derivative of
    # the POSITIVE weight H is built from), which is the negative of the third
    # derivative of the log density. See the sign note in
    # dev_notes/laplace_exact_gradient.md -- a gaussian check cannot catch this,
    # because dw/deta is identically zero there.
    Smat <- 0.5 * (Om %*% (Rm + Vm - Cm) %*% Om - G * Om)
    Smat <- (Smat + t(Smat)) / 2
    gm <- tryCatch(
      cpp_recov_block_grad(Smat, Lm, isTRUE(bl$full), 1.0),
      error = function(e) NULL
    )
    if (is.null(gm)) return(NULL)
    grad <- c(grad, gm)

    if (!is.null(J)) {
      dS_list <- .re_block_dSigma(Lm, nc, isTRUE(bl$full))
      if (length(dS_list) != length(gm)) return(NULL)
      Jm <- vapply(dS_list, function(dSig) {
        dOm <- -Om %*% dSig %*% Om
        # (dP/dtheta_j) x_hat is zero outside this block, and inside it is
        # dOmega applied to each group's coefficients.
        rhs <- numeric(n_x)
        rhs[idx] <- as.numeric(dOm %*% b_mat)
        -as.numeric(Hinv %*% rhs)
      }, numeric(n_x))
      J <- cbind(J, Jm)
    }
  }

  if (any(!is.finite(grad))) return(NULL)
  if (is.null(J)) return(grad)
  if (any(!is.finite(J))) return(NULL)
  list(grad = grad, J = J)
}
