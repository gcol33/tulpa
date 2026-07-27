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


# The count predictor's design in the latent column order the kernel builds,
# [X | 0 | Z]. A zero-inflation process appends its coefficients to the fixed
# block -- laplace_core.cpp lays the mode out as [beta | beta_zi | RE] -- but
# they move a SECOND linear predictor, so their columns are zero in the count
# predictor's image. The random effects enter the count predictor only, which is
# what `data.sharing.re[1] = false` sets on the kernel side.
.laplace_count_design <- function(X, X_zi, re_list, n_obs) {
  parts <- list(Matrix::Matrix(X, sparse = TRUE))
  if (!is.null(X_zi)) {
    parts[[length(parts) + 1L]] <-
      Matrix::Matrix(0, nrow = n_obs, ncol = ncol(X_zi), sparse = TRUE)
  }
  Z <- .re_design_matrix(re_list, n_obs)
  if (!is.null(Z)) parts[[length(parts) + 1L]] <- Z
  do.call(cbind, parts)
}


# W_obs - w per observation: what the kernel's working-weight curvature is
# missing to be the observed one. Identically zero for every family whose Newton
# weight already IS the observed curvature, and non-zero only for
# neg_binomial_1 and truncated_neg_binomial_2.
#
# With a zero-inflation process it applies at y != 0 only. The mixture is
# additively separable there, so its count block is the base family's working
# weight (builtin_family_zi.h:136); the y = 0 branch already differentiates the
# density itself through obs_grad_hess_for_family (:168), so correcting it again
# would double-count. Returns NULL rather than a non-finite vector.
.laplace_obs_delta <- function(y, n_trials, eta, family, phi, phi2 = NULL,
                               weights = NULL, has_zi = FALSE) {
  delta <- tryCatch(
    cpp_family_obs_curvature_delta_vec(
      as.numeric(y), as.integer(n_trials), as.numeric(eta),
      family, phi, phi2 %||% NA_real_),
    error = function(e) NULL)
  if (is.null(delta)) return(NULL)
  if (isTRUE(has_zi)) delta[y == 0] <- 0
  if (!is.null(weights)) delta <- delta * as.numeric(weights)
  if (any(!is.finite(delta))) return(NULL)
  delta
}


# d(W_obs - w)/deta, the eta-derivative of the correction above.
#
# The closed Hessian differentiates u, which is formed on the TRUE-curvature
# inverse; that needs dH_true/dtheta = dH/dtheta + A' diag(this * eta_dot) A.
# Masked and weighted exactly like the delta it differentiates, so the pair
# cannot drift. Returns NULL where the derivative is not registered, which sends
# the Hessian back to the stencil rather than pairing two different inverses.
.laplace_obs_delta_deta <- function(y, n_trials, eta, family, phi, phi2 = NULL,
                                    weights = NULL, has_zi = FALSE) {
  if (!isTRUE(tryCatch(cpp_family_has_obs_curvature_delta_derivative(family),
                       error = function(e) FALSE))) return(NULL)
  dd <- tryCatch(
    cpp_family_obs_curvature_delta_deta_vec(
      as.numeric(y), as.integer(n_trials), as.numeric(eta),
      family, phi, phi2 %||% NA_real_),
    error = function(e) NULL)
  if (is.null(dd)) return(NULL)
  if (isTRUE(has_zi)) dd[y == 0] <- 0
  if (!is.null(weights)) dd <- dd * as.numeric(weights)
  if (any(!is.finite(dd))) return(NULL)
  dd
}


# Per-observation dispersion derivatives of the likelihood the objective uses,
# in ONE table whether or not a zero-inflation process is present.
#
# Two disjoint sources, summed over disjoint rows:
#
#  * the base family's registry (.FAMILY_DPHI / .FAMILY_DPHI2) covers every row
#    the model leaves additively separable in the two predictors -- all rows
#    without zero inflation, and the y != 0 rows with it;
#  * the coupled y = 0 branch of a GENUINE mixture comes from the C++ engine
#    (laplace_family_zi_phi.h), which returns zero everywhere the registry
#    already applies, including a hurdle's y = 0 rows (phi-free there).
#
# `zmask` is what keeps the two from overlapping, and it is the only place the
# hurdle / genuine-ZI distinction is expressed: a hurdle's y = 0 rows are zero on
# BOTH sides, which is the statement that its zero branch carries no dispersion.
#
# The column set is the engine's, so a one-process fit simply leaves the six
# z-direction columns at zero and every assembly downstream reads one shape.
# Values are unweighted, as the registry returns them; `weights` are applied at
# the point of use, matching how the curvature channel is handled.
.laplace_phi_fields <- function(cq, y, n_trials, family, phi, phi2,
                                dphi, dphi2 = NULL) {
  eta  <- cq$eta
  cols <- c("dl_dp", "dsc_e_dp", "dsc_z_dp", "dWee_dp", "dWez_dp", "dWzz_dp",
            "dWee_dp_de", "dWee_dp_dz", "dWez_dp_dz", "dWzz_dp_dz",
            "dl_dp2", "dsc_e_dp2", "dsc_z_dp2", "dWee_dp2", "dWez_dp2",
            "dWzz_dp2")
  PD <- matrix(0, length(y), length(cols), dimnames = list(NULL, cols))
  zmask <- if (isTRUE(cq$has_zi)) as.numeric(y != 0) else rep(1, length(y))

  PD[, "dl_dp"]    <- dphi$dloglik(eta, y, n_trials, phi) * zmask
  PD[, "dsc_e_dp"] <- dphi$dscore(eta, y, n_trials, phi) * zmask
  PD[, "dWee_dp"]  <- dphi$dweight(eta, y, n_trials, phi) * zmask
  if (!is.null(dphi2)) {
    PD[, "dl_dp2"]     <- dphi2$dloglik2(eta, y, n_trials, phi) * zmask
    PD[, "dsc_e_dp2"]  <- dphi2$dscore2(eta, y, n_trials, phi) * zmask
    PD[, "dWee_dp2"]   <- dphi2$dweight2(eta, y, n_trials, phi) * zmask
    PD[, "dWee_dp_de"] <- dphi2$dweight_deta(eta, y, n_trials, phi) * zmask
  }

  if (isTRUE(cq$has_zi)) {
    ZP <- tryCatch(
      cpp_zi_mixture_phi_deriv(as.numeric(y), as.integer(n_trials), eta,
                               cq$z_lin, family, phi, phi2 %||% NA_real_),
      error = function(e) NULL)
    if (is.null(ZP)) return(NULL)
    PD <- PD + ZP[, cols, drop = FALSE]
  }
  if (any(!is.finite(PD))) return(NULL)
  PD
}


# Marginal fixed-effect precision from the kernel's joint curvature.
#
# The joint Hessian is the negative-log-POSTERIOR curvature over the whole
# latent vector -- priors and the random-effect penalty included -- so the
# marginal precision on the fixed block is its Schur complement over the random
# effects, with no separate penalty to add back. Two things make this the only
# route rather than one of two: it is exact where a hand-assembled X'WX built
# from the Fisher weight is not (1.8% on neg_binomial_2), and it carries the
# zero-inflation process, which a single-predictor assembly cannot express.
#
# `p_fixed` counts BOTH fixed blocks, so the returned precision is
# (p + p_zi) square and lines up with the [beta | beta_zi] prefix that coef(),
# vcov() and confint() slice.
.laplace_marginal_H_fixed <- function(H_joint, mode, y, n_trials, X, X_zi,
                                      re_list, family, phi, phi2 = NULL,
                                      weights = NULL, offset = NULL) {
  if (is.null(H_joint) || is.null(mode)) return(NULL)
  n_obs   <- length(y)
  p_fixed <- ncol(X) + (if (is.null(X_zi)) 0L else ncol(X_zi))
  n_x     <- nrow(H_joint)
  if (length(mode) != n_x || p_fixed > n_x) return(NULL)

  # The count design has to span the whole latent vector, or the observed-weight
  # correction below cannot be formed. Refused rather than skipped: for the two
  # families whose working weight is not the observed curvature, skipping it
  # silently returns a precision that is wrong by a percent.
  H <- H_joint
  A <- .laplace_count_design(X, X_zi, re_list, n_obs)
  if (ncol(A) != n_x) return(NULL)
  eta <- as.numeric(A %*% mode) + (offset %||% 0)
  delta <- .laplace_obs_delta(y, n_trials, eta, family, phi, phi2,
                              weights, has_zi = !is.null(X_zi))
  if (is.null(delta)) return(NULL)
  if (max(abs(delta)) > 1e-12) {
    H <- H + Matrix::crossprod(A, Matrix::Diagonal(x = delta) %*% A)
  }

  idx <- seq_len(p_fixed)
  if (p_fixed == n_x) return(as.matrix(H))
  B <- H[idx, -idx, drop = FALSE]
  D <- H[-idx, -idx, drop = FALSE]
  mid <- tryCatch(Matrix::solve(D, Matrix::t(B)), error = function(e) NULL)
  if (is.null(mid)) return(NULL)
  out <- as.matrix(H[idx, idx, drop = FALSE] - B %*% mid)
  if (any(!is.finite(out))) return(NULL)
  (out + t(out)) / 2
}


# Whether the exact outer gradient covers a zero-inflation process for this
# family. The gradient differentiates log|H| through dW/dtheta, so with two
# linear predictors it needs the eta-derivative of the MIXTURE's count-block
# curvature, not the base family's. The gate is separate from
# has_curvature_derivative() because a family can have the single-process
# derivative and not the mixture one.
.laplace_exact_supports_zi <- function(family) {
  isTRUE(tryCatch(cpp_family_has_zi_curvature_derivative(family),
                  error = function(e) FALSE))
}


# First derivatives dL/dtheta_p for one block, in the coordinate order of
# .re_cov_theta_to_L_list: for a diagonal block, log sigma_i (i = 1..nc); for a
# full block, the column-major lower triangle with log L_ii on the diagonal and
# raw L_ij below. Because the diagonal coordinates are log-parameterized,
# dL_ii/dtheta = L_ii and d2L_ii/dtheta^2 = L_ii, both flagged here so dSigma and
# d2Sigma share one definition of which coordinate is a log.
.re_block_L_derivs <- function(L, nc, full) {
  Lp <- list(); is_diag <- logical(0); di <- integer(0)
  if (!isTRUE(full)) {
    for (i in seq_len(nc)) {
      dL <- matrix(0, nc, nc); dL[i, i] <- L[i, i]
      Lp[[length(Lp) + 1L]] <- dL
      is_diag <- c(is_diag, TRUE); di <- c(di, i)
    }
  } else {
    for (j in seq_len(nc)) for (i in j:nc) {
      dL <- matrix(0, nc, nc)
      if (i == j) { dL[i, i] <- L[i, i]; is_diag <- c(is_diag, TRUE);  di <- c(di, i) }
      else        { dL[i, j] <- 1;       is_diag <- c(is_diag, FALSE); di <- c(di, NA_integer_) }
      Lp[[length(Lp) + 1L]] <- dL
    }
  }
  list(Lp = Lp, is_diag = is_diag, di = di)
}


# dSigma/dtheta_j for one block: Sigma = L L', so dSigma = dL L' + L dL'.
.re_block_dSigma <- function(L, nc, full) {
  Lp <- .re_block_L_derivs(L, nc, full)$Lp
  lapply(Lp, function(dL) dL %*% t(L) + L %*% t(dL))
}


# d2Sigma/dtheta_p dtheta_q for one block, returned as a list-of-lists indexed
# out[[p]][[q]]. From Sigma = L L',
#
#   d2Sigma = L_pq L' + L_p L_q' + L_q L_p' + L L_pq'
#
# and L_pq is zero except when p == q is a diagonal (log) coordinate, where
# d2L_ii/dtheta^2 = L_ii. Term A of the exact Hessian
# (dev_notes/laplace_exact_hessian.md) contracts this against the Sigma-gradient
# the gradient already forms.
.re_block_d2Sigma <- function(L, nc, full) {
  d  <- .re_block_L_derivs(L, nc, full)
  Lp <- d$Lp; k <- length(Lp)
  out <- lapply(seq_len(k), function(.) vector("list", k))
  for (p in seq_len(k)) for (q in seq_len(k)) {
    d2 <- Lp[[p]] %*% t(Lp[[q]]) + Lp[[q]] %*% t(Lp[[p]])
    if (p == q && isTRUE(d$is_diag[p])) {
      i <- d$di[p]
      Lpp <- matrix(0, nc, nc); Lpp[i, i] <- L[i, i]
      d2 <- d2 + Lpp %*% t(L) + L %*% t(Lpp)
    }
    out[[p]][[q]] <- d2
  }
  out
}


# Shared intermediates for the exact gradient AND the exact Hessian: the mode,
# H^-1, the linear-predictor variance s, the dW/dtheta solve u, and per RE block
# the moments (R, V, C), the covariance/precision (Sigma via L, Omega) and the
# Sigma-gradient S. Both derivatives are contractions against these, so forming
# them once is the single source of truth the two assemblies read.
#
# Returns NULL rather than a wrong number whenever an ingredient is missing or
# unusable: no joint Hessian, a singular H or Sigma, a family without an exact
# curvature derivative, or a layout that does not line up with the mode.
.laplace_exact_core <- function(fit, y, X, n_trials, offset, weights,
                                re_list, layout, L_list, family, phi, phi2,
                                X_zi = NULL) {
  H <- fit$H_joint
  x <- fit$mode
  if (is.null(H) || is.null(x)) return(NULL)
  if (!isTRUE(cpp_family_has_curvature_derivative(family))) return(NULL)
  has_zi <- !is.null(X_zi)
  if (has_zi && !.laplace_exact_supports_zi(family)) return(NULL)

  n_obs   <- length(y)
  p_fix   <- ncol(X)
  p_zi    <- if (has_zi) ncol(X_zi) else 0L
  p_fixed <- p_fix + p_zi
  n_x     <- length(x)
  Z <- .re_design_matrix(re_list, n_obs)
  if (is.null(Z)) return(NULL)
  # A is the COUNT predictor's design. With zero inflation the latent vector
  # also carries beta_zi, whose columns are zero here and carry the second
  # predictor in A_zi instead.
  A <- .laplace_count_design(X, X_zi, re_list, n_obs)
  if (ncol(A) != n_x) return(NULL)
  A_zi <- if (!has_zi) NULL else cbind(
    Matrix::Matrix(0, nrow = n_obs, ncol = p_fix, sparse = TRUE),
    Matrix::Matrix(X_zi, sparse = TRUE),
    Matrix::Matrix(0, nrow = n_obs, ncol = n_x - p_fixed, sparse = TRUE))

  eta <- as.numeric(A %*% x)
  if (!is.null(offset)) eta <- eta + as.numeric(offset)
  z_lin <- if (has_zi) as.numeric(A_zi %*% x) else NULL

  # H^-1 in full. The alternative is one back-solve per observation for s and
  # per group for V; at n_x below a few thousand the explicit inverse is the
  # same order and serves every downstream piece from one factorization.
  Hinv <- tryCatch(as.matrix(Matrix::solve(H)), error = function(e) NULL)
  if (is.null(Hinv) || any(!is.finite(Hinv))) return(NULL)

  # Mode-Jacobian inverse. dx_hat/dtheta = -(A' diag(W_obs) A + P)^-1 (dP) x_hat
  # is governed by the OBSERVED curvature W_obs, which the working-weight H_joint
  # equals only for canonical / constant-curvature families. Where they differ
  # (neg_binomial_1, truncated_neg_binomial_2) the true Hessian is H_joint plus
  # A' diag(W_obs - w) A; where no exact observed form exists the Jacobian cannot
  # be formed exactly and the caller differences the mode instead. Everything
  # else in the gradient/Hessian reads H_joint, since the objective's log|H| is
  # the working-weight one -- only the mode motion uses this inverse.
  # The correction rides the COUNT predictor: it is the base family's weight
  # that the mixture's y != 0 branch carries, and the y = 0 branch already uses
  # the observed form (.laplace_obs_delta zeroes it there).
  exact_jac <- isTRUE(cpp_family_has_exact_mode_jacobian(family))
  Hinv_mode <- Hinv
  # Whether the Newton working weight already IS the observed curvature. False
  # only for neg_binomial_1 and truncated_neg_binomial_2, and read off the
  # delta rather than a second hand-kept family list.
  working_is_observed <- TRUE
  if (exact_jac) {
    delta <- .laplace_obs_delta(y, n_trials, eta, family, phi, phi2, weights,
                                has_zi = has_zi)
    if (is.null(delta)) return(NULL)
    if (max(abs(delta)) > 1e-12) {
      working_is_observed <- FALSE
      H_true <- H + Matrix::crossprod(A, Matrix::Diagonal(x = delta) %*% A)
      Hinv_mode <- tryCatch(as.matrix(Matrix::solve(H_true)), error = function(e) NULL)
      if (is.null(Hinv_mode) || any(!is.finite(Hinv_mode))) {
        exact_jac <- FALSE; Hinv_mode <- Hinv
      }
    }
  }

  # s_i = (A H^-1 A')_ii, the posterior variance of the linear predictor. With
  # two predictors the dW channel needs the full 2 x 2 predictor covariance --
  # the variance of z and their covariance as well.
  AH <- as.matrix(A %*% Hinv)
  s  <- rowSums(AH * as.matrix(A))
  s_zz <- if (!has_zi) NULL else
    rowSums(as.matrix(A_zi %*% Hinv) * as.matrix(A_zi))
  s_ez <- if (!has_zi) NULL else rowSums(AH * as.matrix(A_zi))

  # The dW/dtheta channel. tr(H^-1 dH_W) contracts each curvature partial
  # against the matching predictor (co)variance, and what multiplies the mode
  # motion is
  #
  #   q_eta = dW_ee/deta s_ee + 2 dW_ez/deta s_ez + dW_zz/deta s_zz
  #   q_z   = dW_ee/dz   s_ee + 2 dW_ez/dz   s_ez + dW_zz/dz   s_zz
  #
  # so v_r = A' q_eta + A_zi' q_z. Without zero inflation W is the scalar
  # count-block weight, q_eta collapses to dw * s and the second term is absent
  # -- the one-process formula is this one with an empty A_zi.
  wt_vec <- if (is.null(weights)) NULL else as.numeric(weights)
  dW <- NULL
  if (!has_zi) {
    dw <- cpp_family_curvature_deta_vec(as.numeric(y), as.integer(n_trials),
                                        eta, family, phi, phi2)
    if (!is.null(wt_vec)) dw <- dw * wt_vec
    if (any(!is.finite(dw))) return(NULL)
    q_eta <- dw * s
    q_z   <- NULL
  } else {
    dW <- tryCatch(
      cpp_zi_mixture_curvature_deriv(as.numeric(y), as.integer(n_trials),
                                     eta, z_lin, family, phi, phi2),
      error = function(e) NULL)
    if (is.null(dW) || any(!is.finite(dW))) return(NULL)
    if (!is.null(wt_vec)) dW <- dW * wt_vec
    # Kept under the one-process name so the Hessian's du/dtheta and the phi
    # block read the count-block derivative by the same handle either way.
    dw    <- dW[, "dWee_deta"]
    q_eta <- dw * s + 2 * dW[, "dWez_deta"] * s_ez + dW[, "dWzz_deta"] * s_zz
    q_z   <- dW[, "dWee_dz"] * s + 2 * dW[, "dWez_dz"] * s_ez +
             dW[, "dWzz_dz"] * s_zz
  }

  # One solve carries the whole dW/dtheta channel, independently of how many
  # hyperparameter coordinates there are. v_r is kept for the Hessian's
  # du/dtheta, where it is the vector u differentiates.
  #
  # Hinv_mode, not Hinv. This channel is v_r' (dx_hat/dtheta), and the mode
  # motion follows the TRUE stationarity condition, so it is governed by the
  # observed curvature -- the same inverse the mode Jacobian uses. The two
  # coincide wherever the working weight is the observed one, which is why a
  # poisson or gaussian check cannot see the difference; on
  # truncated_neg_binomial_2 they differ by ~9% and the gradient was off ~0.8%.
  v_r <- as.numeric(Matrix::crossprod(A, q_eta))
  if (has_zi) v_r <- v_r + as.numeric(Matrix::crossprod(A_zi, q_z))
  u   <- as.numeric(Hinv_mode %*% v_r)

  blocks <- vector("list", length(layout))
  pos <- p_fixed
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
    M0   <- Rm + Vm - Cm
    Smat <- 0.5 * (Om %*% M0 %*% Om - G * Om)
    Smat <- (Smat + t(Smat)) / 2

    blocks[[m]] <- list(idx = idx, nc = nc, G = G, full = isTRUE(bl$full),
                        b_mat = b_mat, u_mat = u_mat, Rm = Rm, Vm = Vm, Cm = Cm,
                        Lm = Lm, Om = Om, M0 = M0, Smat = Smat)
  }

  list(A = A, n_x = n_x, p_fix = p_fix, p_fixed = p_fixed, n_obs = n_obs,
       Hinv = Hinv, x = x, eta = eta, dw = dw, s = s, u = u, v_r = v_r,
       blocks = blocks, Hinv_mode = Hinv_mode, exact_jac = exact_jac,
       working_is_observed = working_is_observed,
       # q_eta / q_z are what any mode motion contracts against, whatever moved
       # the mode. The RE coordinates reach them through v_r; the dispersion
       # coordinate reads them directly, so the two-process channel is written
       # once rather than once per coordinate kind. `dW` is the raw six-column
       # block the Hessian differentiates a second time.
       q_eta = q_eta, q_z = q_z, dW = dW,
       has_zi = has_zi, A_zi = A_zi, z_lin = z_lin, s_zz = s_zz, s_ez = s_ez)
}


# Exact Hessian of the Laplace log-marginal w.r.t. the stacked log-Cholesky
# theta. The random-effect block is assembled column by column from the shared
# core: for each coordinate k the mode Jacobian J_k, the weight/precision moves
# dW_k and dP_k, and dH^-1_k feed a per-block transport of the Sigma-gradient
# (term B) plus the parameterization curvature (term A). Every off-diagonal is
# built twice -- as column j and as column k -- and the two agree to the inner
# solver's noise, so the raw matrix is symmetric before it is symmetrized.
#
# With `phi_block` supplied the returned matrix is bordered by the dispersion
# coordinate (dev_notes/laplace_phi_hessian.md): a log-phi column
# H_{theta_j, psi} reusing term B with the phi mode-motion (no dP, since the RE
# prior is phi-free), and the diagonal H_{psi, psi} differentiating the phi
# gradient once more. Without it the matrix is the RE block alone, the shape the
# fixed-dispersion path expects.
#
# With a zero-inflation process the same assembly runs over the full 2 x 2
# curvature block rather than one scalar weight: each coordinate's mode motion
# moves the zero predictor as well as the count predictor, so every entry of the
# block moves through both, dH_k picks up the cross and zero blocks, and du_k a
# matching A_zi' dq_z contribution. A hurdle is the case where the mixed
# fourth-order fields vanish, and the same code covers it -- the terms multiply
# by zero rather than being skipped.
#
# `dS_by_block` is the same per-block dSigma list the gradient's Jacobian uses,
# passed in so the chain rule is formed once. Returns NULL on a non-finite
# second curvature derivative.
.laplace_exact_re_hess <- function(cq, y, n_trials, family, phi, phi2, weights,
                                   dS_by_block, phi_block = NULL) {
  A <- cq$A; Hinv <- cq$Hinv; x <- cq$x; n_x <- cq$n_x
  Hinv_mode <- cq$Hinv_mode
  s <- cq$s; dwdeta <- cq$dw; u <- cq$u; v_r <- cq$v_r
  blocks <- cq$blocks; nb <- length(blocks)

  # The mixture's curvature block is -Hess(log density), so its second
  # derivatives are the fourth derivatives of one scalar: five distinct values
  # covering all nine second partials, indexed by how many derivatives are in
  # eta. A hurdle zeroes the three mixed ones, which is what made the decoupled
  # assembly enough for it; the coupled branch below is the general form and
  # reduces to that one term by term.
  has_zi  <- isTRUE(cq$has_zi)
  A_zi    <- cq$A_zi
  s_zz    <- cq$s_zz
  s_ez    <- cq$s_ez
  dW      <- cq$dW
  if (!has_zi) {
    d2w <- cpp_family_curvature_deta2_vec(as.numeric(y), as.integer(n_trials),
                                          cq$eta, family, phi, phi2)
    d4 <- NULL
  } else {
    d4 <- tryCatch(
      cpp_zi_mixture_curvature_deriv2(as.numeric(y), as.integer(n_trials),
                                      cq$eta, cq$z_lin, family, phi, phi2),
      error = function(e) NULL)
    if (is.null(d4) || any(!is.finite(d4))) return(NULL)
    if (!is.null(weights)) d4 <- d4 * as.numeric(weights)
    d2w <- d4[, "d4_e4"]
  }
  if (!has_zi && !is.null(weights)) d2w <- d2w * as.numeric(weights)
  if (any(!is.finite(d2w))) return(NULL)

  # u is formed on the observed-curvature inverse, so differentiating it needs
  # dH_true/dtheta rather than dH/dtheta. Where the working weight already IS
  # the observed curvature the two coincide and this stays NULL, which is the
  # cheaper path as well as the equivalent one.
  ddelta <- NULL
  if (!isTRUE(cq$working_is_observed)) {
    ddelta <- .laplace_obs_delta_deta(y, n_trials, cq$eta, family, phi, phi2,
                                      weights, has_zi = has_zi)
    if (is.null(ddelta)) return(NULL)
  }

  klens  <- vapply(dS_by_block, length, integer(1))
  ktot   <- sum(klens)
  gstart <- c(0L, cumsum(klens))
  blk_of <- integer(ktot); loc_of <- integer(ktot)
  for (m in seq_len(nb)) {
    rng <- gstart[m] + seq_len(klens[m])
    blk_of[rng] <- m; loc_of[rng] <- seq_len(klens[m])
  }
  d2S_by_block <- lapply(seq_len(nb), function(m)
    .re_block_d2Sigma(blocks[[m]]$Lm, blocks[[m]]$nc, blocks[[m]]$full))

  Hm <- matrix(0, ktot, ktot)
  for (k in seq_len(ktot)) {
    mk <- blk_of[k]; lk <- loc_of[k]
    bk <- blocks[[mk]]
    Om_k   <- bk$Om
    dOm_k  <- -Om_k %*% dS_by_block[[mk]][[lk]] %*% Om_k

    # dP_k: dOmega_k on each group of block mk, zero elsewhere.
    dP_k <- matrix(0, n_x, n_x)
    for (g in seq_len(bk$G)) {
      sl <- bk$idx[(g - 1L) * bk$nc + seq_len(bk$nc)]
      dP_k[sl, sl] <- dOm_k
    }

    J_k     <- -as.numeric(Hinv_mode %*% (dP_k %*% x))  # mode Jacobian column
    eta_dot <- as.numeric(A %*% J_k)
    # The mode motion moves BOTH predictors, so every entry of the curvature
    # block moves through both. Without zero inflation only the first line
    # survives and z_dot does not exist.
    z_dot <- if (!has_zi) NULL else as.numeric(A_zi %*% J_k)
    dW_k  <- if (!has_zi) dwdeta * eta_dot else
      dW[, "dWee_deta"] * eta_dot + dW[, "dWee_dz"] * z_dot
    dH_k  <- as.matrix(Matrix::crossprod(A, Matrix::Diagonal(x = dW_k) %*% A)) + dP_k
    if (has_zi) {
      dWez_k <- dW[, "dWez_deta"] * eta_dot + dW[, "dWez_dz"] * z_dot
      dWzz_k <- dW[, "dWzz_deta"] * eta_dot + dW[, "dWzz_dz"] * z_dot
      cross  <- as.matrix(Matrix::crossprod(
        A, Matrix::Diagonal(x = dWez_k) %*% A_zi))
      dH_k <- dH_k + cross + t(cross) +
        as.matrix(Matrix::crossprod(A_zi, Matrix::Diagonal(x = dWzz_k) %*% A_zi))
    }
    dHinv_k <- -Hinv %*% dH_k %*% Hinv
    AdH     <- as.matrix(A %*% dHinv_k)
    ds_k    <- rowSums(AdH * as.matrix(A))

    # d q_eta / d theta_k and d q_z / d theta_k: the gradient's channel
    # differentiated once more. Each of the three curvature partials in q moves
    # through both predictors, and each of the three predictor (co)variances
    # moves with the mode, so every term is a product rule over one of the five
    # fourth-order fields and one variance derivative.
    if (!has_zi) {
      dq_eta_k <- (d2w * eta_dot) * s + dwdeta * ds_k
      dq_z_k   <- NULL
    } else {
      ds_zz_k <- rowSums(as.matrix(A_zi %*% dHinv_k) * as.matrix(A_zi))
      ds_ez_k <- rowSums(AdH * as.matrix(A_zi))
      dq_eta_k <-
        (d4[, "d4_e4"]   * eta_dot + d4[, "d4_e3z"]  * z_dot) * s +
        dW[, "dWee_deta"] * ds_k +
        2 * ((d4[, "d4_e3z"]  * eta_dot + d4[, "d4_e2z2"] * z_dot) * s_ez +
             dW[, "dWez_deta"] * ds_ez_k) +
        (d4[, "d4_e2z2"] * eta_dot + d4[, "d4_ez3"]  * z_dot) * s_zz +
        dW[, "dWzz_deta"] * ds_zz_k
      dq_z_k <-
        (d4[, "d4_e3z"]  * eta_dot + d4[, "d4_e2z2"] * z_dot) * s +
        dW[, "dWee_dz"] * ds_k +
        2 * ((d4[, "d4_e2z2"] * eta_dot + d4[, "d4_ez3"] * z_dot) * s_ez +
             dW[, "dWez_dz"] * ds_ez_k) +
        (d4[, "d4_ez3"]  * eta_dot + d4[, "d4_z4"]  * z_dot) * s_zz +
        dW[, "dWzz_dz"] * ds_zz_k
    }

    # d(H_true)^-1/dtheta_k. H_true is H plus the observed-minus-working
    # correction, and only the correction's own eta-motion distinguishes its
    # derivative from dHinv_k -- the prior term dP_k is already inside dH_k and
    # carries no weight. Identical to dHinv_k wherever the two weights agree.
    dHinv_mode_k <- if (is.null(ddelta)) dHinv_k else
      -Hinv_mode %*%
        (dH_k + as.matrix(Matrix::crossprod(
           A, Matrix::Diagonal(x = ddelta * eta_dot) %*% A))) %*% Hinv_mode
    # Hinv_mode, matching how u itself is formed. s, V and log|H| above stay on
    # the working-weight inverse, which is the objective's own.
    du_k <- as.numeric(dHinv_mode_k %*% v_r) +
            as.numeric(Hinv_mode %*% as.numeric(Matrix::crossprod(A, dq_eta_k)))
    if (has_zi) {
      du_k <- du_k +
        as.numeric(Hinv_mode %*% as.numeric(Matrix::crossprod(A_zi, dq_z_k)))
    }

    for (m in seq_len(nb)) {
      bq <- blocks[[m]]; nc <- bq$nc; G <- bq$G; idx <- bq$idx
      b_m  <- bq$b_mat
      u_m  <- bq$u_mat
      Jb_m <- matrix(J_k[idx],  nrow = nc)
      du_m <- matrix(du_k[idx], nrow = nc)

      dR_m <- Jb_m %*% t(b_m) + b_m %*% t(Jb_m)
      dV_m <- matrix(0, nc, nc)
      for (g in seq_len(G)) {
        sl <- idx[(g - 1L) * nc + seq_len(nc)]
        dV_m <- dV_m + dHinv_k[sl, sl, drop = FALSE]
      }
      Cp   <- Jb_m %*% t(u_m) + b_m %*% t(du_m); dC_m <- 0.5 * (Cp + t(Cp))
      dM0_m <- dR_m + dV_m - dC_m

      Om_m <- bq$Om; M0_m <- bq$M0
      dS_m <- 0.5 * (Om_m %*% dM0_m %*% Om_m)
      if (m == mk)
        dS_m <- dS_m +
          0.5 * (dOm_k %*% M0_m %*% Om_m + Om_m %*% M0_m %*% dOm_k - G * dOm_k)

      # term B: <dSigma_j, dS_m> for every coordinate j in block m.
      dS_list <- dS_by_block[[m]]
      for (lj in seq_along(dS_list)) {
        gj <- gstart[m] + lj
        Hm[gj, k] <- Hm[gj, k] + sum(dS_list[[lj]] * dS_m)
      }
    }

    # term A: <d2Sigma_{j,k}, S_mk> for coordinate j in block mk (block-local).
    S_mk    <- bk$Smat
    d2S_mk  <- d2S_by_block[[mk]]
    for (lj in seq_len(klens[mk])) {
      gj <- gstart[mk] + lj
      Hm[gj, k] <- Hm[gj, k] + sum(d2S_mk[[lj]][[lk]] * S_mk)
    }
  }

  # ---- dispersion column and diagonal (chi = [theta..., log phi]) -----------
  # Borders the RE block with the log-phi coordinate. dxdphi (the phi mode
  # motion) and the phi gradient value come pre-formed from the caller so the
  # two are one computation; .family_dphi2 is registered only where the working
  # and observed weights coincide, so Hinv_mode is Hinv and dH_true/dpsi is
  # dH_joint/dpsi here -- the coincidence that makes the closed form exact.
  if (!is.null(phi_block) && !is.null(phi_block$dphi2)) {
    wt <- phi_block$wt; PD <- phi_block$PD
    dxdphi <- phi_block$dxdphi; deta_dphi <- phi_block$deta_dphi
    dz_dphi <- phi_block$dz_dphi

    # J_psi = phi dx/dphi is the log-phi mode-Jacobian column; eta_dot and z_dot
    # its two linear-predictor images. d(H_joint)/dpsi moves through the mode in
    # BOTH predictors and explicitly through phi, with no dP term (the RE prior
    # is phi-free). Without a mixture only the first line survives.
    J_psi   <- phi * dxdphi
    eta_dot <- phi * deta_dphi
    z_dot   <- if (!has_zi) NULL else phi * dz_dphi

    dWee_psi <- dwdeta * eta_dot + phi * (wt * PD[, "dWee_dp"])
    if (has_zi) dWee_psi <- dWee_psi + dW[, "dWee_dz"] * z_dot
    dH_psi <- as.matrix(Matrix::crossprod(
      A, Matrix::Diagonal(x = dWee_psi) %*% A))
    if (has_zi) {
      dWez_psi <- dW[, "dWez_deta"] * eta_dot + dW[, "dWez_dz"] * z_dot +
        phi * (wt * PD[, "dWez_dp"])
      dWzz_psi <- dW[, "dWzz_deta"] * eta_dot + dW[, "dWzz_dz"] * z_dot +
        phi * (wt * PD[, "dWzz_dp"])
      cross_psi <- as.matrix(Matrix::crossprod(
        A, Matrix::Diagonal(x = dWez_psi) %*% A_zi))
      dH_psi <- dH_psi + cross_psi + t(cross_psi) +
        as.matrix(Matrix::crossprod(
          A_zi, Matrix::Diagonal(x = dWzz_psi) %*% A_zi))
    }
    dHinv_psi <- -Hinv %*% dH_psi %*% Hinv
    AdH_psi <- as.matrix(A %*% dHinv_psi)
    ds_psi  <- rowSums(AdH_psi * as.matrix(A))
    ds_zz_psi <- if (!has_zi) NULL else
      rowSums(as.matrix(A_zi %*% dHinv_psi) * as.matrix(A_zi))
    ds_ez_psi <- if (!has_zi) NULL else rowSums(AdH_psi * as.matrix(A_zi))

    # Each of the curvature block's own derivatives moves with both predictors
    # and explicitly with phi: the fourth-order fields supply the first two
    # directions, PD the third. The mixed fields are the same fourth derivative
    # reached two ways -- dW_ez/deta is dW_ee/dz, and dW_zz/deta is dW_ez/dz --
    # so four of the six are shared rather than computed twice.
    if (!has_zi) {
      dWee_de_psi <- d2w * eta_dot + phi * (wt * PD[, "dWee_dp_de"])
      dq_eta_psi  <- dWee_de_psi * s + dwdeta * ds_psi
      dq_z_psi    <- NULL
    } else {
      mv <- function(f_e, f_z, pcol)
        d4[, f_e] * eta_dot + d4[, f_z] * z_dot + phi * (wt * PD[, pcol])
      dWee_de_psi <- mv("d4_e4",   "d4_e3z",  "dWee_dp_de")
      dWee_dz_psi <- mv("d4_e3z",  "d4_e2z2", "dWee_dp_dz")
      dWez_dz_psi <- mv("d4_e2z2", "d4_ez3",  "dWez_dp_dz")
      dWzz_dz_psi <- mv("d4_ez3",  "d4_z4",   "dWzz_dp_dz")
      dq_eta_psi <- dWee_de_psi * s + dwdeta * ds_psi +
        2 * (dWee_dz_psi * s_ez + dW[, "dWez_deta"] * ds_ez_psi) +
        dWez_dz_psi * s_zz + dW[, "dWzz_deta"] * ds_zz_psi
      dq_z_psi <- dWee_dz_psi * s + dW[, "dWee_dz"] * ds_psi +
        2 * (dWez_dz_psi * s_ez + dW[, "dWez_dz"] * ds_ez_psi) +
        dWzz_dz_psi * s_zz + dW[, "dWzz_dz"] * ds_zz_psi
    }
    du_psi <- as.numeric(dHinv_psi %*% v_r) +
              as.numeric(Hinv %*% as.numeric(Matrix::crossprod(A, dq_eta_psi)))
    if (has_zi) {
      du_psi <- du_psi +
        as.numeric(Hinv %*% as.numeric(Matrix::crossprod(A_zi, dq_z_psi)))
    }

    # phi column: term B alone (psi does not touch Sigma, so term A and dOm drop).
    phi_col <- numeric(ktot)
    for (m in seq_len(nb)) {
      bq <- blocks[[m]]; nc <- bq$nc; G <- bq$G; idx <- bq$idx
      b_m <- bq$b_mat; u_m <- bq$u_mat
      Jb_m <- matrix(J_psi[idx],  nrow = nc)
      du_m <- matrix(du_psi[idx], nrow = nc)
      dR_m <- Jb_m %*% t(b_m) + b_m %*% t(Jb_m)
      dV_m <- matrix(0, nc, nc)
      for (g in seq_len(G)) {
        sl <- idx[(g - 1L) * nc + seq_len(nc)]
        dV_m <- dV_m + dHinv_psi[sl, sl, drop = FALSE]
      }
      Cp <- Jb_m %*% t(u_m) + b_m %*% t(du_m); dC_m <- 0.5 * (Cp + t(Cp))
      dM0_m <- dR_m + dV_m - dC_m
      dS_m  <- 0.5 * (bq$Om %*% dM0_m %*% bq$Om)
      dS_list <- dS_by_block[[m]]
      for (lj in seq_along(dS_list)) {
        gj <- gstart[m] + lj
        phi_col[gj] <- phi_col[gj] + sum(dS_list[[lj]] * dS_m)
      }
    }

    # phi diagonal: differentiate the phi gradient once more in psi. The second
    # mode derivative d(dx/dphi)/dpsi is the one genuinely new solve. Its right
    # hand side needs d(score)/dpsi in each predictor, and those come from the
    # identity that the score's eta- and z-derivatives are -W: no new function,
    # just the dW/dphi fields already in PD.
    dA <- sum(wt * (PD[, "dsc_e_dp"] * eta_dot + phi * PD[, "dl_dp2"]))
    rhs_e <- (-PD[, "dWee_dp"]) * eta_dot + phi * PD[, "dsc_e_dp2"]
    if (has_zi) rhs_e <- rhs_e - PD[, "dWez_dp"] * z_dot
    d2x_rhs <- as.numeric(Matrix::crossprod(A, wt * rhs_e))
    if (has_zi) {
      dA <- dA + sum(wt * PD[, "dsc_z_dp"] * z_dot)
      rhs_z <- (-PD[, "dWez_dp"]) * eta_dot - PD[, "dWzz_dp"] * z_dot +
        phi * PD[, "dsc_z_dp2"]
      d2x_rhs <- d2x_rhs +
        as.numeric(Matrix::crossprod(A_zi, wt * rhs_z))
    }
    d2x_dpsi <- as.numeric(dHinv_psi %*% phi_block$rhs_mode) +
                as.numeric(Hinv_mode %*% d2x_rhs)
    d_deta_dphi_dpsi <- as.numeric(A %*% d2x_dpsi)

    # dB is the log|H| half of the gradient differentiated once more. Written as
    # the product rule over (predictor covariance) x (explicit dW/dphi) plus the
    # mode-motion channel q . d(eta,z)/dphi, which is the same decomposition the
    # gradient itself uses -- so the one-process case reduces to it term by term.
    dWee_dp_psi <- PD[, "dWee_dp_de"] * eta_dot + phi * PD[, "dWee_dp2"]
    if (has_zi) dWee_dp_psi <- dWee_dp_psi + PD[, "dWee_dp_dz"] * z_dot
    dB <- sum(ds_psi * wt * PD[, "dWee_dp"] + s * wt * dWee_dp_psi +
              dq_eta_psi * deta_dphi + cq$q_eta * d_deta_dphi_dpsi)
    if (has_zi) {
      d_dz_dphi_dpsi <- as.numeric(A_zi %*% d2x_dpsi)
      dWez_dp_psi <- PD[, "dWee_dp_dz"] * eta_dot +
        PD[, "dWez_dp_dz"] * z_dot + phi * PD[, "dWez_dp2"]
      dWzz_dp_psi <- PD[, "dWez_dp_dz"] * eta_dot +
        PD[, "dWzz_dp_dz"] * z_dot + phi * PD[, "dWzz_dp2"]
      dB <- dB +
        sum(2 * (ds_ez_psi * wt * PD[, "dWez_dp"] + s_ez * wt * dWez_dp_psi) +
            ds_zz_psi * wt * PD[, "dWzz_dp"] + s_zz * wt * dWzz_dp_psi +
            dq_z_psi * dz_dphi + cq$q_z * d_dz_dphi_dpsi)
    }
    phi_diag <- phi_block$grad_phi + phi * (dA - 0.5 * dB)
    if (!is.finite(phi_diag) || any(!is.finite(phi_col))) return(NULL)

    # unname: cbind/rbind would otherwise label the border after the local
    # variable (`phi_col`); the marginal correction sets the real coordinate
    # names, and a stray one here diverges from the stencils' plain matrices.
    Hm <- unname(rbind(cbind(Hm, phi_col), c(phi_col, phi_diag)))
  }

  if (any(!is.finite(Hm))) return(NULL)
  (Hm + t(Hm)) / 2
}


# Exact gradient of the Laplace log-marginal w.r.t. the stacked log-Cholesky
# theta, given a fit produced with `return_joint_hessian = TRUE`. On request
# also the exact mode Jacobian J (closed form) and the exact theta-Hessian H.
#
# Returns NULL rather than a wrong number whenever an ingredient is missing or
# unusable; the caller then falls back to the derivative-free path instead of
# optimizing a fiction. With `want_hessian` the returned H is NULL (grad and J
# still valid) for a family that has no closed-form second curvature derivative,
# which sends only the Hessian back to the differencing stencil.
.laplace_exact_re_grad <- function(fit, y, X, n_trials, offset, weights,
                                   re_list, layout, L_list, family, phi,
                                   phi2 = NA_real_, want_jacobian = FALSE,
                                   want_hessian = FALSE, want_phi = FALSE,
                                   X_zi = NULL) {
  cq <- .laplace_exact_core(fit, y, X, n_trials, offset, weights,
                            re_list, layout, L_list, family, phi, phi2,
                            X_zi = X_zi)
  if (is.null(cq)) return(NULL)
  # The dispersion block reads the BASE family's phi derivatives for every row
  # the mixture leaves additively separable, and the coupled y = 0 branch from
  # the mixture engine for the rest (.laplace_phi_fields). Under a hurdle the
  # second source is identically zero -- a zero-truncated base has P(Y = 0) = 0,
  # so its zero branch is log(pi), phi-free -- which is why the base registry
  # alone was already exact there. Under genuine zero inflation the y = 0 branch
  # is log(pi + (1 - pi) P(Y = 0, phi)) and carries phi through P(Y = 0) as well,
  # coupled to both predictors; that is what the engine supplies.
  #
  # Gated on the engine rather than on the base being truncated, so a family
  # without a registered phi column is declined instead of receiving the base
  # derivatives under a model they do not describe.
  if (isTRUE(cq$has_zi) && isTRUE(want_phi) &&
      !isTRUE(tryCatch(cpp_family_has_zi_phi_deriv(family),
                       error = function(e) FALSE))) return(NULL)
  A <- cq$A; Hinv <- cq$Hinv; x <- cq$x; n_x <- cq$n_x
  Hinv_mode <- cq$Hinv_mode
  s <- cq$s; dw <- cq$dw; eta <- cq$eta; n_obs <- cq$n_obs

  want_J <- isTRUE(want_jacobian) || isTRUE(want_hessian)
  # The mode Jacobian, and the closed Hessian that reads it, are exact only when
  # the true observed curvature is available (has_exact_mode_jacobian). Decline
  # both rather than return a working-weight Jacobian; the caller then differences
  # the mode. The gradient value itself is unaffected and still returned below.
  if (want_J && !isTRUE(cq$exact_jac)) return(NULL)

  grad <- numeric(0)
  # Exact mode Jacobian dx_hat/dtheta = -H^-1 (dP/dtheta) x_hat, assembled
  # column by column as the blocks are walked. Each column is a matrix-vector
  # product against the H^-1 already formed, so the whole Jacobian costs no
  # further factorization -- against 2k full inner Laplace re-solves to
  # finite-difference the mode. `dS_by_block` is kept for the Hessian, which
  # reuses the same chain rule rather than re-forming it.
  J <- if (want_J) matrix(0, n_x, 0) else NULL
  dS_by_block <- vector("list", length(cq$blocks))
  for (m in seq_along(cq$blocks)) {
    bq <- cq$blocks[[m]]
    gm <- tryCatch(
      cpp_recov_block_grad(bq$Smat, bq$Lm, bq$full, 1.0),
      error = function(e) NULL
    )
    if (is.null(gm)) return(NULL)
    grad <- c(grad, gm)

    if (want_J) {
      dS_list <- .re_block_dSigma(bq$Lm, bq$nc, bq$full)
      if (length(dS_list) != length(gm)) return(NULL)
      dS_by_block[[m]] <- dS_list
      Jm <- vapply(dS_list, function(dSig) {
        dOm <- -bq$Om %*% dSig %*% bq$Om
        # (dP/dtheta_j) x_hat is zero outside this block, and inside it is
        # dOmega applied to each group's coefficients.
        rhs <- numeric(n_x)
        rhs[bq$idx] <- as.numeric(dOm %*% bq$b_mat)
        -as.numeric(Hinv_mode %*% rhs)
      }, numeric(n_x))
      J <- cbind(J, Jm)
    }
  }

  if (any(!is.finite(grad))) return(NULL)

  # ---- Dispersion coordinate ------------------------------------------------
  # Formed before the Hessian so its mode motion (dx_hat/dphi) and gradient value
  # are one computation the phi Hessian block reads back, rather than two. The
  # stacked theta the optimizer walks is [block coordinates ..., log phi]; the
  # random-effect half is untouched when the dispersion is held fixed.
  #
  #   dm/dphi = sum_i w_i dloglik_i/dphi
  #             - 0.5 sum_i [ w_i s_i dW_i/dphi
  #                           + q_eta_i (A dx_hat/dphi)_i
  #                           + q_z_i (A_zi dx_hat/dphi)_i ]
  #
  # The mode terms are the reason this cannot be read off the likelihood alone:
  # log|H| moves with phi both explicitly, through W(eta, phi), and through the
  # mode. They contract against the same q_eta / q_z channel the covariance
  # coordinates travel, so one derivation covers both kinds of coordinate.
  #
  # Both reduce under the mixtures this path admits, and measurably so rather
  # than by assumption (dev_notes/probe_hurdle_phi_gradient.R): a hurdle's two
  # processes are variation-independent, which zeroes dWez and dWzz's
  # eta-derivative, leaving q_eta = dw * s; and it makes H block-diagonal
  # between the zero block and the count-plus-random-effect block, so the phi
  # mode motion has no zero-block component and q_z's term is identically zero.
  # The general form is kept because q_eta / q_z are already formed in the core,
  # so writing the hurdle's special case would cost nothing and would be
  # silently wrong if the gate above ever admits an untruncated mixture. What
  # actually distinguishes the mixture here is the mask, not these terms.
  #
  # `dw` already carries the observation weights (applied in the core), so only
  # the explicit dW/dphi is scaled here.
  grad_phi <- NULL
  phi_block <- NULL
  if (isTRUE(want_phi)) {
    dphi <- .family_dphi(family)
    if (is.null(dphi)) return(NULL)
    if (!is.finite(phi) || phi <= 0) return(NULL)
    wt <- if (is.null(weights)) rep(1, n_obs) else as.numeric(weights)

    # Under a hurdle the y = 0 branch is log(pi): it carries no phi, and the
    # base family's derivatives are not even finite there (a zero-truncated
    # density is -Inf at y = 0). Zeroing them is what makes the base registry
    # the mixture's derivative rather than an approximation to it.
    dphi2 <- .family_dphi2(family)
    PD <- .laplace_phi_fields(cq, y, n_trials, family, phi, phi2, dphi, dphi2)
    if (is.null(PD)) return(NULL)

    # dx_hat/dphi from implicit differentiation of the inner stationarity
    # condition, on the OBSERVED-curvature H:
    #
    #   H_true dx_hat/dphi = A'(w d(score_eta)/dphi) + A_zi'(w d(score_z)/dphi)
    #
    # The second term is what genuine zero inflation adds. Its zero predictor is
    # logistic and carries no phi of its own, but at y = 0 the mixture weight
    # log(pi + (1 - pi) P(Y = 0, phi)) does, so the zero-predictor score moves
    # with phi through P(Y = 0). Under a hurdle that probability is zero and the
    # term vanishes identically, which is why the count-predictor solve alone was
    # right there.
    rhs_mode <- as.numeric(Matrix::crossprod(A, wt * PD[, "dsc_e_dp"]))
    if (isTRUE(cq$has_zi)) {
      rhs_mode <- rhs_mode +
        as.numeric(Matrix::crossprod(cq$A_zi, wt * PD[, "dsc_z_dp"]))
    }
    dxdphi <- as.numeric(Hinv_mode %*% rhs_mode)
    deta_dphi <- as.numeric(A %*% dxdphi)
    dz_dphi <- if (!isTRUE(cq$has_zi)) NULL else
      as.numeric(cq$A_zi %*% dxdphi)

    # The explicit dW/dphi term runs over the whole 2 x 2 block, each partial
    # contracted against the matching predictor (co)variance -- the same
    # contraction q_eta / q_z make in the eta and z directions.
    dW_term <- wt * s * PD[, "dWee_dp"]
    if (isTRUE(cq$has_zi)) {
      dW_term <- dW_term +
        wt * (2 * cq$s_ez * PD[, "dWez_dp"] + cq$s_zz * PD[, "dWzz_dp"])
    }

    g_phi <- sum(wt * PD[, "dl_dp"]) -
      0.5 * sum(dW_term + cq$q_eta * deta_dphi +
                (if (is.null(dz_dphi)) 0 else cq$q_z * dz_dphi))
    # The optimizer works in log phi, where the positivity constraint is free.
    grad_phi <- phi * g_phi
    if (!is.finite(grad_phi)) return(NULL)
    grad <- c(grad, grad_phi)

    # log-phi column of the mode Jacobian, dx_hat/dlog_phi = phi dx_hat/dphi.
    if (want_J) J <- cbind(J, phi * dxdphi)
    phi_block <- list(dphi = dphi, dphi2 = dphi2, wt = wt, PD = PD,
                      grad_phi = grad_phi, dxdphi = dxdphi,
                      deta_dphi = deta_dphi, dz_dphi = dz_dphi,
                      rhs_mode = rhs_mode)
  }

  # Closed-form theta-Hessian, gated on the second curvature derivative. A family
  # that has only the first derivative leaves H NULL, so the gradient and J are
  # still exact and only the Hessian returns to the stencil. When the dispersion
  # is estimated but has no second-order block, the phi row/column cannot be
  # formed closed either, so H is left NULL and the caller differences the
  # gradient (which carries the phi row) instead.
  # The closed Hessian differentiates the curvature once more, which under a
  # mixture means the SECOND derivatives of the 2 x 2 block. Because that block
  # is -Hess(log density), those are the fourth derivatives of one scalar: five
  # distinct fields covering all nine second partials, of which a hurdle needs
  # only two. Registered per family by has_zi_curvature_2nd_derivative(), which
  # additionally needs the second eta-derivative of the observed curvature for
  # the coupled y = 0 branch.
  #
  # Where the working weight is not the observed curvature (neg_binomial_1,
  # truncated_neg_binomial_2, and so hurdle_nbinom2), u is formed on the
  # observed-curvature inverse and must be differentiated on it too, which needs
  # d(W_obs - w)/deta. .laplace_exact_re_hess declines on its own when that is
  # unregistered, so the condition lives there rather than being duplicated
  # here.
  #
  # The dispersion border runs over both predictors as well: its dW/dpsi is the
  # full 2 x 2 block and its second mode derivative solves against both scores.
  # It still needs the family's second-order phi registry, which is what leaves
  # a hurdle over truncated_neg_binomial_2 on the stencil -- .FAMILY_DPHI2 has no
  # entry for the truncated pair, independently of the mixture.
  hess_curv_ok <- isTRUE(tryCatch(
    if (isTRUE(cq$has_zi)) cpp_family_has_zi_curvature_2nd_derivative(family)
    else cpp_family_has_curvature_2nd_derivative(family),
    error = function(e) FALSE))

  Hmat <- NULL
  if (isTRUE(want_hessian) && hess_curv_ok &&
      !(isTRUE(want_phi) && is.null(phi_block$dphi2))) {
    Hmat <- .laplace_exact_re_hess(cq, y, n_trials, family, phi, phi2, weights,
                                   dS_by_block, phi_block = phi_block)
  }

  if (!want_J) return(grad)
  if (any(!is.finite(J))) return(NULL)
  list(grad = grad, J = J, H = Hmat, grad_phi = grad_phi)
}
