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
# `dS_by_block` is the same per-block dSigma list the gradient's Jacobian uses,
# passed in so the chain rule is formed once. Returns NULL on a non-finite
# second curvature derivative.
.laplace_exact_re_hess <- function(cq, y, n_trials, family, phi, phi2, weights,
                                   dS_by_block, phi_block = NULL) {
  A <- cq$A; Hinv <- cq$Hinv; x <- cq$x; n_x <- cq$n_x
  Hinv_mode <- cq$Hinv_mode
  s <- cq$s; dwdeta <- cq$dw; u <- cq$u; v_r <- cq$v_r
  blocks <- cq$blocks; nb <- length(blocks)

  d2w <- cpp_family_curvature_deta2_vec(as.numeric(y), as.integer(n_trials),
                                        cq$eta, family, phi, phi2)
  if (!is.null(weights)) d2w <- d2w * as.numeric(weights)
  if (any(!is.finite(d2w))) return(NULL)

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
    dW_k    <- dwdeta * eta_dot
    dH_k    <- as.matrix(Matrix::crossprod(A, Matrix::Diagonal(x = dW_k) %*% A)) + dP_k
    dHinv_k <- -Hinv %*% dH_k %*% Hinv
    ds_k    <- rowSums(as.matrix(A %*% dHinv_k) * as.matrix(A))
    dr_k    <- (d2w * eta_dot) * s + dwdeta * ds_k
    du_k    <- as.numeric(dHinv_k %*% v_r) +
               as.numeric(Hinv %*% as.numeric(Matrix::crossprod(A, dr_k)))

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
    eta <- cq$eta
    dphi <- phi_block$dphi; dphi2 <- phi_block$dphi2; wt <- phi_block$wt
    dxdphi <- phi_block$dxdphi; deta_dphi <- phi_block$deta_dphi
    Sc1  <- dphi$dscore(eta, y, n_trials, phi)
    DW   <- dphi$dweight(eta, y, n_trials, phi)
    L2   <- dphi2$dloglik2(eta, y, n_trials, phi)
    Sc2  <- dphi2$dscore2(eta, y, n_trials, phi)
    DW2  <- dphi2$dweight2(eta, y, n_trials, phi)
    DWde <- dphi2$dweight_deta(eta, y, n_trials, phi)
    if (any(!is.finite(c(Sc1, DW, L2, Sc2, DW2, DWde)))) return(NULL)

    # J_psi = phi dx/dphi is the log-phi mode-Jacobian column; eta_dot its
    # linear-predictor image. d(H_joint)/dpsi moves through the mode
    # (dwdeta * eta_dot) and explicitly (phi wt DW), with no dP term.
    J_psi   <- phi * dxdphi
    eta_dot <- phi * deta_dphi
    dW_psi  <- dwdeta * eta_dot + phi * (wt * DW)
    dH_psi  <- as.matrix(Matrix::crossprod(A, Matrix::Diagonal(x = dW_psi) %*% A))
    dHinv_psi <- -Hinv %*% dH_psi %*% Hinv
    ds_psi  <- rowSums(as.matrix(A %*% dHinv_psi) * as.matrix(A))
    dr_psi  <- (d2w * eta_dot + phi * (wt * DWde)) * s + dwdeta * ds_psi
    du_psi  <- as.numeric(dHinv_psi %*% v_r) +
               as.numeric(Hinv %*% as.numeric(Matrix::crossprod(A, dr_psi)))

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
    # mode derivative d(dx/dphi)/dpsi is the one genuinely new solve; the
    # identity Sc1_de = -DW (score's eta-derivative is -W_obs) supplies dSc1/dpsi.
    ATwSc1 <- as.numeric(Matrix::crossprod(A, wt * Sc1))
    dA <- sum(wt * (Sc1 * eta_dot + phi * L2))
    d_ddw_dpsi <- d2w * eta_dot + phi * (wt * DWde)         # d(dwdeta)/dpsi
    rhs_Sc1    <- (-DW) * eta_dot + phi * Sc2               # dSc1/dpsi
    d2x_dpsi   <- as.numeric(dHinv_psi %*% ATwSc1) +
                  as.numeric(Hinv_mode %*% as.numeric(Matrix::crossprod(A, wt * rhs_Sc1)))
    d_deta_dphi_dpsi <- as.numeric(A %*% d2x_dpsi)
    dB <- sum(ds_psi * (wt * DW + dwdeta * deta_dphi) +
              s * (wt * (DWde * eta_dot + phi * DW2) +
                   d_ddw_dpsi * deta_dphi + dwdeta * d_deta_dphi_dpsi))
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
  # The dispersion block reads .family_dphi(), which registers the BASE family's
  # phi derivatives. Under a mixture the y = 0 branch carries its own phi
  # dependence through P(Y = 0), so those derivatives describe a different
  # objective. Decline rather than return a plausible wrong number; the caller
  # refuses `estimate_phi` on this path.
  if (isTRUE(cq$has_zi) && isTRUE(want_phi)) return(NULL)
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
  #             - 0.5 sum_i s_i [ w_i dW_i/dphi + (dW_i/deta_i)(A dx_hat/dphi)_i ]
  #
  # The second bracket is the whole reason this cannot be read off the
  # likelihood alone: log|H| moves with phi both explicitly, through W(eta, phi),
  # and through the mode. `dw` already carries the observation weights (applied
  # above), so only the explicit dW/dphi is scaled here.
  grad_phi <- NULL
  phi_block <- NULL
  if (isTRUE(want_phi)) {
    dphi <- .family_dphi(family)
    if (is.null(dphi)) return(NULL)
    if (!is.finite(phi) || phi <= 0) return(NULL)
    wt <- if (is.null(weights)) rep(1, n_obs) else as.numeric(weights)

    dl_dphi <- dphi$dloglik(eta, y, n_trials, phi)
    ds_dphi <- dphi$dscore(eta, y, n_trials, phi)
    dW_dphi <- dphi$dweight(eta, y, n_trials, phi)
    if (any(!is.finite(dl_dphi)) || any(!is.finite(ds_dphi)) ||
        any(!is.finite(dW_dphi))) return(NULL)

    # dx_hat/dphi from implicit differentiation of the inner stationarity
    # condition, on the OBSERVED-curvature H: H_true dx_hat/dphi = A'(w dscore/dphi).
    # `.family_dphi` is registered only where working and observed weights
    # coincide, so Hinv_mode is Hinv here -- but reading Hinv_mode keeps the mode
    # motion correct by construction rather than by that coincidence.
    dxdphi <- as.numeric(Hinv_mode %*% as.numeric(Matrix::crossprod(A, wt * ds_dphi)))
    deta_dphi <- as.numeric(A %*% dxdphi)

    g_phi <- sum(wt * dl_dphi) -
      0.5 * sum(s * (wt * dW_dphi + dw * deta_dphi))
    # The optimizer works in log phi, where the positivity constraint is free.
    grad_phi <- phi * g_phi
    if (!is.finite(grad_phi)) return(NULL)
    grad <- c(grad, grad_phi)

    # log-phi column of the mode Jacobian, dx_hat/dlog_phi = phi dx_hat/dphi.
    if (want_J) J <- cbind(J, phi * dxdphi)
    phi_block <- list(dphi = dphi, dphi2 = .family_dphi2(family), wt = wt,
                      grad_phi = grad_phi, dxdphi = dxdphi, deta_dphi = deta_dphi)
  }

  # Closed-form theta-Hessian, gated on the second curvature derivative. A family
  # that has only the first derivative leaves H NULL, so the gradient and J are
  # still exact and only the Hessian returns to the stencil. When the dispersion
  # is estimated but has no second-order block, the phi row/column cannot be
  # formed closed either, so H is left NULL and the caller differences the
  # gradient (which carries the phi row) instead.
  # The closed Hessian differentiates the curvature once more, which under a
  # mixture means the SECOND derivatives of the 2 x 2 block -- eighteen fields
  # rather than six, and not registered. The gradient and J stay exact, so the
  # caller central-differences this exact gradient (2k gradient evaluations)
  # instead of falling all the way back to re-solving the objective (1 + 2k^2).
  #
  # It also needs the working weight to BE the observed curvature. Its
  # du/dtheta differentiates u through Hinv, while u itself is formed on
  # Hinv_mode; where the two inverses differ (neg_binomial_1,
  # truncated_neg_binomial_2) that pairing is inconsistent, so those families
  # take the same gradient stencil.
  Hmat <- NULL
  if (isTRUE(want_hessian) && !isTRUE(cq$has_zi) &&
      isTRUE(cq$working_is_observed) &&
      isTRUE(cpp_family_has_curvature_2nd_derivative(family)) &&
      !(isTRUE(want_phi) && is.null(phi_block$dphi2))) {
    Hmat <- .laplace_exact_re_hess(cq, y, n_trials, family, phi, phi2, weights,
                                   dS_by_block, phi_block = phi_block)
  }

  if (!want_J) return(grad)
  if (any(!is.finite(J))) return(NULL)
  list(grad = grad, J = J, H = Hmat, grad_phi = grad_phi)
}
