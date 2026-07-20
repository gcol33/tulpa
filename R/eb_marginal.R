# ============================================================================
# Marginal (hyperparameter-corrected) empirical Bayes.
#
# tulpa_eb() reports the fixed effects conditional on theta_hat: solve(H_beta)
# is Var(beta | theta_hat, y), so its intervals omit the uncertainty in the
# variance components themselves and are narrower than the truth -- increasingly
# so as the number of groups falls, where the variance-component marginal is
# skewed and weakly determined.
#
# tulpa_re_cov_nested() answers this by integrating a grid. The correction here
# is the cheaper middle rung: keep the single maximizer, and propagate the
# curvature of the outer objective around it. Write m(theta) for the outer log
# objective (the inner log marginal plus the hyperprior) and x_hat(theta) for
# the inner mode. Near the maximizer
#
#     theta | y   ~  N(theta_hat, H_theta^-1),   H_theta = -m''(theta_hat)
#     x_hat(theta) ~  x_hat(theta_hat) + J (theta - theta_hat)
#
# with J = d x_hat / d theta. The law of total variance then reads
#
#     Var(x | y) = E_theta[Var(x | theta, y)] + Var_theta(E[x | theta, y])
#                ~  Var(x | theta_hat, y)  +  J H_theta^-1 J'
#
# so the conditional covariance gains an additive, positive-semidefinite term
# and never shrinks. Both ingredients are finite differences of quantities the
# EB fit already produces: m(theta) is the objective the outer optimizer drove,
# and x_hat(theta) is the inner solve's mode. One stencil yields both, so the
# inner Laplace solves are shared between them rather than repeated per
# quantity.
#
# This is an approximation in two places -- the Gaussian posterior for theta and
# the linearization of x_hat -- and both degrade in the same regime, a strongly
# skewed variance-component marginal. It widens intervals in the right direction
# without claiming the nested integrator's accuracy; when the variance
# components themselves are the target, integrate.
#
# Cost: the stencil is 1 + 2k^2 inner Laplace solves for k hyperparameter
# coordinates (2k for the gradient-free J and the diagonal of H, 2k(k-1) for the
# off-diagonals), doubled again under Richardson extrapolation. k is the total
# over blocks: 1 for a scalar (1 | g), 3 for a correlated (1 + x | g).
# ============================================================================


# Cached objective/mode evaluator. Every stencil point is one inner Laplace
# solve, and the two stencils of a Richardson pair share theta_hat, so the
# results are memoized on a caller-supplied key rather than recomputed. Returns
# NULL for a point whose inner solve failed or produced a non-finite objective;
# the caller treats a single NULL as "no correction" rather than silently
# working with a hole in the stencil.
.eb_fd_evaluator <- function(core) {
  cache <- new.env(parent = emptyenv())
  function(theta, key) {
    if (exists(key, envir = cache, inherits = FALSE)) {
      return(get(key, envir = cache, inherits = FALSE))
    }
    fit <- core$inner_fit(.re_cov_theta_to_L_list(theta, core$layout))
    ok <- !is.null(fit) && !is.null(fit$mode) &&
      length(fit$log_marginal) == 1L && is.finite(fit$log_marginal)
    val <- if (!ok) NULL else {
      lp <- core$log_prior_theta(theta)
      if (!is.finite(lp)) NULL else
        list(m = fit$log_marginal + lp, x = as.numeric(fit$mode))
    }
    assign(key, val, envir = cache)
    val
  }
}


# Central-difference stencil at a single step size. Returns the negative
# objective Hessian H (k x k) and the mode Jacobian J (n_x x k), or NULL if any
# stencil point failed.
#
#   J[, j]  = (x_hat(+h e_j) - x_hat(-h e_j)) / 2h
#   H[j, j] = -(m(+h e_j) - 2 m(0) + m(-h e_j)) / h^2
#   H[j, l] = -(m(++) - m(+-) - m(-+) + m(--)) / 4h^2
#
# The +/- e_j points serve J and the diagonal of H at once, which is why the two
# are built together here rather than by separate drivers.
.eb_fd_stencil <- function(ev, theta_hat, step) {
  k <- length(theta_hat)
  at <- function(delta) {
    key <- if (all(delta == 0L)) "0" else
      paste0(signif(step, 12), ":", paste(delta, collapse = ","))
    ev(theta_hat + step * delta, key)
  }
  e <- function(j, s) { d <- integer(k); d[j] <- s; d }

  f0 <- at(integer(k))
  if (is.null(f0)) return(NULL)
  n_x <- length(f0$x)
  if (n_x < 1L) return(NULL)

  J <- matrix(0, n_x, k)
  H <- matrix(0, k, k)
  for (j in seq_len(k)) {
    fp <- at(e(j, 1L))
    fm <- at(e(j, -1L))
    if (is.null(fp) || is.null(fm)) return(NULL)
    if (length(fp$x) != n_x || length(fm$x) != n_x) return(NULL)
    J[, j]  <- (fp$x - fm$x) / (2 * step)
    H[j, j] <- -(fp$m - 2 * f0$m + fm$m) / step^2
  }
  if (k > 1L) {
    for (j in seq_len(k - 1L)) {
      for (l in (j + 1L):k) {
        pp <- at(e(j, 1L)  + e(l, 1L))
        pm <- at(e(j, 1L)  + e(l, -1L))
        mp <- at(e(j, -1L) + e(l, 1L))
        mm <- at(e(j, -1L) + e(l, -1L))
        if (is.null(pp) || is.null(pm) || is.null(mp) || is.null(mm)) return(NULL)
        H[j, l] <- H[l, j] <-
          -(pp$m - pm$m - mp$m + mm$m) / (4 * step^2)
      }
    }
  }
  list(H = (H + t(H)) / 2, J = J, n_x = n_x)
}


# Names for the stacked theta coordinates, in the packing order of
# .re_cov_theta_to_L_list: per block, either the log standard deviations of a
# diagonal Sigma or the column-major lower-triangular log-Cholesky coordinates
# (log L_ii on the diagonal, raw L_ij below it).
.eb_theta_names <- function(layout) {
  unlist(lapply(seq_along(layout), function(m) {
    bl  <- layout[[m]]
    lab <- .re_cov_block_label(bl, m)
    if (!isTRUE(bl$full)) {
      if (bl$nc == 1L) paste0(lab, "_log_sd")
      else sprintf("%s_log_sd[%d]", lab, seq_len(bl$nc))
    } else {
      nm <- character(0)
      for (j in seq_len(bl$nc)) {
        for (i in j:bl$nc) {
          nm <- c(nm, if (i == j) sprintf("%s_logL[%d,%d]", lab, i, j)
                      else sprintf("%s_L[%d,%d]", lab, i, j))
        }
      }
      nm
    }
  }), use.names = FALSE)
}


# Assemble the correction at theta_hat. Returns NULL (with no warning -- the
# caller owns the message) when the stencil, the positive-definiteness check or
# either inversion fails, which leaves the conditional covariance in place
# rather than reporting a covariance built from an unusable Hessian.
.eb_marginal_correction <- function(core, theta_hat, p_fix, H_beta,
                                    step = 1e-3, richardson = FALSE) {
  k <- length(theta_hat)
  if (k < 1L) return(NULL)
  if (is.null(H_beta)) return(NULL)

  ev   <- .eb_fd_evaluator(core)
  base <- .eb_fd_stencil(ev, theta_hat, step)
  if (is.null(base)) return(NULL)

  # Central differences carry O(step^2) truncation error, so evaluating the same
  # stencil at step/2 and combining as (4 A(step/2) - A(step)) / 3 cancels the
  # leading term and leaves O(step^4). It applies to J and H alike because both
  # are central differences. It doubles the solve count and amplifies the inner
  # solver's own noise, so it pays only when that noise is well below the
  # truncation error -- a tight inner tolerance.
  if (isTRUE(richardson)) {
    half <- .eb_fd_stencil(ev, theta_hat, step / 2)
    if (!is.null(half) && identical(half$n_x, base$n_x)) {
      base$H <- (4 * half$H - base$H) / 3
      base$J <- (4 * half$J - base$J) / 3
      base$H <- (base$H + t(base$H)) / 2
    }
  }

  # H_theta must be positive definite for theta_hat to be a maximizer and for
  # H_theta^-1 to be a covariance. It fails when theta_hat is not actually
  # interior (a collapsed variance component sitting on the boundary is the
  # usual cause) or when the step is small enough that the inner solver's noise
  # dominates the second difference.
  evals <- tryCatch(eigen(base$H, symmetric = TRUE, only.values = TRUE)$values,
                    error = function(e) NULL)
  if (is.null(evals) || any(!is.finite(evals)) || min(evals) <= 0) return(NULL)

  H_inv <- tryCatch(solve(base$H), error = function(e) NULL)
  if (is.null(H_inv)) return(NULL)

  cov_cond <- tryCatch(solve(H_beta), error = function(e) NULL)
  if (is.null(cov_cond)) return(NULL)
  p <- min(as.integer(p_fix), nrow(cov_cond), nrow(base$J))
  if (p < 1L) return(NULL)
  idx      <- seq_len(p)
  cov_cond <- cov_cond[idx, idx, drop = FALSE]
  J_beta   <- base$J[idx, , drop = FALSE]

  inflation <- J_beta %*% H_inv %*% t(J_beta)
  cov_marg  <- cov_cond + inflation
  cov_marg  <- (cov_marg + t(cov_marg)) / 2

  th_names <- tryCatch(.eb_theta_names(core$layout), error = function(e) NULL)
  if (!is.null(th_names) && length(th_names) == k) {
    dimnames(base$H) <- list(th_names, th_names)
    dimnames(H_inv)  <- list(th_names, th_names)
    colnames(base$J) <- th_names
  }

  list(
    cov_marginal    = cov_marg,
    cov_conditional = cov_cond,
    inflation       = inflation,
    H_theta         = base$H,
    theta_cov       = H_inv,
    J               = base$J,
    theta_names     = th_names,
    step            = step,
    richardson      = isTRUE(richardson)
  )
}


# Posterior variance of each block's hyperparameter coordinates, read off a fit
# that carries the outer curvature this file produces. A sampler warm start uses
# it to mass the log_sigma_re slots, which otherwise start at 1 and adapt.
#
# Returns a per-block list, NULL for a block whose coordinates do not map onto
# the sampler's log_sigma_re: a correlated block is in log-Cholesky coordinates,
# where log L_ii is not log sigma_i (sigma_i is the norm of row i of L), so its
# variances are not the sampler's. NULL overall when the fit carries no
# `theta_cov`, or when it does not line up with the block layout -- in which
# case every mass stays at its adapting default rather than being filled from a
# mismatched vector.
.warm_start_hyper_var <- function(fit) {
  tc  <- fit$theta_cov
  lay <- fit$layout
  if (!is.matrix(tc) || !is.list(lay) || length(lay) == 0L) return(NULL)
  v <- tryCatch(diag(tc), error = function(e) NULL)
  if (is.null(v) || !all(is.finite(v)) || any(v <= 0)) return(NULL)
  ks <- tryCatch(vapply(lay, function(b) as.integer(b$k), integer(1)),
                 error = function(e) NULL)
  if (is.null(ks) || sum(ks) != length(v)) return(NULL)
  pos <- 0L
  out <- vector("list", length(lay))
  for (m in seq_along(lay)) {
    idx <- pos + seq_len(ks[m])
    pos <- pos + ks[m]
    # Leave a skipped block's slot at its preallocated NULL. Assigning NULL with
    # [[<-, rather than skipping, would DELETE the element and shift every later
    # block down one, so term t would be massed from term t+1's curvature.
    if (!isTRUE(lay[[m]]$full)) out[[m]] <- unname(v[idx])
  }
  out
}
