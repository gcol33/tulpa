# Central-composite-design (CCD) outer integration for the joint multi-block
# nested-Laplace path (gcol33/tulpa#59).
#
# Part of the joint nested-Laplace driver; the public entry point
# tulpa_nested_laplace_joint() lives in nested_laplace_joint.R, the tensor-grid
# dispatch in nested_laplace_joint_multi.R.
#
# The tensor product over the per-block hyperparameter axes is k^d outer cells
# (d axes, k points each); each cell is one inner Newton solve. A CCD places
# 1 + 2d + 2^d nodes around the joint hyperparameter posterior mode, oriented by
# the Cholesky of its posterior covariance, and integrates with the corrected
# R-INLA design weights -- the same ccd_grid() / ccd_weights() machinery the
# SPDE (fit_spde_nested.R) and RE-covariance (nested_laplace_re_cov.R) paths
# use, here generalised to the joint path's mixed-support axes via the per-axis
# unconstraining transforms already defined for the outer Pareto-k diagnostic
# (nested_laplace_joint_pareto_k.R). For d = 4 that is 25 nodes vs 81 tensor
# cells; the gap widens with d.

# Resolve per-axis unconstraining transform tags for the joint multi-block
# LATENT grid (the b<N>.<axis> columns, phi dispersion axes excluded), reusing
# the Pareto-k registry so "which axes are safely transformable" lives in one
# place. Returns one tag ("log" / "logit01" / "identity") per latent axis, or
# NULL to DECLINE CCD when any axis has unguessable support (CAR_proper's
# rho_car, a non-BYM2 rho) -- the caller then falls back to the tensor grid.
.joint_ccd_axis_tags <- function(axis_names, axis_offsets, prepared) {
    if (length(axis_names) == 0L) return(NULL)
    pseudo <- list(
        theta_grid   = matrix(0, 1L, length(axis_names),
                              dimnames = list(NULL, axis_names)),
        axis_offsets = axis_offsets,
        blocks       = prepared
    )
    .joint_pareto_axis_tags(pseudo)
}

# Central-difference value / gradient / Hessian of an objective `eval1` (a
# closure mapping an [S x d] u-space matrix to a length-S vector of values) at
# point u. The centre, the 2d axial points (+/- h e_j) and the 4 * C(d, 2)
# mixed corners are evaluated in ONE `eval1` call, so every inner solve in a
# round fans out across the outer-grid threads.
#   g_i  = (f(+e_i) - f(-e_i)) / (2 h_i)
#   H_ii = (f(+e_i) - 2 f(0) + f(-e_i)) / h_i^2
#   H_ij = (f(+i+j) - f(+i-j) - f(-i+j) + f(-i-j)) / (4 h_i h_j)
.joint_ccd_fd_stencil <- function(u, eval1, h) {
    d <- length(u)
    rows <- list(u)                    # 1: centre
    key  <- c("0")
    for (j in seq_len(d)) {            # axials
        up <- u; up[j] <- u[j] + h[j]; rows[[length(rows) + 1L]] <- up
        dn <- u; dn[j] <- u[j] - h[j]; rows[[length(rows) + 1L]] <- dn
        key <- c(key, paste0("+", j), paste0("-", j))
    }
    if (d >= 2L) for (i in seq_len(d - 1L)) for (j in (i + 1L):d) {  # corners
        for (si in c(1, -1)) for (sj in c(1, -1)) {
            v <- u; v[i] <- u[i] + si * h[i]; v[j] <- u[j] + sj * h[j]
            rows[[length(rows) + 1L]] <- v
            key <- c(key, paste0(if (si > 0) "+" else "-", i,
                                 if (sj > 0) "+" else "-", j))
        }
    }
    U <- do.call(rbind, rows)
    f <- eval1(U)
    names(f) <- key
    f0 <- f["0"]
    g <- numeric(d)
    H <- matrix(0, d, d)
    for (j in seq_len(d)) {
        g[j]    <- (f[paste0("+", j)] - f[paste0("-", j)]) / (2 * h[j])
        H[j, j] <- (f[paste0("+", j)] - 2 * f0 + f[paste0("-", j)]) / (h[j]^2)
    }
    if (d >= 2L) for (i in seq_len(d - 1L)) for (j in (i + 1L):d) {
        hij <- (f[paste0("+", i, "+", j)] - f[paste0("+", i, "-", j)] -
                f[paste0("-", i, "+", j)] + f[paste0("-", i, "-", j)]) /
               (4 * h[i] * h[j])
        H[i, j] <- hij
        H[j, i] <- hij
    }
    list(f0 = as.numeric(f0), grad = g, hess = H)
}

# Negative-definite regularisation of a symmetric matrix: eigen-clip every
# eigenvalue to <= -ridge so the log-posterior curvature gives an ascent
# direction (and a usable Gaussian precision -H).
.joint_ccd_neg_def <- function(H, ridge_rel = 1e-6) {
    Hs <- 0.5 * (H + t(H))
    eg <- eigen(Hs, symmetric = TRUE)
    cap <- -ridge_rel * max(abs(eg$values), 1)
    ev  <- pmin(eg$values, cap)
    eg$vectors %*% (ev * t(eg$vectors))
}

# Damped-Newton mode-find for the joint hyperparameter log-posterior on the
# unconstrained scale. Each round fits a local quadratic via one batched
# stencil call (gradient + Hessian), regularises the Hessian negative-definite,
# and takes a backtracked, box-clamped Newton step uphill. Robust to the inner
# log-marginal's small non-smoothness (a least-curvature quadratic model rather
# than a gradient line search, which a finite-difference gradient breaks on).
# Returns list(par, hess, value, converged) or NULL on failure.
.joint_ccd_modefind <- function(u0, eval1, lower, upper, h,
                                max_rounds = 30L, tol = 1e-3) {
    u   <- pmin(pmax(u0, lower), upper)
    f_u <- as.numeric(eval1(matrix(u, nrow = 1L)))
    if (!is.finite(f_u)) return(NULL)
    H <- NULL
    converged <- FALSE
    for (iter in seq_len(max_rounds)) {
        st <- tryCatch(.joint_ccd_fd_stencil(u, eval1, h),
                       error = function(e) NULL)
        if (is.null(st) || any(!is.finite(st$grad)) || any(!is.finite(st$hess)))
            return(NULL)
        H     <- st$hess
        H_reg <- .joint_ccd_neg_def(H)
        step  <- tryCatch(as.numeric(-solve(H_reg, st$grad)),
                          error = function(e) NULL)
        if (is.null(step) || any(!is.finite(step))) return(NULL)
        # Backtracking line search: keep the step inside the box and uphill.
        t_step <- 1
        u_try <- u; f_try <- f_u
        repeat {
            cand   <- pmin(pmax(u + t_step * step, lower), upper)
            f_cand <- as.numeric(eval1(matrix(cand, nrow = 1L)))
            if (is.finite(f_cand) && f_cand >= f_u - 1e-8) {
                u_try <- cand; f_try <- f_cand; break
            }
            t_step <- t_step / 2
            if (t_step < 1e-4) break
        }
        delta <- max(abs(u_try - u))
        u <- u_try; f_u <- f_try
        if (delta < tol) { converged <- TRUE; break }
    }
    # Clean Hessian at the final point for the CCD scale.
    st_fin <- tryCatch(.joint_ccd_fd_stencil(u, eval1, h),
                       error = function(e) NULL)
    if (!is.null(st_fin) && all(is.finite(st_fin$hess))) H <- st_fin$hess
    list(par = u, hess = H, value = f_u, converged = converged)
}

# Build the CCD node grid for the joint multi-block path. `eval_logpost` maps a
# user-facing [S x d] theta matrix (columns = `axis_names`) to a length-S
# log-posterior vector (inner log-marginal + baked hyperprior). `axis_values`
# is a length-d list of the per-axis grid values the user supplied (used to set
# the mode-find box and the initial point). Returns
#   list(grid, dnode, u_hat, L_scale, tags)
# with `grid` the [n_node x d] physical node matrix (colnames = axis_names) and
# `dnode` the corrected CCD design weights, or NULL to fall back to the tensor
# grid (unguessable axis, degenerate mode-find, or non-PD outer Hessian).
.joint_ccd_grid <- function(axis_names, axis_offsets, prepared, axis_values,
                            eval_logpost, verbose = FALSE) {
    d <- length(axis_names)
    tags <- .joint_ccd_axis_tags(axis_names, axis_offsets, prepared)
    if (is.null(tags)) return(NULL)

    # u-space search box, generously wider than the supplied per-axis grid
    # range so the mode-find can reach a posterior mode that sits beyond the
    # user's coarse grid (the common case: the grid is a net, not a support
    # bound). The CCD axials are clamped to this box so a design point never
    # runs to a numerically-infeasible hyperparameter; the box also bounds the
    # mode-find, and a mode pinned to its (wide) edge is treated as a runaway /
    # boundary fit and declined to the tensor grid.
    u_vals <- lapply(seq_len(d), function(j)
        .joint_pareto_fwd(tags[j], as.numeric(axis_values[[j]])))
    lower0 <- vapply(u_vals, min, numeric(1))
    upper0 <- vapply(u_vals, max, numeric(1))
    u0     <- vapply(u_vals, stats::median, numeric(1))
    span   <- pmax(upper0 - lower0, 1e-3)
    pad    <- pmax(1.5 * span, 0.5)
    lower  <- lower0 - pad
    upper  <- upper0 + pad
    # Degenerate axis (a single supplied value) carries no curvature; CCD
    # cannot orient a design across it -- fall back to the tensor grid.
    if (any(!is.finite(c(lower, upper, u0))) || any(upper - lower <= 0)) {
        return(NULL)
    }

    # Map a u-space matrix to physical theta (per-axis inverse transform).
    u_to_theta <- function(U) {
        M <- matrix(0, nrow(U), ncol(U), dimnames = list(NULL, axis_names))
        for (j in seq_len(ncol(U))) {
            M[, j] <- .joint_pareto_inv(tags[j], U[, j])$theta
        }
        M
    }
    eval1 <- function(U) {
        lp <- eval_logpost(u_to_theta(U))
        lp[!is.finite(lp)] <- -1e10
        lp
    }

    # Finite-difference step on the (unconstrained) u scale: large enough to
    # average over the inner log-marginal's small non-smoothness, small enough
    # to resolve the local curvature on log / logit / identity axes.
    h_fd <- rep(0.1, d)

    mf <- .joint_ccd_modefind(u0, eval1, lower, upper, h_fd)
    if (is.null(mf)) {
        if (verbose) message("tulpa CCD: outer mode-find failed; using the ",
                             "tensor grid.")
        return(NULL)
    }
    u_hat <- mf$par
    # A mode pinned to the (wide) box edge is a runaway / boundary-supported
    # hyperparameter; the Gaussian CCD does not apply -> tensor grid.
    eps_box <- 1e-3 * pmax(upper - lower, 1)
    if (any(abs(u_hat - lower) < eps_box) || any(abs(u_hat - upper) < eps_box)) {
        if (verbose) message("tulpa CCD: outer mode pinned to the axis box ",
                             "(boundary-supported hyperparameter); using the ",
                             "tensor grid.")
        return(NULL)
    }

    H <- mf$hess
    if (is.null(H) || any(!is.finite(H))) return(NULL)
    # Hessian of the log-posterior is negative-definite at a mode; -H is the
    # Gaussian precision. Reject a flat / indefinite curvature (CCD nodes would
    # land at extreme hyperparameters where the inner Newton fails).
    neg_H <- -0.5 * (H + t(H))
    ev <- eigen(neg_H, symmetric = TRUE, only.values = TRUE)$values
    if (any(!is.finite(ev)) || max(abs(ev)) <= 0 ||
        min(ev) <= 1e-6 * max(abs(ev))) {
        if (verbose) message("tulpa CCD: outer Hessian degenerate at the mode; ",
                             "using the tensor grid.")
        return(NULL)
    }
    post_cov <- tryCatch(solve(neg_H), error = function(e) NULL)
    if (is.null(post_cov)) return(NULL)
    L_scale <- tryCatch(t(chol(post_cov)), error = function(e) NULL)
    if (is.null(L_scale)) return(NULL)

    # CCD design: factorial corners at +/- 1.1 per whitened axis (INLA's f0),
    # with the matching corrected design weights (single source of truth with
    # the SPDE / RE-cov paths; see ccd_grid.R). Map z -> u_hat + L z, clamp to
    # the box, inverse-transform to physical theta.
    ccd   <- ccd_grid(d, f_0 = sqrt(d) * 1.1)
    dnode <- ccd_weights(ccd)
    u_grid <- ccd_to_theta(ccd$z, u_hat, L_scale)
    for (j in seq_len(d)) {
        u_grid[, j] <- pmin(pmax(u_grid[, j], lower[j]), upper[j])
    }
    grid <- u_to_theta(u_grid)
    list(grid = grid, dnode = dnode, u_hat = u_hat, L_scale = L_scale,
         tags = tags)
}

# Integration weights for an outer grid carrying CCD design weights `dnode`:
# w_k proportional to dnode_k * exp(log_marginal_k). With dnode == NULL this is
# the plain softmax (uniform tensor-cell weight) of `.nl_normalise_weights`.
.joint_integration_weights <- function(log_marginal, dnode = NULL) {
    if (is.null(dnode)) return(.nl_normalise_weights(log_marginal))
    fin <- is.finite(log_marginal)
    if (!any(fin)) return(rep(NA_real_, length(log_marginal)))
    w <- dnode * exp(log_marginal - max(log_marginal[fin]))
    w[!is.finite(w)] <- 0
    w[w < 0] <- 0
    s <- sum(w)
    if (s <= 0) return(rep(NA_real_, length(log_marginal)))
    w / s
}
