# Areal spatially-varying coefficient (SVC) on a joint multi-block ICAR field
# (Gap B / Fix B of dev_notes/plan_n_field_copy.md).
#
# An areal block (icar / bym2 / car_proper) contributes its field with a
# per-observation design weight:
#     eta_i += svc_weight[[k]][i] * amplitude * z[cell_i]
# This is the INLA f(cell, weight, copy=...) semantics for the downstream
# trend field (tulpaObs#15), where `weight` is a per-row `time` covariate.
#
# Coverage:
#   (1) FD gate (correctness-critical): the joint log-posterior GRADIENT wrt
#       the field latent z and a beta matches a central finite difference of
#       the joint log-posterior at a fixed single grid point, to ~1e-4. This
#       catches a double-counted or wrong-weight scatter that a recovery test
#       alone cannot.
#   (2) Identity: svc_weight = 1 everywhere reproduces the no-svc fit exactly.
#   (3) Recovery: a 2-arm chain-graph model where one ICAR field enters arm-2
#       weighted by a per-row covariate w_i; the weighted fit recovers the
#       field (|cor| > 0.7) and recovers it materially better than a fit that
#       ignores the weight.
#   (4) Validation: bad svc_weight lengths error before the C++ entry.

skip_on_cran()
skip_if_fast()

.chain_adj_svc <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn),
         n_spatial_units = n_s)
}

# --------------------------------------------------------------------------- #
# (1) FD gate                                                                 #
# --------------------------------------------------------------------------- #

test_that("joint log-posterior gradient matches central FD with areal svc_weight", {
    set.seed(11)
    n_s <- 8L
    adj <- .chain_adj_svc(n_s)

    N1 <- 25L
    N2 <- 30L
    s1 <- sample.int(n_s, N1, replace = TRUE)
    s2 <- sample.int(n_s, N2, replace = TRUE)
    X1 <- cbind(1, rnorm(N1))
    X2 <- cbind(1, rnorm(N2))
    w1 <- rep(1.0, N1)                 # donor arm: unit weight
    w2 <- runif(N2, 0.3, 2.0)          # copy arm: non-trivial per-row weight
    y1 <- rnorm(N1)
    y2 <- rnorm(N2)

    arm <- function(y, X, s) list(
        y = y, n_trials = rep(1L, length(y)), X = X,
        re_idx = rep(0, length(y)), n_re_groups = 0L, sigma_re = 1.0,
        spatial_idx = s, family = "gaussian", phi = 1.0)
    arms_list <- list(arm(y1, X1, s1), arm(y2, X2, s2))
    # normalise the way the dispatcher does (fills field_coef / cell-coupling).
    arms_norm <- lapply(seq_along(arms_list), function(k)
        tulpa:::.normalise_joint_arm_multi(arms_list[[k]], k))

    block_spec_R <- list(
        type = "icar",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr,
        adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors,
        spatial_idx = list(s1, s2),
        svc_weight  = list(w1, w2)
    )
    bs_cpp <- tulpa:::.joint_block_spec_for_cpp(block_spec_R, n_arms = 2L,
                                                block_index = 1L,
                                                arms = arms_norm)

    # Single grid point: non-copy ICAR has one axis (tau). Fix tau = 1.3.
    theta_grid   <- matrix(1.3, nrow = 1L, ncol = 1L)
    axis_offsets <- as.integer(c(0L, 1L))

    # n_x = p1 + p2 + n_s (no RE). Build a non-trivial x.
    n_x <- ncol(X1) + ncol(X2) + n_s
    set.seed(99)
    x0 <- rnorm(n_x, sd = 0.4)

    ev <- function(x) {
        tulpa:::cpp_test_joint_logpost_grad(
            arms_list = arms_norm,
            copy_arms = integer(0), copy_blocks = integer(0),
            blocks_spec = list(bs_cpp),
            theta_grid = theta_grid, axis_offsets = axis_offsets,
            x = x, k_grid = 0L)
    }

    base <- ev(x0)
    expect_equal(base$n_x, n_x)
    g_analytic <- base$grad

    # Central FD over every coordinate (small n_x).
    h <- 1e-5
    g_fd <- numeric(n_x)
    for (j in seq_len(n_x)) {
        xp <- x0; xp[j] <- xp[j] + h
        xm <- x0; xm[j] <- xm[j] - h
        g_fd[j] <- (ev(xp)$logpost - ev(xm)$logpost) / (2 * h)
    }

    # Field-latent coords are the trailing n_s entries; betas are the leading.
    field_idx <- (ncol(X1) + ncol(X2) + 1L):n_x
    beta_idx  <- seq_len(ncol(X1) + ncol(X2))

    max_err <- max(abs(g_analytic - g_fd))
    expect_lt(max_err, 1e-4)
    # Spell out the two required sub-checks explicitly.
    expect_lt(max(abs(g_analytic[field_idx] - g_fd[field_idx])), 1e-4)
    expect_lt(max(abs(g_analytic[beta_idx]  - g_fd[beta_idx])),  1e-4)

    # The weighted field gradient must genuinely depend on w2 (guard against a
    # silently-dropped weight): zeroing w2 changes the field gradient.
    bs_cpp0 <- bs_cpp; bs_cpp0$svc_weight <- list(w1, rep(0.0, N2))
    g0 <- tulpa:::cpp_test_joint_logpost_grad(
        arms_list = arms_norm,
        copy_arms = integer(0), copy_blocks = integer(0),
        blocks_spec = list(bs_cpp0),
        theta_grid = theta_grid, axis_offsets = axis_offsets,
        x = x0, k_grid = 0L)$grad
    expect_gt(max(abs(g0[field_idx] - g_analytic[field_idx])), 1e-6)
})

# --------------------------------------------------------------------------- #
# (2) svc_weight = 1 reproduces the no-svc fit                                 #
# --------------------------------------------------------------------------- #

test_that("svc_weight of all ones reproduces the no-svc joint fit", {
    set.seed(21)
    n_s <- 12L
    adj <- .chain_adj_svc(n_s)
    N1 <- 200L; N2 <- 200L
    s1 <- sample.int(n_s, N1, replace = TRUE)
    s2 <- sample.int(n_s, N2, replace = TRUE)
    X1 <- cbind(1, rnorm(N1)); X2 <- cbind(1, rnorm(N2))
    f  <- cumsum(rnorm(n_s)); f <- (f - mean(f)) / sd(f)
    y1 <- rnorm(N1, X1 %*% c(0.1, 0.3) + f[s1], 0.3)
    y2 <- rnorm(N2, X2 %*% c(-0.2, 0.4) + 0.8 * f[s2], 0.3)

    arm <- function(y, X, s) list(
        y = y, n_trials = rep(1L, length(y)), X = X,
        re_idx = rep(0, length(y)), n_re_groups = 0L, sigma_re = 1.0,
        spatial_idx = s, family = "gaussian", phi = 1.0)

    base_block <- function(svc) {
        b <- list(type = "icar",
                  n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr,
                  adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors,
                  sigma_grid  = c(0.6, 1.0, 1.4),
                  spatial_idx = list(s1, s2))
        if (!is.null(svc)) b$svc_weight <- svc
        b
    }

    fit_plain <- tulpa_nested_laplace_joint(
        responses = list(occ = arm(y1, X1, s1), pos = arm(y2, X2, s2)),
        prior = list(base_block(NULL)),
        copy = list(list(arm = "pos", block = 1L, alpha_grid = c(0.5, 0.8, 1.1))),
        control = list(max_iter = 50L, tol = 1e-7))
    fit_ones <- tulpa_nested_laplace_joint(
        responses = list(occ = arm(y1, X1, s1), pos = arm(y2, X2, s2)),
        prior = list(base_block(list(rep(1.0, N1), rep(1.0, N2)))),
        copy = list(list(arm = "pos", block = 1L, alpha_grid = c(0.5, 0.8, 1.1))),
        control = list(max_iter = 50L, tol = 1e-7))

    expect_equal(as.numeric(fit_ones$log_marginal),
                 as.numeric(fit_plain$log_marginal),
                 tolerance = 1e-8,
                 info = "svc_weight = 1 must leave the areal fit unchanged")
})

# --------------------------------------------------------------------------- #
# (3) Recovery: weighted field, and the weight materially helps               #
# --------------------------------------------------------------------------- #

test_that("areal svc_weight recovers a per-row-weighted field; ignoring it is worse", {
    set.seed(31)
    n_s <- 30L
    adj <- .chain_adj_svc(n_s)

    f <- cumsum(rnorm(n_s)); f <- (f - mean(f)); f <- f / sd(f)
    sigma_true <- 1.2

    # Arm 2 is the dominant source of field information (and the only arm
    # carrying the per-row weight). Arm 1 is small + noisy so that dropping
    # arm 2's weight genuinely degrades the recovered field rather than
    # leaving arm 1 to pin it.
    N1 <- 120L; N2 <- 1200L
    s1 <- sample.int(n_s, N1, replace = TRUE)
    s2 <- sample.int(n_s, N2, replace = TRUE)
    X1 <- cbind(1, rnorm(N1)); X2 <- cbind(1, rnorm(N2))
    # Per-row weight on arm 2 (the trend column): a mean-zero covariate so
    # ignoring it (treating it as a constant 1) scrambles the field's sign
    # contribution and genuinely degrades the estimate.
    w2 <- runif(N2, -1.5, 1.5)

    eta1 <- X1 %*% c(0.2, 0.5) + sigma_true * f[s1]
    eta2 <- X2 %*% c(-0.1, 0.3) + w2 * sigma_true * f[s2]
    # Arm 1 is deliberately noisy (cannot pin the field on its own); arm 2
    # is clean and dominant.
    y1 <- rnorm(N1, eta1, 1.5)
    y2 <- rnorm(N2, eta2, 0.3)

    arm <- function(y, X, s) list(
        y = y, n_trials = rep(1L, length(y)), X = X,
        re_idx = rep(0, length(y)), n_re_groups = 0L, sigma_re = 1.0,
        spatial_idx = s, family = "gaussian", phi = 1.0)

    icar_block <- function(svc) {
        b <- list(type = "icar",
                  n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr,
                  adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors,
                  sigma_grid  = c(0.8, 1.1, 1.4),
                  spatial_idx = list(s1, s2))
        if (!is.null(svc)) b$svc_weight <- svc
        b
    }

    field_hat <- function(fit) {
        fs <- fit$arm_layout$field_starts[1L]
        cols <- fs + seq_len(n_s)
        as.numeric(crossprod(fit$weights, fit$modes[, cols, drop = FALSE]))
    }

    # Weighted fit: the copy arm carries the per-row weight w2.
    fit_w <- suppressWarnings(tulpa_nested_laplace_joint(
        responses = list(occ = arm(y1, X1, s1), pos = arm(y2, X2, s2)),
        prior = list(icar_block(list(rep(1.0, N1), w2))),
        copy = list(list(arm = "pos", block = 1L, alpha_grid = c(0.8, 1.0, 1.2))),
        control = list(max_iter = 50L, tol = 1e-6)))

    # Ignoring fit: same data, but the weight is dropped (copy arm sees the
    # field with uniform weight 1). The trend information is mis-modeled.
    fit_ign <- suppressWarnings(tulpa_nested_laplace_joint(
        responses = list(occ = arm(y1, X1, s1), pos = arm(y2, X2, s2)),
        prior = list(icar_block(NULL)),
        copy = list(list(arm = "pos", block = 1L, alpha_grid = c(0.8, 1.0, 1.2))),
        control = list(max_iter = 50L, tol = 1e-6)))

    expect_s3_class(fit_w, "tulpa_nested_laplace_joint_multi")
    expect_true(all(is.finite(fit_w$log_marginal)))

    cor_w   <- abs(cor(field_hat(fit_w),   f))
    cor_ign <- abs(cor(field_hat(fit_ign), f))

    expect_gt(cor_w, 0.7)
    # The weighted fit recovers the field materially better than the one that
    # ignores the per-row weight.
    expect_gt(cor_w, cor_ign + 0.05)
})

# --------------------------------------------------------------------------- #
# (4) Validation                                                              #
# --------------------------------------------------------------------------- #

test_that("bad svc_weight lengths error before the C++ entry", {
    set.seed(41)
    n_s <- 10L
    adj <- .chain_adj_svc(n_s)
    N <- 40L
    s <- sample.int(n_s, N, replace = TRUE)
    X <- cbind(1, rnorm(N))
    arm <- list(y = rnorm(N), n_trials = rep(1L, N), X = X,
                re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
                spatial_idx = s, family = "gaussian", phi = 1.0)
    blk <- function(svc) list(
        type = "icar",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr,
        adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors,
        sigma_grid  = c(0.5, 1.0),
        spatial_idx = list(s, s),
        svc_weight  = svc)

    expect_error(
        tulpa_nested_laplace_joint(
            responses = list(a = arm, b = arm),
            prior = list(blk(list(rep(1.0, N), rep(1.0, N - 1L)))),
            copy = NULL,
            control = list(max_iter = 5L, tol = 1e-4)),
        "svc_weight")
})
