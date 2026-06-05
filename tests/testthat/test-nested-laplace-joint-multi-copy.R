# Joint multi-block nested-Laplace: N coupled fields (multiple copy blocks).
#
# Two independent shared ICAR fields f1, f2 on a 1-D chain graph, each
# coupled onto arm 2 with its own copy-scale alpha. Validates that the
# list-of-specs `copy` resolves both fields, integrates each over its own
# alpha axis, and recovers both copy scales and both field shapes.
#
#   arm1 (gaussian): eta1 = X1 b1 + sigma1 * f1[s] + sigma2 * f2[s]
#   arm2 (gaussian): eta2 = X2 b2 + alpha1*sigma1*f1[s] + alpha2*sigma2*f2[s]

.chain_adj_copy <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn),
         n_spatial_units = n_s)
}

test_that("joint multi-block recovers TWO coupled ICAR fields (list-of-specs copy)", {
    set.seed(1)
    n_s <- 60L
    adj <- .chain_adj_copy(n_s)

    # Two independent smooth fields over the chain, centered + unit-scaled.
    f1 <- cumsum(rnorm(n_s, 0, 1)); f1 <- (f1 - mean(f1)); f1 <- f1 / sd(f1)
    f2 <- cumsum(rnorm(n_s, 0, 1)); f2 <- (f2 - mean(f2)); f2 <- f2 / sd(f2)

    sigma1_true <- 1.0
    sigma2_true <- 1.0
    alpha1_true <- 1.4
    alpha2_true <- 0.7

    N1 <- 800L
    N2 <- 800L
    s1 <- sample.int(n_s, N1, replace = TRUE)
    s2 <- sample.int(n_s, N2, replace = TRUE)
    X1 <- cbind(1, rnorm(N1))
    X2 <- cbind(1, rnorm(N2))

    eta1 <- X1 %*% c(0.2, 0.5) +
            sigma1_true * f1[s1] + sigma2_true * f2[s1]
    eta2 <- X2 %*% c(-0.1, 0.3) +
            alpha1_true * sigma1_true * f1[s2] +
            alpha2_true * sigma2_true * f2[s2]
    y1 <- rnorm(N1, eta1, 0.3)
    y2 <- rnorm(N2, eta2, 0.3)

    arm1 <- list(y = y1, n_trials = rep(1L, N1), X = X1,
                 re_idx = rep(0, N1), n_re_groups = 0L, sigma_re = 1.0,
                 family = "gaussian", phi = 1.0)
    arm2 <- list(y = y2, n_trials = rep(1L, N2), X = X2,
                 re_idx = rep(0, N2), n_re_groups = 0L, sigma_re = 1.0,
                 family = "gaussian", phi = 1.0)

    icar_block <- function(sigma_grid) {
        list(type = "icar",
             n_spatial_units = adj$n_spatial_units,
             adj_row_ptr = adj$adj_row_ptr,
             adj_col_idx = adj$adj_col_idx,
             n_neighbors = adj$n_neighbors,
             sigma_grid = sigma_grid,
             spatial_idx = list(s1, s2))
    }
    prior <- list(
        icar_block(c(0.6, 0.9, 1.2)),
        icar_block(c(0.6, 0.9, 1.2))
    )

    alpha_grid1 <- c(0.8, 1.1, 1.4, 1.7)
    alpha_grid2 <- c(0.4, 0.7, 1.0)

    fit <- suppressWarnings(
        tulpa_nested_laplace_joint(
            responses = list(occ = arm1, pos = arm2),
            prior = prior,
            copy = list(
                list(arm = "pos", block = 1L, alpha_grid = alpha_grid1),
                list(arm = "pos", block = 2L, alpha_grid = alpha_grid2)
            ),
            control = list(max_iter = 50L, tol = 1e-6)
        )
    )

    expect_s3_class(fit, "tulpa_nested_laplace_joint_multi")
    expect_true(all(is.finite(fit$log_marginal)))
    expect_length(fit$block_moments, 2L)

    # Each copy block exposes its own (sigma, alpha) axis pair.
    expect_named(fit$block_moments[[1L]]$mean, c("sigma", "alpha"))
    expect_named(fit$block_moments[[2L]]$mean, c("sigma", "alpha"))

    alpha1_hat <- fit$block_moments[[1L]]$mean[["alpha"]]
    alpha2_hat <- fit$block_moments[[2L]]$mean[["alpha"]]

    # Both alpha posterior means recovered within ~0.5 of truth.
    expect_lt(abs(alpha1_hat - alpha1_true), 0.5)
    expect_lt(abs(alpha2_hat - alpha2_true), 0.5)

    # Posterior-mean fields: weight the per-grid mode rows, slice each
    # field's latent columns via arm_layout$field_starts (0-based offsets,
    # one per spatial block in block order).
    fs <- fit$arm_layout$field_starts
    expect_length(fs, 2L)
    w  <- fit$weights
    field_hat <- function(start0) {
        cols <- start0 + seq_len(n_s)     # 1-based columns into modes
        as.numeric(crossprod(w, fit$modes[, cols, drop = FALSE]))
    }
    f1_hat <- field_hat(fs[1L])
    f2_hat <- field_hat(fs[2L])

    # Both field shapes recovered up to sign / scale.
    expect_gt(abs(cor(f1_hat, f1)), 0.7)
    expect_gt(abs(cor(f2_hat, f2)), 0.7)

    # The two coupled fields are distinguishable (not collapsed onto each
    # other): the recovered fields track their own truth more strongly than
    # the other field's truth.
    expect_gt(abs(cor(f1_hat, f1)), abs(cor(f1_hat, f2)))
    expect_gt(abs(cor(f2_hat, f2)), abs(cor(f2_hat, f1)))
})

test_that("duplicate copy block targets error", {
    set.seed(7)
    n_s <- 10L
    adj <- .chain_adj_copy(n_s)
    N <- 40L
    s <- sample.int(n_s, N, replace = TRUE)
    X <- cbind(1, rnorm(N))
    arm <- list(y = rnorm(N), n_trials = rep(1L, N), X = X,
                re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
                family = "gaussian", phi = 1.0)
    icar_block <- list(type = "icar",
                       n_spatial_units = adj$n_spatial_units,
                       adj_row_ptr = adj$adj_row_ptr,
                       adj_col_idx = adj$adj_col_idx,
                       n_neighbors = adj$n_neighbors,
                       sigma_grid = c(0.5, 1.0),
                       spatial_idx = list(s, s))
    expect_error(
        tulpa_nested_laplace_joint(
            responses = list(a = arm, b = arm),
            prior = list(icar_block, icar_block),
            copy = list(
                list(arm = "b", block = 1L, alpha_grid = c(0.5, 1.0)),
                list(arm = "b", block = 1L, alpha_grid = c(0.5, 1.0))
            )
        ),
        "distinct block"
    )
})

test_that("joint multi-block recovers a copied TEMPORAL (rw1) field", {
    # gcol33/tulpa#76: copy (alpha-coupling) on a non-spatial block. A shared
    # smooth temporal trend f[t] is donated to arm1 at amplitude sigma and
    # copied onto arm2 at amplitude alpha * sigma.
    set.seed(11)
    n_t <- 40L

    f <- cumsum(rnorm(n_t, 0, 1)); f <- f - mean(f); f <- f / sd(f)

    sigma_true <- 1.0
    alpha_true <- 1.6

    N1 <- 900L
    N2 <- 900L
    t1 <- sample.int(n_t, N1, replace = TRUE)
    t2 <- sample.int(n_t, N2, replace = TRUE)
    X1 <- cbind(1, rnorm(N1))
    X2 <- cbind(1, rnorm(N2))

    eta1 <- X1 %*% c(0.2, 0.5) + sigma_true * f[t1]
    eta2 <- X2 %*% c(-0.1, 0.3) + alpha_true * sigma_true * f[t2]
    y1 <- rnorm(N1, eta1, 0.3)
    y2 <- rnorm(N2, eta2, 0.3)

    arm1 <- list(y = y1, n_trials = rep(1L, N1), X = X1,
                 re_idx = rep(0, N1), n_re_groups = 0L, sigma_re = 1.0,
                 family = "gaussian", phi = 1.0)
    arm2 <- list(y = y2, n_trials = rep(1L, N2), X = X2,
                 re_idx = rep(0, N2), n_re_groups = 0L, sigma_re = 1.0,
                 family = "gaussian", phi = 1.0)

    rw1_block <- list(type = "rw1", n_times = n_t,
                      sigma_grid = c(0.6, 0.9, 1.2),
                      temporal_idx = list(t1, t2))

    fit <- suppressWarnings(
        tulpa_nested_laplace_joint(
            responses = list(occ = arm1, pos = arm2),
            prior = list(rw1_block),
            copy = list(arm = "pos", block = 1L,
                        alpha_grid = c(1.0, 1.3, 1.6, 1.9)),
            control = list(max_iter = 50L, tol = 1e-6)
        )
    )

    expect_s3_class(fit, "tulpa_nested_laplace_joint_multi")
    expect_true(all(is.finite(fit$log_marginal)))
    expect_named(fit$block_moments[[1L]]$mean, c("sigma", "alpha"))

    alpha_hat <- fit$block_moments[[1L]]$mean[["alpha"]]
    expect_lt(abs(alpha_hat - alpha_true), 0.5)

    # Posterior-mean temporal field: the rw1 block is the first (only) block,
    # so its latent columns start at block_start[1] (0-based).
    bs0  <- fit$arm_layout$block_start[1L]
    w    <- fit$weights
    cols <- bs0 + seq_len(n_t)
    f_hat <- as.numeric(crossprod(w, fit$modes[, cols, drop = FALSE]))
    expect_gt(abs(cor(f_hat, f)), 0.7)
})

test_that("joint multi-block recovers a copied TEMPORAL (ar1) field", {
    # gcol33/tulpa#76: ar1 copy carries an extra rho axis, so the donor / copy
    # amplitude axes sit at (axis0, axis0 + 1) and rho moves to axis0 + 2.
    # Exercises that axis shift end-to-end.
    set.seed(12)
    n_t <- 40L

    rho_true <- 0.8
    f <- numeric(n_t); f[1] <- rnorm(1)
    for (t in 2:n_t) f[t] <- rho_true * f[t - 1] + rnorm(1, 0, sqrt(1 - rho_true^2))
    f <- f - mean(f); f <- f / sd(f)

    sigma_true <- 1.0
    alpha_true <- 1.5

    N1 <- 900L
    N2 <- 900L
    t1 <- sample.int(n_t, N1, replace = TRUE)
    t2 <- sample.int(n_t, N2, replace = TRUE)
    X1 <- cbind(1, rnorm(N1))
    X2 <- cbind(1, rnorm(N2))

    eta1 <- X1 %*% c(0.2, 0.5) + sigma_true * f[t1]
    eta2 <- X2 %*% c(-0.1, 0.3) + alpha_true * sigma_true * f[t2]
    y1 <- rnorm(N1, eta1, 0.3)
    y2 <- rnorm(N2, eta2, 0.3)

    arm1 <- list(y = y1, n_trials = rep(1L, N1), X = X1,
                 re_idx = rep(0, N1), n_re_groups = 0L, sigma_re = 1.0,
                 family = "gaussian", phi = 1.0)
    arm2 <- list(y = y2, n_trials = rep(1L, N2), X = X2,
                 re_idx = rep(0, N2), n_re_groups = 0L, sigma_re = 1.0,
                 family = "gaussian", phi = 1.0)

    ar1_block <- list(type = "ar1", n_times = n_t,
                      sigma_grid = c(0.6, 0.9, 1.2),
                      rho_grid = c(0.5, 0.7, 0.9),
                      temporal_idx = list(t1, t2))

    fit <- suppressWarnings(
        tulpa_nested_laplace_joint(
            responses = list(occ = arm1, pos = arm2),
            prior = list(ar1_block),
            copy = list(arm = "pos", block = 1L,
                        alpha_grid = c(0.9, 1.2, 1.5, 1.8)),
            control = list(max_iter = 50L, tol = 1e-6)
        )
    )

    expect_s3_class(fit, "tulpa_nested_laplace_joint_multi")
    expect_true(all(is.finite(fit$log_marginal)))
    expect_named(fit$block_moments[[1L]]$mean, c("sigma", "alpha", "rho"))

    alpha_hat <- fit$block_moments[[1L]]$mean[["alpha"]]
    expect_lt(abs(alpha_hat - alpha_true), 0.5)

    bs0  <- fit$arm_layout$block_start[1L]
    w    <- fit$weights
    cols <- bs0 + seq_len(n_t)
    f_hat <- as.numeric(crossprod(w, fit$modes[, cols, drop = FALSE]))
    expect_gt(abs(cor(f_hat, f)), 0.7)
})

test_that("joint multi-block recovers a copied IID field", {
    # gcol33/tulpa#76: copy on an unstructured (iid) block. A shared per-group
    # random effect u[g] enters arm1 at amplitude sigma and arm2 at alpha*sigma.
    set.seed(13)
    n_g <- 30L

    u <- rnorm(n_g, 0, 1); u <- u - mean(u); u <- u / sd(u)

    sigma_true <- 1.0
    alpha_true <- 0.6

    # Many observations per group so the unstructured effect is identifiable.
    reps <- 40L
    g1 <- rep(seq_len(n_g), each = reps)
    g2 <- rep(seq_len(n_g), each = reps)
    N1 <- length(g1)
    N2 <- length(g2)
    X1 <- cbind(1, rnorm(N1))
    X2 <- cbind(1, rnorm(N2))

    eta1 <- X1 %*% c(0.2, 0.4) + sigma_true * u[g1]
    eta2 <- X2 %*% c(-0.1, 0.3) + alpha_true * sigma_true * u[g2]
    y1 <- rnorm(N1, eta1, 0.3)
    y2 <- rnorm(N2, eta2, 0.3)

    arm1 <- list(y = y1, n_trials = rep(1L, N1), X = X1,
                 re_idx = rep(0, N1), n_re_groups = 0L, sigma_re = 1.0,
                 family = "gaussian", phi = 1.0)
    arm2 <- list(y = y2, n_trials = rep(1L, N2), X = X2,
                 re_idx = rep(0, N2), n_re_groups = 0L, sigma_re = 1.0,
                 family = "gaussian", phi = 1.0)

    iid_block <- list(type = "iid", n_units = n_g,
                      sigma_grid = c(0.6, 0.9, 1.2),
                      obs_idx = list(g1, g2))

    fit <- suppressWarnings(
        tulpa_nested_laplace_joint(
            responses = list(occ = arm1, pos = arm2),
            prior = list(iid_block),
            copy = list(arm = "pos", block = 1L,
                        alpha_grid = c(0.3, 0.6, 0.9, 1.2)),
            control = list(max_iter = 50L, tol = 1e-6)
        )
    )

    expect_s3_class(fit, "tulpa_nested_laplace_joint_multi")
    expect_true(all(is.finite(fit$log_marginal)))
    expect_named(fit$block_moments[[1L]]$mean, c("sigma", "alpha"))

    alpha_hat <- fit$block_moments[[1L]]$mean[["alpha"]]
    expect_lt(abs(alpha_hat - alpha_true), 0.4)

    bs0  <- fit$arm_layout$block_start[1L]
    w    <- fit$weights
    cols <- bs0 + seq_len(n_g)
    u_hat <- as.numeric(crossprod(w, fit$modes[, cols, drop = FALSE]))
    expect_gt(abs(cor(u_hat, u)), 0.7)
})

test_that("copy on an unsupported block type (lf) errors", {
    set.seed(17)
    n_s <- 10L
    adj <- .chain_adj_copy(n_s)
    N <- 40L
    s <- sample.int(n_s, N, replace = TRUE)
    X <- cbind(1, rnorm(N))
    arm <- list(y = rnorm(N), n_trials = rep(1L, N), X = X,
                re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
                family = "gaussian", phi = 1.0)
    lf_block <- list(type = "lf", n_latent = n_s, obs_idx = list(s, s))
    expect_error(
        tulpa_nested_laplace_joint(
            responses = list(a = arm, b = arm),
            prior = list(lf_block),
            copy = list(arm = "b", block = 1L, alpha_grid = c(0.5, 1.0))
        ),
        "supported on types"
    )
})
