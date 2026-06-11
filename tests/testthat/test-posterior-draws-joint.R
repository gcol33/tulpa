# Tests for tulpa_posterior_draws() -- the joint nested-Laplace outer-grid
# mixture sampler (gcol33/tulpa#44).

make_grid_adjacency <- function(nr, nc) {
    n <- nr * nc
    adj <- vector("list", n)
    idx <- function(r, c) (r - 1L) * nc + c
    for (r in seq_len(nr)) {
        for (c in seq_len(nc)) {
            nb <- integer(0)
            if (r > 1L)  nb <- c(nb, idx(r - 1L, c))
            if (r < nr)  nb <- c(nb, idx(r + 1L, c))
            if (c > 1L)  nb <- c(nb, idx(r, c - 1L))
            if (c < nc)  nb <- c(nb, idx(r, c + 1L))
            adj[[idx(r, c)]] <- sort(nb)
        }
    }
    adj
}

# Build a small two-arm ICAR joint fit (binomial occupancy + gaussian cover,
# copy coefficient on the cover arm). store_Q is on so the sampler has the
# per-grid precision. `sigma_grid` length controls the outer-grid size.
build_icar_joint_fit <- function(nr = 4L, nc = 4L, sigma_grid = c(0.5, 1.0),
                                 alpha_grid = c(0.4, 0.8), seed = 11L) {
    set.seed(seed)
    adj_list <- make_grid_adjacency(nr, nc)
    n_s <- length(adj_list)
    n_neighbors <- vapply(adj_list, length, integer(1))
    adj_row_ptr <- c(0L, cumsum(n_neighbors))
    adj_col_idx <- unlist(adj_list) - 1L

    rr <- ((seq_len(n_s) - 1L) %/% nc) + 1L
    cc <- ((seq_len(n_s) - 1L) %% nc) + 1L
    f_true <- scale(sin(rr / 2) + cos(cc / 3))[, 1] * 0.8

    X1 <- cbind(1, rnorm(n_s))
    X2 <- cbind(1, rnorm(n_s))
    eta1 <- as.numeric(X1 %*% c(0.2, 0.5)) + f_true
    y1 <- rbinom(n_s, 1, plogis(eta1))
    eta2 <- as.numeric(X2 %*% c(-0.3, 0.8)) + 0.6 * f_true
    y2 <- rnorm(n_s, eta2, 0.5)

    responses <- list(
        occ = list(y = y1, n_trials = rep(1L, n_s), X = X1,
                   spatial_idx = seq_len(n_s), family = "binomial"),
        cover = list(y = y2, n_trials = rep(1L, n_s), X = X2,
                     spatial_idx = seq_len(n_s), family = "gaussian",
                     field_coef = list(name = "alpha", grid = alpha_grid))
    )
    prior <- list(type = "icar", n_spatial_units = n_s,
                  adj_row_ptr = adj_row_ptr, adj_col_idx = adj_col_idx,
                  n_neighbors = n_neighbors, sigma_grid = sigma_grid)

    fit <- tulpa_nested_laplace_joint(
        responses = responses, prior = prior,
        control = list(store_Q = TRUE, diagnose_k = FALSE)
    )
    fit$.n_s <- n_s
    fit
}

test_that("store_Q = FALSE gives a clear error", {
    skip_on_cran()
    skip_if_fast()
    set.seed(1)
    adj_list <- make_grid_adjacency(3L, 3L)
    n_s <- length(adj_list)
    nb <- vapply(adj_list, length, integer(1))
    X <- cbind(1, rnorm(n_s))
    y <- rbinom(n_s, 1, 0.5)
    fit <- tulpa_nested_laplace_joint(
        responses = list(a = list(y = y, n_trials = rep(1L, n_s), X = X,
                                  spatial_idx = seq_len(n_s),
                                  family = "binomial")),
        prior = list(type = "icar", n_spatial_units = n_s,
                     adj_row_ptr = c(0L, cumsum(nb)),
                     adj_col_idx = unlist(adj_list) - 1L,
                     n_neighbors = nb, sigma_grid = c(0.5, 1.0)),
        control = list(store_Q = FALSE, diagnose_k = FALSE)
    )
    expect_error(tulpa_posterior_draws(fit, n = 10),
                 "store_Q", ignore.case = TRUE)
})

test_that("mixture draws reproduce the law-of-total-covariance mean + Sigma", {
    skip_on_cran()
    skip_if_fast()
    fit <- build_icar_joint_fit()
    # focus on the fixed effects + a few field cells (a representative idx)
    layout <- fit$arm_layout
    n_s <- fit$.n_s
    beta_idx  <- seq_len(sum(layout$p))                 # 4 betas (2 per arm)
    field_idx <- layout$phi_start + seq_len(n_s)        # 1-based field block
    idx <- c(beta_idx, field_idx[1:3])

    target <- tulpa:::.joint_mixture_moments(fit, idx)
    expect_false(is.null(target))

    set.seed(99)
    draws <- tulpa_posterior_draws(fit, idx = idx, n = 40000L)
    expect_equal(dim(draws), c(40000L, length(idx)))
    expect_identical(attr(draws, "draws_kind"), "iid")

    mc_mean <- colMeans(draws)
    mc_cov  <- stats::cov(draws)

    # MC error ~ sd / sqrt(n); use a tolerance generous enough for 4e4 draws.
    expect_equal(unname(mc_mean), unname(target$mean), tolerance = 0.03)
    # compare covariance entrywise on a robust scale
    sc <- sqrt(outer(diag(target$Sigma), diag(target$Sigma)))
    expect_lt(max(abs(mc_cov - target$Sigma) / sc), 0.06)
})

test_that("constrained field draws satisfy the sum-to-zero constraint", {
    skip_on_cran()
    skip_if_fast()
    # single-cell grid (sigma length 1, no copy) => degenerate ESS = 1 mixture,
    # which isolates the kriging constraint from between-cell mode spread.
    set.seed(7)
    adj_list <- make_grid_adjacency(4L, 4L)
    n_s <- length(adj_list)
    nb <- vapply(adj_list, length, integer(1))
    X <- cbind(1, rnorm(n_s))
    eta <- as.numeric(X %*% c(0.1, 0.4)) + scale(seq_len(n_s))[, 1] * 0.5
    y <- rbinom(n_s, 1, plogis(eta))
    fit <- tulpa_nested_laplace_joint(
        responses = list(a = list(y = y, n_trials = rep(1L, n_s), X = X,
                                  spatial_idx = seq_len(n_s),
                                  family = "binomial")),
        prior = list(type = "icar", n_spatial_units = n_s,
                     adj_row_ptr = c(0L, cumsum(nb)),
                     adj_col_idx = unlist(adj_list) - 1L,
                     n_neighbors = nb, sigma_grid = 1.0),
        control = list(store_Q = TRUE, diagnose_k = FALSE)
    )
    expect_equal(length(fit$weights), 1L)

    field_idx <- fit$arm_layout$phi_start + seq_len(n_s)
    set.seed(3)
    draws <- tulpa_posterior_draws(fit, idx = field_idx, n = 2000L)
    field_sums <- rowSums(draws)
    # the constant-field direction carries no posterior variance under the
    # sum-to-zero constraint, and the mode is centred, so every draw's field
    # sums to ~0 with negligible spread.
    expect_lt(stats::sd(field_sums), 1e-6)
    expect_lt(max(abs(field_sums)), 1e-4)
})

test_that("degenerate single-cell grid draws N(m_1, V_1)", {
    skip_on_cran()
    skip_if_fast()
    fit <- build_icar_joint_fit(sigma_grid = 1.0, alpha_grid = 0.6)
    expect_equal(length(fit$weights), 1L)

    layout <- fit$arm_layout
    idx <- seq_len(sum(layout$p))   # the four fixed effects
    target <- tulpa:::.joint_mixture_moments(fit, idx)

    set.seed(42)
    draws <- tulpa_posterior_draws(fit, idx = idx, n = 30000L)
    # single cell => mean equals that cell's mode[idx]
    expect_equal(unname(colMeans(draws)), unname(fit$modes[1, idx]),
                 tolerance = 0.03)
    sc <- sqrt(outer(diag(target$Sigma), diag(target$Sigma)))
    expect_lt(max(abs(stats::cov(draws) - target$Sigma) / sc), 0.06)
})

test_that("idx = NULL returns the full latent vector", {
    skip_on_cran()
    skip_if_fast()
    fit <- build_icar_joint_fit()
    draws <- tulpa_posterior_draws(fit, n = 50L)
    expect_equal(ncol(draws), fit$Q_csc_n)
    expect_equal(colnames(draws)[1], "x1")
})

test_that("BYM2 draws satisfy both phi and theta sum-to-zero constraints", {
    skip_on_cran()
    skip_if_fast()
    set.seed(21)
    adj_list <- make_grid_adjacency(5L, 5L)
    n_s <- length(adj_list)
    nb <- vapply(adj_list, length, integer(1))
    X <- cbind(1, rnorm(n_s))
    f_true <- scale(sin(seq_len(n_s) / 3))[, 1]
    eta <- as.numeric(X %*% c(0.1, 0.4)) + f_true
    y <- rbinom(n_s, 1, plogis(eta))
    fit <- tulpa_nested_laplace_joint(
        responses = list(a = list(y = y, n_trials = rep(1L, n_s), X = X,
                                  spatial_idx = seq_len(n_s),
                                  family = "binomial")),
        prior = list(type = "bym2", n_spatial_units = n_s,
                     adj_row_ptr = c(0L, cumsum(nb)),
                     adj_col_idx = unlist(adj_list) - 1L,
                     n_neighbors = nb, sigma_grid = 1.0, rho_grid = 0.5),
        control = list(store_Q = TRUE, diagnose_k = FALSE)
    )
    layout <- fit$arm_layout
    phi_idx   <- layout$phi_start + seq_len(n_s)
    theta_idx <- layout$theta_start + seq_len(n_s)
    set.seed(5)
    draws <- tulpa_posterior_draws(fit, idx = c(phi_idx, theta_idx), n = 1500L)
    phi_sums   <- rowSums(draws[, seq_len(n_s), drop = FALSE])
    theta_sums <- rowSums(draws[, n_s + seq_len(n_s), drop = FALSE])
    # both spatial sub-blocks are constrained sum-to-zero
    expect_lt(stats::sd(phi_sums), 1e-6)
    expect_lt(stats::sd(theta_sums), 1e-6)
    expect_lt(max(abs(phi_sums)), 1e-4)
    expect_lt(max(abs(theta_sums)), 1e-4)
})

test_that("multi-block joint fit samples through inherited dispatch", {
    skip_on_cran()
    skip_if_fast()
    set.seed(202)
    adj_list <- make_grid_adjacency(5L, 5L)
    n_s <- length(adj_list)
    nb <- vapply(adj_list, length, integer(1))
    X1 <- cbind(1, rnorm(n_s)); X2 <- cbind(1, rnorm(n_s))
    f_true <- scale(sin(seq_len(n_s) / 3))[, 1]
    y1 <- rbinom(n_s, 1, plogis(as.numeric(X1 %*% c(0.3, 0.5)) + f_true))
    y2 <- rnorm(n_s, as.numeric(X2 %*% c(-0.2, 0.6)) + 0.7 * f_true, 0.4)
    responses <- list(
        occ = list(y = y1, n_trials = rep(1L, n_s), X = X1, family = "binomial"),
        cov = list(y = y2, n_trials = rep(1L, n_s), X = X2, family = "gaussian")
    )
    prior <- list(
        list(type = "icar", n_spatial_units = n_s,
             adj_row_ptr = c(0L, cumsum(nb)), adj_col_idx = unlist(adj_list) - 1L,
             n_neighbors = nb,
             spatial_idx = list(seq_len(n_s), seq_len(n_s)))
    )
    fit <- tulpa_nested_laplace_joint(
        responses = responses, prior = prior,
        control = list(store_Q = TRUE, diagnose_k = FALSE)
    )
    expect_s3_class(fit, "tulpa_nested_laplace_joint_multi")

    # field block for the multi-block layout lives at field_starts[1].
    field_idx <- fit$arm_layout$field_starts[1] + seq_len(n_s)
    beta_idx  <- seq_len(sum(fit$arm_layout$p))
    idx <- c(beta_idx, field_idx)

    target <- tulpa:::.joint_mixture_moments(fit, idx)
    expect_false(is.null(target))
    set.seed(8)
    draws <- tulpa_posterior_draws(fit, idx = idx, n = 20000L)
    # field sum-to-zero through the multi-block constraint branch
    field_sums <- rowSums(draws[, length(beta_idx) + seq_len(n_s), drop = FALSE])
    expect_lt(max(abs(field_sums)), 1e-4)
    # moments match the law-of-total-covariance target
    sc <- sqrt(outer(diag(target$Sigma), diag(target$Sigma)))
    expect_lt(max(abs(stats::cov(draws) - target$Sigma) / sc), 0.08)
    expect_equal(unname(colMeans(draws)), unname(target$mean), tolerance = 0.05)
})

test_that("a non-converged (NaN log_marginal) cell does not poison the weights", {
    # A non-converged inner Newton returns NaN log_marginal. The dense-tensor
    # integration weights are read directly by tulpa_posterior_draws() and
    # theta_mean; an unguarded max(log_marginal) would propagate the NaN to
    # every weight, leaving the whole outer grid unsamplable. The finite-guarded
    # normaliser must drop the NaN cell and renormalise over the rest.
    lm <- c(-2.0, -1.0, NaN, -3.0, NA_real_, -0.5)
    w  <- tulpa:::.joint_integration_weights(lm, dnode = NULL)
    expect_length(w, length(lm))
    expect_true(all(is.finite(w)))
    expect_equal(sum(w), 1, tolerance = 1e-12)
    expect_identical(w[c(3L, 5L)], c(0, 0))      # non-finite cells carry no mass
    expect_gt(w[6L], w[1L])                       # ordering by log_marginal kept

    # All-non-finite -> all-NA (degenerate), not NaN, with a warning.
    expect_warning(w0 <- tulpa:::.joint_integration_weights(c(NaN, NA_real_)),
                   "non-finite")
    expect_true(all(is.na(w0)))
})

test_that("posterior draws survive a grid with a non-converged cell", {
    skip_on_cran()
    skip_if_fast()
    fit <- build_icar_joint_fit()
    # Inject a NaN log_marginal cell (a non-converged inner Newton) and recompute
    # the integration weights the way the fitter does: the sampler must still
    # find valid mass instead of erroring on an all-NaN weight vector.
    fit$log_marginal[1L] <- NaN
    fit$weights <- tulpa:::.joint_integration_weights(fit$log_marginal, dnode = NULL)
    expect_true(any(is.finite(fit$weights) & fit$weights > 0))
    set.seed(3)
    draws <- tulpa_posterior_draws(fit, n = 200L)
    expect_equal(nrow(draws), 200L)
    expect_true(all(is.finite(draws)))
    # the poisoned cell must never be sampled from
    expect_false(any(attr(draws, "cells") == 1L))
})
