# Joint multi-block nested-Laplace tests (Phase J-B).
#
# (1) Parity: a length-1 ICAR block routed via the multi-block list-of-
#     blocks API must reproduce log_marginal cell-by-cell against the
#     single-block API (prior = list(type = "icar", ...)). Both paths
#     dispatch through the same C++ kernel
#     (cpp_nested_laplace_joint_multi) since Phase J-E folded the
#     legacy per-backend wrappers into the multi-block driver via
#     `.joint_call_kernel_via_multi`. Bit-exact gate: 1e-10 on
#     log_marginal at every outer-grid cell.
#
# (2) End-to-end: two arms (binomial occupancy + gaussian positive) with
#     BYM2 (areal habitat, copy on the gaussian arm) + AR1 (year, shared)
#     + IID (observer, shared). Runs to convergence and produces sane
#     posterior moments (block_moments populated, log_marginal finite,
#     alpha posterior in a plausible range).

source(test_path("test-sparse-cholesky.R"), local = TRUE)

# Reuse the chain adjacency helper from the single-block joint test.
.chain_adj_jm <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    n_neighbors <- vapply(nbr, length, integer(1))
    list(
        adj_row_ptr = as.integer(c(0L, cumsum(n_neighbors))),
        adj_col_idx = as.integer(unlist(nbr)) - 1L,
        n_neighbors = as.integer(n_neighbors),
        n_spatial_units = n_s
    )
}

# --------------------------------------------------------------------------- #
# (1) Parity: length-1 ICAR block via multi-block dispatch                    #
# --------------------------------------------------------------------------- #

test_that("joint multi-block (1 x ICAR copy) matches joint ICAR cell-by-cell", {
    set.seed(31)
    n_s <- 20
    N1 <- 150L
    N2 <- 120L
    adj <- .chain_adj_jm(n_s)

    # Two-arm simulation with a shared phi field.
    sigma_true <- 0.6
    alpha_true <- 1.0
    rw <- cumsum(rnorm(n_s, 0, sigma_true / sqrt(n_s)))
    phi_s <- rw - mean(rw)

    s_idx_1 <- sample.int(n_s, N1, replace = TRUE)
    s_idx_2 <- sample.int(n_s, N2, replace = TRUE)
    X1 <- cbind(1, rnorm(N1))
    X2 <- cbind(1, rnorm(N2))
    eta1 <- X1 %*% c(-0.2, 0.4) + phi_s[s_idx_1]
    eta2 <- X2 %*% c(0.1, -0.3) + alpha_true * phi_s[s_idx_2]
    y1 <- rbinom(N1, 1, plogis(eta1))
    y2 <- rnorm(N2, eta2, 0.5)

    sigma_grid <- c(0.3, 0.6, 1.0)
    alpha_grid <- c(0.5, 1.0, 1.5)

    arm_occ <- list(
        y = as.numeric(y1), n_trials = rep(1L, N1), X = X1,
        spatial_idx = s_idx_1,
        re_idx = rep(0, N1), n_re_groups = 0L, sigma_re = 1.0,
        family = "binomial", phi = 1.0
    )
    arm_pos <- list(
        y = y2, n_trials = rep(1L, N2), X = X2,
        spatial_idx = s_idx_2,
        re_idx = rep(0, N2), n_re_groups = 0L, sigma_re = 1.0,
        family = "gaussian", phi = 1.0
    )

    # Single-block path (legacy).
    prior_single <- list(
        type = "icar",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors,
        sigma_grid = sigma_grid
    )
    fit_legacy <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior = prior_single,
        copy = list(arm = "pos", alpha_grid = alpha_grid)
    )

    # Multi-block path (length-1 ICAR block, copy on the spatial block).
    prior_multi <- list(
        list(
            type = "icar",
            n_spatial_units = adj$n_spatial_units,
            adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
            n_neighbors = adj$n_neighbors,
            sigma_grid = sigma_grid,
            spatial_idx = list(s_idx_1, s_idx_2)
        )
    )
    fit_multi <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior = prior_multi,
        copy = list(block = 1, arm = "pos", alpha_grid = alpha_grid)
    )

    # Both paths build a (sigma x alpha) Cartesian grid of the same size.
    # The legacy build_grids varies sigma fastest then alpha; the multi
    # build_grids matches that ordering (expand.grid varies first arg
    # fastest). Reordering would still pass via key matching, but a direct
    # comparison should align.
    expect_equal(length(fit_legacy$log_marginal),
                 length(fit_multi$log_marginal))

    # Bit-exact log_marginal up to FP reordering noise. The two code paths
    # compose the same operations (per-arm scatter, ICAR prior at tau=1,
    # arm_scale = sigma/(alpha*sigma), no centering on theta) in the same
    # order, so the cell-by-cell difference should be at the level of
    # accumulated rounding, well under 1e-10.
    expect_lt(max(abs(fit_legacy$log_marginal - fit_multi$log_marginal)),
              1e-10)

    # Both fits should agree on the joint posterior weights.
    expect_lt(max(abs(fit_legacy$weights - fit_multi$weights)), 1e-10)

    # Per-block moments are present on the multi-block side; the legacy
    # path doesn't have block_moments (single-block keeps just theta_mean).
    expect_length(fit_multi$block_moments, 1L)
    expect_named(fit_multi$block_moments[[1]]$mean,
                 c("sigma", "alpha"))
    # alpha is a direct outer-grid axis; theta_mean carries it from both
    # paths and the two should agree at FP rounding. The multi-block path
    # prefixes per-block axes (`b<N>.alpha`); the legacy single-block path
    # uses the bare name.
    expect_true("b1.alpha" %in% names(fit_multi$theta_mean))
    expect_lt(abs(fit_legacy$theta_mean[["alpha"]] -
                  fit_multi$theta_mean[["b1.alpha"]]), 1e-8)
})

# --------------------------------------------------------------------------- #
# (2) End-to-end: BYM2 + AR1 + IID, 2 arms, copy on BYM2                       #
# --------------------------------------------------------------------------- #

test_that("joint multi-block (BYM2 copy + AR1 + IID) runs end-to-end", {
    set.seed(11)
    n_sites <- 16L
    n_years <- 6L
    n_obs   <- 12L
    N1 <- 200L
    N2 <- 160L

    # Build a 4 x 4 grid adjacency for BYM2.
    grid_adj <- function(nx, ny) {
        n <- nx * ny
        edges <- list()
        for (i in seq_len(nx)) {
            for (j in seq_len(ny)) {
                s <- (i - 1L) * ny + j
                if (i > 1L) edges[[length(edges) + 1L]] <- c(s, (i - 2L) * ny + j)
                if (i < nx) edges[[length(edges) + 1L]] <- c(s, i * ny + j)
                if (j > 1L) edges[[length(edges) + 1L]] <- c(s, (i - 1L) * ny + (j - 1L))
                if (j < ny) edges[[length(edges) + 1L]] <- c(s, (i - 1L) * ny + (j + 1L))
            }
        }
        nbr <- vector("list", n)
        for (e in edges) {
            nbr[[e[1L]]] <- c(nbr[[e[1L]]], e[2L])
        }
        n_neighbors <- vapply(nbr, length, integer(1))
        list(
            adj_row_ptr = as.integer(c(0L, cumsum(n_neighbors))),
            adj_col_idx = as.integer(unlist(nbr)) - 1L,
            n_neighbors = as.integer(n_neighbors),
            n_spatial_units = n
        )
    }
    adj <- grid_adj(4L, 4L)

    s_idx_1 <- sample.int(n_sites, N1, replace = TRUE)
    s_idx_2 <- sample.int(n_sites, N2, replace = TRUE)
    t_idx_1 <- sample.int(n_years, N1, replace = TRUE)
    t_idx_2 <- sample.int(n_years, N2, replace = TRUE)
    o_idx_1 <- sample.int(n_obs,   N1, replace = TRUE)
    o_idx_2 <- sample.int(n_obs,   N2, replace = TRUE)

    X1 <- cbind(1, rnorm(N1))
    X2 <- cbind(1, rnorm(N2))

    # Simulate from a BYM2-ish field + AR1 + IID effect.
    phi_s <- rnorm(n_sites, 0, 0.4)
    phi_s <- phi_s - mean(phi_s)
    ar_t  <- numeric(n_years)
    ar_t[1L] <- rnorm(1, 0, 0.3)
    for (t in 2:n_years) ar_t[t] <- 0.7 * ar_t[t - 1L] + rnorm(1, 0, 0.3 * sqrt(1 - 0.7^2))
    iota_o <- rnorm(n_obs, 0, 0.25)

    eta1 <- X1 %*% c(-0.2, 0.5) + 0.6 * phi_s[s_idx_1] +
            ar_t[t_idx_1] + iota_o[o_idx_1]
    eta2 <- X2 %*% c(0.1, -0.3) + 0.9 * phi_s[s_idx_2] +
            ar_t[t_idx_2] + iota_o[o_idx_2]
    y1 <- rbinom(N1, 1, plogis(eta1))
    y2 <- rnorm(N2, eta2, 0.4)

    arm_occ <- list(
        y = as.numeric(y1), n_trials = rep(1L, N1), X = X1,
        re_idx = rep(0, N1), n_re_groups = 0L, sigma_re = 1.0,
        family = "binomial", phi = 1.0
    )
    arm_pos <- list(
        y = y2, n_trials = rep(1L, N2), X = X2,
        re_idx = rep(0, N2), n_re_groups = 0L, sigma_re = 1.0,
        family = "gaussian", phi = 1.0
    )

    prior <- list(
        list(
            type = "bym2",
            n_spatial_units = adj$n_spatial_units,
            adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
            n_neighbors = adj$n_neighbors,
            scale_factor = 1.0,
            sigma_grid = c(0.4, 0.8),
            rho_grid = c(0.5, 0.85),
            spatial_idx = list(s_idx_1, s_idx_2)
        ),
        list(
            type = "ar1",
            n_times = as.integer(n_years),
            tau_grid = c(3, 8, 20),
            rho_grid = c(0.4, 0.8),
            temporal_idx = list(t_idx_1, t_idx_2)
        ),
        list(
            type = "iid",
            n_units = as.integer(n_obs),
            sigma_grid = c(0.2, 0.5),
            obs_idx = list(o_idx_1, o_idx_2)
        )
    )

    fit <- suppressWarnings(
        tulpa_nested_laplace_joint(
            responses = list(occ = arm_occ, pos = arm_pos),
            prior = prior,
            copy = list(block = 1L, arm = "pos",
                        alpha_grid = c(0.5, 1.0, 1.5)),
            max_iter = 30L, tol = 1e-5
        )
    )

    expect_s3_class(fit, "tulpa_nested_laplace_joint_multi")
    expect_true(all(is.finite(fit$log_marginal)))
    expect_length(fit$block_moments, 3L)

    # Each block reports the expected axis names (after stripping the
    # `b<N>.` prefix in .joint_posterior_moments_multi).
    expect_named(fit$block_moments[[1L]]$mean,
                 c("sigma", "alpha", "rho"))
    expect_named(fit$block_moments[[2L]]$mean, c("tau", "rho"))
    expect_named(fit$block_moments[[3L]]$mean, "sigma")

    # Posterior moments are finite and sit inside the user grid extents.
    bm1 <- fit$block_moments[[1L]]$mean
    expect_true(bm1[["sigma"]] > 0 && bm1[["sigma"]] < 1.5)
    expect_true(bm1[["alpha"]] >= 0.5 && bm1[["alpha"]] <= 1.5)
    expect_true(bm1[["rho"]]   >= 0.5 && bm1[["rho"]] <= 0.85)
    bm2 <- fit$block_moments[[2L]]$mean
    expect_true(bm2[["tau"]] >= 3 && bm2[["tau"]] <= 20)
    bm3 <- fit$block_moments[[3L]]$mean
    expect_true(bm3[["sigma"]] >= 0.2 && bm3[["sigma"]] <= 0.5)

    # alpha is a direct outer-grid axis attached on the copy block; the
    # multi-block path exposes it as `b1.alpha` in theta_mean.
    expect_true("b1.alpha" %in% names(fit$theta_mean))
    expect_true(is.finite(fit$theta_mean[["b1.alpha"]]))
    expect_true(fit$theta_mean[["b1.alpha"]] > 0)
})
