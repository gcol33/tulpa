# Parallel-vs-serial parity for the SPARSE joint nested-Laplace driver
# (gcol33/tulpa#46, lever 2). The sparse outer grid previously ran serially; it
# now honours control$n_threads_outer with per-outer-thread Hessian builders,
# Newton scratch, arm specs, index caches and DENSE_BASIS scratch. The converged
# per-cell log-marginal and modes must match the serial path.
#
# Two configurations matter:
#   * plain sparse (no phi axis) -- exercises the per-thread builder/scratch pool;
#   * sparse + phi_grid -- the phi axis rewrites the shared `arms` dispersion per
#     cell, so the per-thread specs + atomic phi snapshot are what keep concurrent
#     cells from clobbering each other's residual SD. This is the cover-hurdle's
#     actual shape and is NOT covered by the dense parallel test.

skip_on_cran()
skip_if_fast()

.spar_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

.spar_sim <- function(seed) {
    set.seed(seed)
    N <- 400L; n_s <- 50L
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    w_s <- 0.6 * (sqrt(0.7) * rnorm(n_s) + sqrt(0.3) * rnorm(n_s))
    x <- rnorm(N); Xocc <- cbind(1, x)
    eta_occ <- as.numeric(Xocc %*% c(-0.3, 0.5)) + w_s[spatial_idx]
    occur <- rbinom(N, 1, plogis(eta_occ))
    is_pos <- occur == 1L
    Xpos <- Xocc[is_pos, , drop = FALSE]; spi_pos <- spatial_idx[is_pos]
    eta_pos <- as.numeric(Xpos %*% c(0.2, -0.4)) + w_s[spi_pos]
    y_pos <- rnorm(sum(is_pos), eta_pos, 0.5)
    list(N = N, n_s = n_s, spatial_idx = as.integer(spatial_idx),
         Xocc = Xocc, occur = occur, Xpos = Xpos, y_pos = y_pos,
         spi_pos = as.integer(spi_pos))
}

.spar_fit <- function(sim, n_threads_outer, phi_grid = NULL) {
    adj <- .spar_chain_adj(sim$n_s)
    arm_occ <- list(y = as.numeric(sim$occur), n_trials = rep(1L, sim$N),
                    X = sim$Xocc, spatial_idx = sim$spatial_idx,
                    re_idx = rep(0, sim$N), n_re_groups = 0L, sigma_re = 1.0,
                    family = "binomial", phi = 1.0)
    arm_pos <- list(y = sim$y_pos, n_trials = rep(1L, length(sim$y_pos)),
                    X = sim$Xpos, spatial_idx = sim$spi_pos,
                    re_idx = rep(0, length(sim$y_pos)), n_re_groups = 0L,
                    sigma_re = 1.0, family = "gaussian", phi = 0.5)
    prior <- list(type = "bym2", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, scale_factor = 1.0,
                  sigma_grid = c(0.3, 0.6, 1.0), rho_grid = c(0.3, 0.7, 0.9))
    copy <- list(arm = "pos", alpha_grid = c(0.5, 1.0, 1.5))
    tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior = prior, copy = copy, phi_grid = phi_grid,
        control = list(n_threads = 1L, n_threads_outer = n_threads_outer,
                       force_sparse = TRUE, adaptive_grid = FALSE,
                       var_of_means_consistency = FALSE))
}

.spar_expect_equiv <- function(fit_s, fit_p) {
    expect_equal(as.numeric(fit_p$log_marginal), as.numeric(fit_s$log_marginal),
                 tolerance = 1e-5,
                 info = "sparse parallel log_marginal != serial")
    if (is.matrix(fit_s$modes) && is.matrix(fit_p$modes)) {
        expect_equal(dim(fit_p$modes), dim(fit_s$modes))
        expect_equal(as.numeric(fit_p$modes), as.numeric(fit_s$modes),
                     tolerance = 1e-4,
                     info = "sparse parallel modes != serial")
    }
}

test_that("sparse parallel path matches serial (no phi axis)", {
    skip_if_not(parallel::detectCores() >= 2L, "needs multi-core")
    n_outer <- max(2L, min(4L, parallel::detectCores() - 1L))
    for (seed in 7200L + seq_len(3L)) {
        sim <- .spar_sim(seed)
        .spar_expect_equiv(.spar_fit(sim, 1L), .spar_fit(sim, n_outer))
    }
})

test_that("sparse parallel path matches serial (phi_grid axis)", {
    skip_if_not(parallel::detectCores() >= 2L, "needs multi-core")
    n_outer <- max(2L, min(4L, parallel::detectCores() - 1L))
    phi_grid <- list(pos = c(0.35, 0.5, 0.7))
    for (seed in 7300L + seq_len(3L)) {
        sim <- .spar_sim(seed)
        .spar_expect_equiv(.spar_fit(sim, 1L, phi_grid),
                           .spar_fit(sim, n_outer, phi_grid))
    }
})

# Inner-scatter parallelism: a single outer cell forces n_outer == 1, so the
# inner Newton solve receives every thread and the per-obs scatter runs its
# parallel buffer-reduce path (N >= 1000 trips the fill-vs-reduction guard).
# This is orthogonal to the outer-grid parallelism above, which pins the
# inner solve to one thread.
test_that("inner parallel scatter matches serial on a single outer cell", {
    skip_if_not(parallel::detectCores() >= 2L, "needs multi-core")
    set.seed(515)
    N <- 4000L; n_s <- 60L
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    w_s <- 0.6 * rnorm(n_s)
    x <- rnorm(N); Xocc <- cbind(1, x)
    eta_occ <- as.numeric(Xocc %*% c(-0.3, 0.5)) + w_s[spatial_idx]
    occur <- rbinom(N, 1, plogis(eta_occ))
    is_pos <- occur == 1L
    Xpos <- Xocc[is_pos, , drop = FALSE]; spi_pos <- spatial_idx[is_pos]
    eta_pos <- as.numeric(Xpos %*% c(0.2, -0.4)) + w_s[spi_pos]
    y_pos <- rnorm(sum(is_pos), eta_pos, 0.5)
    adj <- .spar_chain_adj(n_s)

    arm_occ <- list(y = as.numeric(occur), n_trials = rep(1L, N), X = Xocc,
                    spatial_idx = spatial_idx, re_idx = rep(0, N),
                    n_re_groups = 0L, sigma_re = 1.0, family = "binomial",
                    phi = 1.0)
    arm_pos <- list(y = y_pos, n_trials = rep(1L, length(y_pos)), X = Xpos,
                    spatial_idx = spi_pos, re_idx = rep(0, length(y_pos)),
                    n_re_groups = 0L, sigma_re = 1.0, family = "gaussian",
                    phi = 0.5)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = 1.0)  # 1 cell
    copy <- list(arm = "pos", alpha_grid = 1.0)

    fit_inner <- function(nt) {
        tulpa_nested_laplace_joint(
            responses = list(occ = arm_occ, pos = arm_pos),
            prior = prior, copy = copy,
            control = list(n_threads = nt, force_sparse = TRUE,
                           adaptive_grid = FALSE))
    }
    .spar_expect_equiv(fit_inner(1L), fit_inner(8L))
})
