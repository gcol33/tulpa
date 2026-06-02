# Sparse-path numerical equivalence for spatio-temporal nested Laplace
# (gcol33/tulpa: Stage 1.4b ST coverage).
#
# Each cpp_nested_laplace_st_<spatial> kernel takes a force_sparse parameter
# that routes the inner Newton through SparseHessianBuilder + CHOLMOD instead
# of the dense DenseMat path. At small n_x both paths build mathematically
# identical Hessians, so log_marginal and modes must agree to machine epsilon
# up to Newton-step quantization.
#
# Coverage: ICAR / CAR_proper / BYM2 x {AR1, RW1, RW2}. The temporal kernel is
# selected at call time via temporal_type ("ar1" / "rw1" / "rw2"); five typed
# spatial entries replace the former 15 spatial-x-temporal functions.
#
# Both paths factor `H + LAPLACE_UNIFORM_RIDGE * I` (1.4e: uniform
# upstream diagonal ridge), so even doubly rank-deficient combos
# (ICAR/BYM2 x RW1/RW2) factor the same matrix and agree to numerical
# tolerance -- no asymmetric dense pivot clamp, no symmetric CHOLMOD
# dbound fallback on a different pivot subset.
#
# HSGP and NNGP spatial kinds carry the dense-only path until 1.4c (their
# sparse fields are empty; force_sparse = TRUE errors out explicitly).

skip_on_cran()
skip_if_fast()

.chain_adj <- function(n_s) {
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

.sim_st_small <- function(seed = 42L, N = 200L, n_s = 20L, n_t = 8L) {
    set.seed(seed)
    adj <- .chain_adj(n_s)
    s_idx <- sample(seq_len(n_s), N, replace = TRUE)
    t_idx <- sample(seq_len(n_t), N, replace = TRUE)
    x1 <- rnorm(N)
    w_s <- as.numeric(arima.sim(n = n_s, list(ar = 0.6))) * 0.4
    w_s <- w_s - mean(w_s)
    w_t <- as.numeric(arima.sim(n = n_t, list(ar = 0.5))) * 0.3
    w_t <- w_t - mean(w_t)
    eta <- 0.2 + 0.4 * x1 + w_s[s_idx] + w_t[t_idx]
    y <- rbinom(N, 1L, plogis(eta))
    list(
        y = as.numeric(y),
        n_trials = rep(1L, N),
        X = cbind(intercept = 1, x = x1),
        re_idx = rep(0, N),
        n_re_groups = 0L,
        sigma_re = 1.0,
        spatial_idx = as.integer(s_idx),
        temporal_idx = as.integer(t_idx),
        n_s = n_s, n_t = n_t, N = N,
        family = "binomial", phi = 1.0,
        adj = adj
    )
}

.expect_equiv <- function(fit_dense, fit_sparse,
                           lm_tol = 1e-6, mode_tol = 1e-5) {
    expect_equal(
        as.numeric(fit_sparse$log_marginal),
        as.numeric(fit_dense$log_marginal),
        tolerance = lm_tol,
        info = "per-cell log_marginal disagreement between dense and sparse ST paths"
    )
    if (!is.null(fit_dense$modes) && !is.null(fit_sparse$modes) &&
        is.matrix(fit_dense$modes) && is.matrix(fit_sparse$modes)) {
        expect_equal(dim(fit_sparse$modes), dim(fit_dense$modes))
        expect_equal(as.numeric(fit_sparse$modes),
                     as.numeric(fit_dense$modes),
                     tolerance = mode_tol,
                     info = "per-cell mode disagreement between dense and sparse ST paths")
    }
}

# Rank-deficient combos now factor `H + ridge * I` on both paths (1.4e),
# so the strict per-cell equivalence holds -- looser tolerance (1e-4
# vs the PD-case 1e-6) only because doubly rank-deficient Newton steps
# accumulate floating-point noise across more elimination operations.
.expect_finite_and_bounded <- function(fit_dense, fit_sparse,
                                          lm_tol = 1e-4) {
    expect_true(all(is.finite(fit_sparse$log_marginal)),
                info = "sparse path produced non-finite log_marginal")
    expect_true(all(is.finite(fit_dense$log_marginal)),
                info = "dense path produced non-finite log_marginal")
    expect_equal(
        as.numeric(fit_sparse$log_marginal),
        as.numeric(fit_dense$log_marginal),
        tolerance = lm_tol,
        info = "per-cell log_marginal disagreement on rank-deficient combo"
    )
}

# ---- ICAR x {AR1, RW1, RW2} -----------------------------------------------

test_that("sparse ST: ICAR x AR1 matches dense", {
    s <- .sim_st_small(seed = 21L)
    args <- list(
        y = s$y, n = s$n_trials, X = s$X, re_idx = s$re_idx,
        n_re_groups = s$n_re_groups, sigma_re = s$sigma_re,
        spatial_idx = s$spatial_idx, n_spatial_units = s$n_s,
        adj_row_ptr = s$adj$adj_row_ptr,
        adj_col_idx = s$adj$adj_col_idx,
        n_neighbors = s$adj$n_neighbors,
        temporal_idx = s$temporal_idx, n_times = s$n_t,
        tau_spatial_grid  = c(1.0, 2.0),
        temporal_type     = "ar1",
        tau_temporal_grid = c(1.0, 1.5),
        rho_temporal_grid = c(0.6, 0.7),
        cyclic = FALSE,
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- do.call(cpp_nested_laplace_st_icar,
                     c(args, list(force_sparse = FALSE)))
    fit_s <- do.call(cpp_nested_laplace_st_icar,
                     c(args, list(force_sparse = TRUE)))
    .expect_equiv(fit_d, fit_s)
})

test_that("sparse ST: ICAR x RW1 finite + bounded vs dense (rank-deficient)", {
    s <- .sim_st_small(seed = 22L)
    args <- list(
        y = s$y, n = s$n_trials, X = s$X, re_idx = s$re_idx,
        n_re_groups = s$n_re_groups, sigma_re = s$sigma_re,
        spatial_idx = s$spatial_idx, n_spatial_units = s$n_s,
        adj_row_ptr = s$adj$adj_row_ptr,
        adj_col_idx = s$adj$adj_col_idx,
        n_neighbors = s$adj$n_neighbors,
        temporal_idx = s$temporal_idx, n_times = s$n_t,
        tau_spatial_grid  = c(1.0, 2.0),
        temporal_type     = "rw1",
        tau_temporal_grid = c(1.0, 2.0),
        rho_temporal_grid = NULL,
        cyclic = FALSE,
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- suppressWarnings(do.call(cpp_nested_laplace_st_icar,
                     c(args, list(force_sparse = FALSE))))
    fit_s <- suppressWarnings(do.call(cpp_nested_laplace_st_icar,
                     c(args, list(force_sparse = TRUE))))
    .expect_finite_and_bounded(fit_d, fit_s)
})

test_that("sparse ST: ICAR x RW2 finite + bounded vs dense (rank-deficient)", {
    s <- .sim_st_small(seed = 23L)
    args <- list(
        y = s$y, n = s$n_trials, X = s$X, re_idx = s$re_idx,
        n_re_groups = s$n_re_groups, sigma_re = s$sigma_re,
        spatial_idx = s$spatial_idx, n_spatial_units = s$n_s,
        adj_row_ptr = s$adj$adj_row_ptr,
        adj_col_idx = s$adj$adj_col_idx,
        n_neighbors = s$adj$n_neighbors,
        temporal_idx = s$temporal_idx, n_times = s$n_t,
        tau_spatial_grid  = c(1.0, 2.0),
        temporal_type     = "rw2",
        tau_temporal_grid = c(1.0, 2.0),
        rho_temporal_grid = NULL,
        cyclic = FALSE,
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- suppressWarnings(do.call(cpp_nested_laplace_st_icar,
                     c(args, list(force_sparse = FALSE))))
    fit_s <- suppressWarnings(do.call(cpp_nested_laplace_st_icar,
                     c(args, list(force_sparse = TRUE))))
    .expect_finite_and_bounded(fit_d, fit_s)
})

# ---- CAR_proper x {AR1, RW1, RW2} -----------------------------------------

test_that("sparse ST: CAR_proper x AR1 matches dense", {
    s <- .sim_st_small(seed = 31L)
    args <- list(
        y = s$y, n = s$n_trials, X = s$X, re_idx = s$re_idx,
        n_re_groups = s$n_re_groups, sigma_re = s$sigma_re,
        spatial_idx = s$spatial_idx, n_spatial_units = s$n_s,
        adj_row_ptr = s$adj$adj_row_ptr,
        adj_col_idx = s$adj$adj_col_idx,
        n_neighbors = s$adj$n_neighbors,
        temporal_idx = s$temporal_idx, n_times = s$n_t,
        tau_spatial_grid  = c(1.0, 1.5),
        rho_spatial_grid  = c(0.85, 0.9),
        temporal_type     = "ar1",
        tau_temporal_grid = c(1.0, 1.5),
        rho_temporal_grid = c(0.5, 0.6),
        cyclic = FALSE,
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- do.call(cpp_nested_laplace_st_car_proper,
                     c(args, list(force_sparse = FALSE)))
    fit_s <- do.call(cpp_nested_laplace_st_car_proper,
                     c(args, list(force_sparse = TRUE)))
    .expect_equiv(fit_d, fit_s)
})

test_that("sparse ST: CAR_proper x RW1 matches dense", {
    s <- .sim_st_small(seed = 32L)
    args <- list(
        y = s$y, n = s$n_trials, X = s$X, re_idx = s$re_idx,
        n_re_groups = s$n_re_groups, sigma_re = s$sigma_re,
        spatial_idx = s$spatial_idx, n_spatial_units = s$n_s,
        adj_row_ptr = s$adj$adj_row_ptr,
        adj_col_idx = s$adj$adj_col_idx,
        n_neighbors = s$adj$n_neighbors,
        temporal_idx = s$temporal_idx, n_times = s$n_t,
        tau_spatial_grid  = c(1.0, 1.5),
        rho_spatial_grid  = c(0.85, 0.9),
        temporal_type     = "rw1",
        tau_temporal_grid = c(1.0, 2.0),
        rho_temporal_grid = NULL,
        cyclic = FALSE,
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- do.call(cpp_nested_laplace_st_car_proper,
                     c(args, list(force_sparse = FALSE)))
    fit_s <- do.call(cpp_nested_laplace_st_car_proper,
                     c(args, list(force_sparse = TRUE)))
    .expect_equiv(fit_d, fit_s)
})

test_that("sparse ST: CAR_proper x RW2 matches dense", {
    s <- .sim_st_small(seed = 33L)
    args <- list(
        y = s$y, n = s$n_trials, X = s$X, re_idx = s$re_idx,
        n_re_groups = s$n_re_groups, sigma_re = s$sigma_re,
        spatial_idx = s$spatial_idx, n_spatial_units = s$n_s,
        adj_row_ptr = s$adj$adj_row_ptr,
        adj_col_idx = s$adj$adj_col_idx,
        n_neighbors = s$adj$n_neighbors,
        temporal_idx = s$temporal_idx, n_times = s$n_t,
        tau_spatial_grid  = c(1.0, 1.5),
        rho_spatial_grid  = c(0.85, 0.9),
        temporal_type     = "rw2",
        tau_temporal_grid = c(1.0, 2.0),
        rho_temporal_grid = NULL,
        cyclic = FALSE,
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- do.call(cpp_nested_laplace_st_car_proper,
                     c(args, list(force_sparse = FALSE)))
    fit_s <- do.call(cpp_nested_laplace_st_car_proper,
                     c(args, list(force_sparse = TRUE)))
    .expect_equiv(fit_d, fit_s)
})

# ---- BYM2 x {AR1, RW1, RW2} -----------------------------------------------

test_that("sparse ST: BYM2 x AR1 matches dense", {
    s <- .sim_st_small(seed = 41L)
    args <- list(
        y = s$y, n = s$n_trials, X = s$X, re_idx = s$re_idx,
        n_re_groups = s$n_re_groups, sigma_re = s$sigma_re,
        spatial_idx = s$spatial_idx, n_spatial_units = s$n_s,
        adj_row_ptr = s$adj$adj_row_ptr,
        adj_col_idx = s$adj$adj_col_idx,
        n_neighbors = s$adj$n_neighbors,
        scale_factor = 1.0,
        temporal_idx = s$temporal_idx, n_times = s$n_t,
        sigma_spatial_grid = c(0.4, 0.6),
        rho_spatial_grid   = c(0.6, 0.7),
        temporal_type      = "ar1",
        tau_temporal_grid  = c(1.0, 1.5),
        rho_temporal_grid  = c(0.4, 0.5),
        cyclic = FALSE,
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- do.call(cpp_nested_laplace_st_bym2,
                     c(args, list(force_sparse = FALSE)))
    fit_s <- do.call(cpp_nested_laplace_st_bym2,
                     c(args, list(force_sparse = TRUE)))
    .expect_equiv(fit_d, fit_s)
})

test_that("sparse ST: BYM2 x RW1 finite + bounded vs dense (rank-deficient)", {
    s <- .sim_st_small(seed = 42L)
    args <- list(
        y = s$y, n = s$n_trials, X = s$X, re_idx = s$re_idx,
        n_re_groups = s$n_re_groups, sigma_re = s$sigma_re,
        spatial_idx = s$spatial_idx, n_spatial_units = s$n_s,
        adj_row_ptr = s$adj$adj_row_ptr,
        adj_col_idx = s$adj$adj_col_idx,
        n_neighbors = s$adj$n_neighbors,
        scale_factor = 1.0,
        temporal_idx = s$temporal_idx, n_times = s$n_t,
        sigma_spatial_grid = c(0.4, 0.6),
        rho_spatial_grid   = c(0.6, 0.7),
        temporal_type      = "rw1",
        tau_temporal_grid  = c(1.0, 2.0),
        rho_temporal_grid  = NULL,
        cyclic = FALSE,
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- suppressWarnings(do.call(cpp_nested_laplace_st_bym2,
                     c(args, list(force_sparse = FALSE))))
    fit_s <- suppressWarnings(do.call(cpp_nested_laplace_st_bym2,
                     c(args, list(force_sparse = TRUE))))
    .expect_finite_and_bounded(fit_d, fit_s)
})

test_that("sparse ST: BYM2 x RW2 finite + bounded vs dense (rank-deficient)", {
    s <- .sim_st_small(seed = 43L)
    args <- list(
        y = s$y, n = s$n_trials, X = s$X, re_idx = s$re_idx,
        n_re_groups = s$n_re_groups, sigma_re = s$sigma_re,
        spatial_idx = s$spatial_idx, n_spatial_units = s$n_s,
        adj_row_ptr = s$adj$adj_row_ptr,
        adj_col_idx = s$adj$adj_col_idx,
        n_neighbors = s$adj$n_neighbors,
        scale_factor = 1.0,
        temporal_idx = s$temporal_idx, n_times = s$n_t,
        sigma_spatial_grid = c(0.4, 0.6),
        rho_spatial_grid   = c(0.6, 0.7),
        temporal_type      = "rw2",
        tau_temporal_grid  = c(1.0, 2.0),
        rho_temporal_grid  = NULL,
        cyclic = FALSE,
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- suppressWarnings(do.call(cpp_nested_laplace_st_bym2,
                     c(args, list(force_sparse = FALSE))))
    fit_s <- suppressWarnings(do.call(cpp_nested_laplace_st_bym2,
                     c(args, list(force_sparse = TRUE))))
    .expect_finite_and_bounded(fit_d, fit_s)
})

# ---- HSGP / NNGP spatio-temporal run through the joint sparse path ---------
# Phase C2 folds ST nngp/hsgp onto run_multi_block_nested_laplace_joint's sparse
# path (the same one the pure-spatial nngp/hsgp entries use), retiring the
# dense-only ST driver that previously refused force_sparse. These are smoke +
# finite checks (no dense reference to compare against -- the basis / NNGP
# precision blocks only run sparse), matching the pure-spatial nngp/hsgp tests.

test_that("ST: HSGP x AR1 runs through the joint sparse path (finite)", {
    set.seed(50L)
    N <- 60L; M <- 12L; n_t <- 5L
    X <- cbind(1, rnorm(N))
    phi_basis <- matrix(rnorm(N * M), N, M)
    lambda_eig <- sort(abs(rnorm(M)) + 0.1, decreasing = TRUE)
    t_idx <- sample(seq_len(n_t), N, replace = TRUE)
    y <- rbinom(N, 1L, 0.5)
    fit <- cpp_nested_laplace_st_hsgp(
        y = as.numeric(y), n = rep(1L, N), X = X, re_idx = rep(0, N),
        n_re_groups = 0L, sigma_re = 1.0,
        phi_basis = phi_basis, lambda_eig = lambda_eig,
        temporal_idx = as.integer(t_idx), n_times = n_t,
        sigma2_spatial_grid      = c(0.5, 1.0),
        lengthscale_spatial_grid = c(0.5, 1.0),
        temporal_type            = "ar1",
        tau_temporal_grid        = c(1.0, 1.5),
        rho_temporal_grid        = c(0.5, 0.6),
        cyclic = FALSE,
        family = "binomial", phi = 1.0,
        max_iter = 40L, tol = 1e-7, n_threads = 1L
    )
    expect_equal(length(as.numeric(fit$log_marginal)), 2L)
    expect_true(all(is.finite(fit$log_marginal)))
})

test_that("ST: NNGP x RW1 runs through the joint sparse path (finite)", {
    set.seed(51L)
    n_s <- 40L; n_t <- 5L; N <- n_s   # NNGP: one obs per spatial unit
    coords <- cbind(runif(n_s), runif(n_s))
    order_idx <- order(coords[, 1], coords[, 2])
    coords_ord <- coords[order_idx, ]
    nnk <- 8L
    nn_idx <- matrix(0L, n_s, nnk); nn_dist <- matrix(0, n_s, nnk)
    for (i in 2:n_s) {
        dd <- sqrt((coords_ord[1:(i - 1), 1] - coords_ord[i, 1])^2 +
                   (coords_ord[1:(i - 1), 2] - coords_ord[i, 2])^2)
        nc <- min(length(dd), nnk); ord <- order(dd)[1:nc]
        nn_idx[i, seq_len(nc)] <- ord; nn_dist[i, seq_len(nc)] <- dd[ord]
    }
    X <- matrix(1, N, 1)
    t_idx <- sample(seq_len(n_t), N, replace = TRUE)
    y <- rbinom(N, 1L, 0.5)
    fit <- cpp_nested_laplace_st_nngp(
        y = as.numeric(y), n = rep(1L, N), X = X, re_idx = rep(0, N),
        n_re_groups = 0L, sigma_re = 1.0,
        spatial_idx = seq_len(n_s), n_spatial = n_s,
        coords = coords_ord, nn_idx = nn_idx, nn_dist = nn_dist,
        nn_order = as.integer(order_idx - 1L), nn = nnk, cov_type = 0L,
        temporal_idx = as.integer(t_idx), n_times = n_t,
        sigma2_spatial_grid = c(0.5, 1.0),
        phi_gp_spatial_grid = c(0.3, 0.5),
        temporal_type       = "rw1",
        tau_temporal_grid   = c(1.0, 2.0),
        rho_temporal_grid   = NULL,
        cyclic = FALSE,
        family = "binomial", phi = 1.0,
        max_iter = 40L, tol = 1e-7, n_threads = 1L
    )
    expect_equal(length(as.numeric(fit$log_marginal)), 2L)
    expect_true(all(is.finite(fit$log_marginal)))
})
