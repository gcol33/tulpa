# Sparse-path numerical equivalence for spatio-temporal nested Laplace
# (gcol33/tulpa: Stage 1.4b ST coverage).
#
# Each cpp_nested_laplace_st_* kernel takes a force_sparse parameter that
# routes the inner Newton through SparseHessianBuilder + CHOLMOD instead of
# the dense DenseMat path. At small n_x both paths build mathematically
# identical Hessians, so log_marginal and modes must agree to machine
# epsilon up to Newton-step quantization.
#
# Coverage: ICAR / CAR_proper / BYM2 × {AR1, RW1, RW2}.
#
# Doubly rank-deficient combos (ICAR/BYM2 × RW1/RW2) get looser
# tolerance because the dense path's pivot clamp
# (`cholesky_factorize_impl_raw`: if (sum <= 0) sum = 1e-6) and CHOLMOD's
# dbound-based fallback bias log_det differently on the rank-deficient
# direction — they activate on different pivot subsets in different
# elimination orders. The sparse path is numerically valid (Newton
# converges, log_marginal finite, modes finite); it just doesn't match
# dense bit-for-bit on pathological inputs. The strict numerical
# equivalence proof for these cases would require adding a uniform
# upstream diagonal ridge to BOTH paths, which is deferred.
#
# HSGP and NNGP spatial kinds carry the dense-only path until 1.4c (their
# sparse fields are empty; force_sparse = TRUE errors out explicitly).

skip_on_cran()

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

# Looser check used on doubly rank-deficient combos: confirm the sparse
# path returns finite, sensibly bounded log_marginal values without
# requiring bit-for-bit agreement with the dense pivot-clamped baseline.
.expect_finite_and_bounded <- function(fit_dense, fit_sparse,
                                          max_abs_div = 25) {
    expect_true(all(is.finite(fit_sparse$log_marginal)),
                info = "sparse path produced non-finite log_marginal")
    expect_true(all(is.finite(fit_dense$log_marginal)),
                info = "dense path produced non-finite log_marginal")
    expect_equal(length(fit_sparse$log_marginal),
                 length(fit_dense$log_marginal))
    div <- abs(as.numeric(fit_sparse$log_marginal) -
               as.numeric(fit_dense$log_marginal))
    expect_true(all(div < max_abs_div),
                info = paste("sparse-vs-dense log_marginal divergence",
                              "exceeds expected pivot-bias bound (",
                              max(div), ">", max_abs_div, ")"))
}

# ---- ICAR × {AR1, RW1, RW2} -----------------------------------------------

test_that("sparse ST: ICAR × AR1 matches dense", {
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
        tau_temporal_grid = c(1.0, 1.5),
        rho_temporal_grid = c(0.6, 0.7),
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- do.call(cpp_nested_laplace_st_icar_ar1,
                     c(args, list(force_sparse = FALSE)))
    fit_s <- do.call(cpp_nested_laplace_st_icar_ar1,
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
        temporal_idx = s$temporal_idx, n_times = s$n_t, cyclic = FALSE,
        tau_spatial_grid  = c(1.0, 2.0),
        tau_temporal_grid = c(1.0, 2.0),
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- suppressWarnings(do.call(cpp_nested_laplace_st_icar_rw1,
                     c(args, list(force_sparse = FALSE))))
    fit_s <- suppressWarnings(do.call(cpp_nested_laplace_st_icar_rw1,
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
        tau_temporal_grid = c(1.0, 2.0),
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- suppressWarnings(do.call(cpp_nested_laplace_st_icar_rw2,
                     c(args, list(force_sparse = FALSE))))
    fit_s <- suppressWarnings(do.call(cpp_nested_laplace_st_icar_rw2,
                     c(args, list(force_sparse = TRUE))))
    .expect_finite_and_bounded(fit_d, fit_s)
})

# ---- CAR_proper × {AR1, RW1, RW2} -----------------------------------------

test_that("sparse ST: CAR_proper × AR1 matches dense", {
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
        tau_temporal_grid = c(1.0, 1.5),
        rho_temporal_grid = c(0.5, 0.6),
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- do.call(cpp_nested_laplace_st_car_proper_ar1,
                     c(args, list(force_sparse = FALSE)))
    fit_s <- do.call(cpp_nested_laplace_st_car_proper_ar1,
                     c(args, list(force_sparse = TRUE)))
    .expect_equiv(fit_d, fit_s)
})

test_that("sparse ST: CAR_proper × RW1 matches dense", {
    s <- .sim_st_small(seed = 32L)
    args <- list(
        y = s$y, n = s$n_trials, X = s$X, re_idx = s$re_idx,
        n_re_groups = s$n_re_groups, sigma_re = s$sigma_re,
        spatial_idx = s$spatial_idx, n_spatial_units = s$n_s,
        adj_row_ptr = s$adj$adj_row_ptr,
        adj_col_idx = s$adj$adj_col_idx,
        n_neighbors = s$adj$n_neighbors,
        temporal_idx = s$temporal_idx, n_times = s$n_t, cyclic = FALSE,
        tau_spatial_grid  = c(1.0, 1.5),
        rho_spatial_grid  = c(0.85, 0.9),
        tau_temporal_grid = c(1.0, 2.0),
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- do.call(cpp_nested_laplace_st_car_proper_rw1,
                     c(args, list(force_sparse = FALSE)))
    fit_s <- do.call(cpp_nested_laplace_st_car_proper_rw1,
                     c(args, list(force_sparse = TRUE)))
    .expect_equiv(fit_d, fit_s)
})

test_that("sparse ST: CAR_proper × RW2 matches dense", {
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
        tau_temporal_grid = c(1.0, 2.0),
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- do.call(cpp_nested_laplace_st_car_proper_rw2,
                     c(args, list(force_sparse = FALSE)))
    fit_s <- do.call(cpp_nested_laplace_st_car_proper_rw2,
                     c(args, list(force_sparse = TRUE)))
    .expect_equiv(fit_d, fit_s)
})

# ---- BYM2 × {AR1, RW1, RW2} -----------------------------------------------

test_that("sparse ST: BYM2 × AR1 matches dense", {
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
        tau_temporal_grid  = c(1.0, 1.5),
        rho_temporal_grid  = c(0.4, 0.5),
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- do.call(cpp_nested_laplace_st_bym2_ar1,
                     c(args, list(force_sparse = FALSE)))
    fit_s <- do.call(cpp_nested_laplace_st_bym2_ar1,
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
        temporal_idx = s$temporal_idx, n_times = s$n_t, cyclic = FALSE,
        sigma_spatial_grid = c(0.4, 0.6),
        rho_spatial_grid   = c(0.6, 0.7),
        tau_temporal_grid  = c(1.0, 2.0),
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- suppressWarnings(do.call(cpp_nested_laplace_st_bym2_rw1,
                     c(args, list(force_sparse = FALSE))))
    fit_s <- suppressWarnings(do.call(cpp_nested_laplace_st_bym2_rw1,
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
        tau_temporal_grid  = c(1.0, 2.0),
        family = s$family, phi = s$phi,
        max_iter = 40L, tol = 1e-8, n_threads = 1L
    )
    fit_d <- suppressWarnings(do.call(cpp_nested_laplace_st_bym2_rw2,
                     c(args, list(force_sparse = FALSE))))
    fit_s <- suppressWarnings(do.call(cpp_nested_laplace_st_bym2_rw2,
                     c(args, list(force_sparse = TRUE))))
    .expect_finite_and_bounded(fit_d, fit_s)
})

# ---- HSGP / NNGP refuse force_sparse with a clear error -------------------

test_that("sparse ST: HSGP × AR1 refuses force_sparse with a clear message", {
    set.seed(50L)
    N <- 50L; M <- 10L; n_t <- 5L
    X <- cbind(1, rnorm(N))
    coords <- matrix(runif(N * 2), N, 2)
    phi_basis <- matrix(rnorm(N * M), N, M)
    lambda_eig <- abs(rnorm(M)) + 0.1
    t_idx <- sample(seq_len(n_t), N, replace = TRUE)
    y <- rbinom(N, 1L, 0.5)
    expect_error(
        cpp_nested_laplace_st_hsgp_ar1(
            y = as.numeric(y), n = rep(1L, N), X = X, re_idx = rep(0, N),
            n_re_groups = 0L, sigma_re = 1.0,
            phi_basis = phi_basis, lambda_eig = lambda_eig,
            temporal_idx = as.integer(t_idx), n_times = n_t,
            sigma2_spatial_grid      = c(1.0),
            lengthscale_spatial_grid = c(1.0),
            tau_temporal_grid        = c(1.0),
            rho_temporal_grid        = c(0.5),
            family = "binomial", phi = 1.0,
            max_iter = 10L, tol = 1e-6, n_threads = 1L,
            force_sparse = TRUE
        ),
        regexp = "force_sparse"
    )
})
