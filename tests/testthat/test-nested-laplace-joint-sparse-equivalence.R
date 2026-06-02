# Sparse path numerical equivalence against the dense path
# (gcol33/tulpa: Stage 1.5b of the joint sparse rollout).
#
# The joint multi-block kernel auto-routes to a sparse Newton path when
# n_x crosses SPARSE_THRESHOLD or when any block contributes through
# non-INDEXED_SINGLE semantics. At small n_x with classic blocks
# (icar/bym2/car_proper/rw1/rw2/ar1/iid) both paths produce mathematically
# identical Hessians; force_sparse = TRUE lets us exercise the sparse code
# at small scale against the validated dense reference.
#
# Tolerance: 1e-6 on log_marginal is well within Newton-step quantization;
# the modes typically agree to <1e-7 component-wise. We assert log_marginal
# equality first because mode comparison can have spurious differences
# when convergence stops early on either path.

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

.sim_joint_small <- function(seed = 42L, N = 200L, n_s = 25L,
                              sigma = 0.6, rho = 0.7, alpha_true = 1.0,
                              tau_b = 0.5) {
    set.seed(seed)
    adj <- .chain_adj(n_s)
    # Sample a smooth field via independent draws + smoothing (good enough
    # for a numerical-equivalence test; we don't care about recovery).
    w <- as.numeric(arima.sim(n = n_s, list(ar = 0.6))) * sigma
    w <- w - mean(w)

    s_idx_1 <- sample(seq_len(n_s), N, replace = TRUE)
    s_idx_2 <- sample(seq_len(n_s), N, replace = TRUE)
    x1 <- rnorm(N); x2 <- rnorm(N)

    eta1 <- 0.3 + 0.5 * x1 + w[s_idx_1]
    eta2 <- -0.2 + 0.4 * x2 + alpha_true * w[s_idx_2]

    y1 <- rbinom(N, 1L, plogis(eta1))
    y2 <- rnorm(N, mean = eta2, sd = 0.4)

    list(
        n_s = n_s, N = N, adj = adj,
        responses = list(
            occ = list(y = as.numeric(y1), n_trials = rep(1L, N),
                       X = cbind(intercept = 1, x = x1),
                       spatial_idx = s_idx_1,
                       re_idx = rep(0L, N), n_re_groups = 0L,
                       sigma_re = 1.0,
                       family = "binomial", phi = 1.0),
            pos = list(y = as.numeric(y2), n_trials = rep(1L, N),
                       X = cbind(intercept = 1, x = x2),
                       spatial_idx = s_idx_2,
                       re_idx = rep(0L, N), n_re_groups = 0L,
                       sigma_re = 1.0,
                       family = "gaussian", phi = 0.4)
        )
    )
}

.expect_equiv_fits <- function(fit_dense, fit_sparse,
                                 lm_tol = 1e-6, mode_tol = 1e-5) {
    expect_equal(length(fit_dense$log_marginal),
                 length(fit_sparse$log_marginal))
    # Per-cell log_marginal must agree.
    expect_equal(
        as.numeric(fit_sparse$log_marginal),
        as.numeric(fit_dense$log_marginal),
        tolerance = lm_tol,
        info = "per-cell log_marginal disagreement between dense and sparse paths"
    )
    # Modes should agree elementwise when both fits stored them.
    if (!is.null(fit_dense$modes) && !is.null(fit_sparse$modes) &&
        is.matrix(fit_dense$modes) && is.matrix(fit_sparse$modes)) {
        expect_equal(dim(fit_sparse$modes), dim(fit_dense$modes))
        expect_equal(as.numeric(fit_sparse$modes),
                     as.numeric(fit_dense$modes),
                     tolerance = mode_tol,
                     info = "per-cell mode disagreement between dense and sparse paths")
    }
}

test_that("sparse path matches dense on joint BYM2 with copy", {
    sim <- .sim_joint_small(seed = 11L, alpha_true = 1.2)
    prior <- c(
        list(type = "bym2",
             sigma_grid = c(0.5, 0.6),
             rho_grid   = c(0.7)),
        sim$adj
    )
    copy_spec <- list(arm = "pos", alpha_grid = c(1.0, 1.2))

    fit_dense  <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = prior, copy = copy_spec,
        control = list(max_iter = 40L, tol = 1e-8, n_threads = 1L, verbose = FALSE,
                       force_sparse = FALSE)
    )
    fit_sparse <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = prior, copy = copy_spec,
        control = list(max_iter = 40L, tol = 1e-8, n_threads = 1L, verbose = FALSE,
                       force_sparse = TRUE)
    )
    .expect_equiv_fits(fit_dense, fit_sparse)
})

test_that("sparse path matches dense on joint ICAR (no copy)", {
    sim <- .sim_joint_small(seed = 12L, alpha_true = 1.0)
    prior <- c(
        list(type = "icar",
             sigma_grid = c(0.5, 0.7)),
        sim$adj
    )

    fit_dense  <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = prior, copy = NULL,
        control = list(max_iter = 40L, tol = 1e-8, n_threads = 1L, verbose = FALSE,
                       force_sparse = FALSE)
    )
    fit_sparse <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = prior, copy = NULL,
        control = list(max_iter = 40L, tol = 1e-8, n_threads = 1L, verbose = FALSE,
                       force_sparse = TRUE)
    )
    .expect_equiv_fits(fit_dense, fit_sparse)
})

test_that("sparse path matches dense on joint CAR_proper", {
    sim <- .sim_joint_small(seed = 13L)
    prior <- c(
        list(type = "car_proper",
             sigma_grid   = c(0.5),
             rho_car_grid = c(0.9, 0.95)),
        sim$adj
    )

    fit_dense  <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = prior, copy = NULL,
        control = list(max_iter = 40L, tol = 1e-8, n_threads = 1L, verbose = FALSE,
                       force_sparse = FALSE)
    )
    fit_sparse <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = prior, copy = NULL,
        control = list(max_iter = 40L, tol = 1e-8, n_threads = 1L, verbose = FALSE,
                       force_sparse = TRUE)
    )
    .expect_equiv_fits(fit_dense, fit_sparse)
})
