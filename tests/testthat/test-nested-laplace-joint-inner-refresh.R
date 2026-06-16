# Inner-Newton Cholesky factor reuse (Shamanskii / chord method),
# control$inner_refresh (gcol33/tulpa#46).
#
# Reusing the cached factor for a few inner steps only changes the PATH to the
# mode -- the gradient is exact on every step and each step is line-search
# safeguarded, so the converged mode, log_det_Q and log_marginal must match the
# re-factorize-every-step default (inner_refresh = 1L) to Newton-step
# quantization. The binomial occ arm is non-quadratic, so the inner Newton
# takes several steps per grid cell and genuinely exercises the reuse branch.
# Reuse is active only on the sparse LM path, so we force_sparse = TRUE.

skip_on_cran()

.ir_chain_adj <- function(n_s) {
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

.ir_sim_joint <- function(seed = 42L, N = 200L, n_s = 25L,
                          sigma = 0.6, alpha_true = 1.2) {
    set.seed(seed)
    adj <- .ir_chain_adj(n_s)
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
        adj = adj,
        responses = list(
            occ = list(y = as.numeric(y1), n_trials = rep(1L, N),
                       X = cbind(intercept = 1, x = x1),
                       spatial_idx = s_idx_1,
                       re_idx = rep(0L, N), n_re_groups = 0L,
                       sigma_re = 1.0, family = "binomial", phi = 1.0),
            pos = list(y = as.numeric(y2), n_trials = rep(1L, N),
                       X = cbind(intercept = 1, x = x2),
                       spatial_idx = s_idx_2,
                       re_idx = rep(0L, N), n_re_groups = 0L,
                       sigma_re = 1.0, family = "gaussian", phi = 0.4)
        )
    )
}

.ir_fit <- function(sim, prior, copy, inner_refresh) {
    tulpa_nested_laplace_joint(
        responses = sim$responses, prior = prior, copy = copy,
        control = list(max_iter = 50L, tol = 1e-8, n_threads = 1L,
                       verbose = FALSE, force_sparse = TRUE,
                       inner_refresh = inner_refresh)
    )
}

.ir_expect_equiv <- function(fit_ref, fit_reuse,
                             lm_tol = 1e-5, mode_tol = 1e-5) {
    expect_equal(
        as.numeric(fit_reuse$log_marginal),
        as.numeric(fit_ref$log_marginal),
        tolerance = lm_tol,
        info = "log_marginal changed under factor reuse"
    )
    if (is.matrix(fit_ref$modes) && is.matrix(fit_reuse$modes)) {
        expect_equal(dim(fit_reuse$modes), dim(fit_ref$modes))
        expect_equal(as.numeric(fit_reuse$modes),
                     as.numeric(fit_ref$modes),
                     tolerance = mode_tol,
                     info = "modes changed under factor reuse")
    }
}

test_that("inner_refresh validates its argument", {
    sim <- .ir_sim_joint(seed = 7L)
    prior <- c(list(type = "bym2", sigma_grid = c(0.6), rho_grid = c(0.7)),
               sim$adj)
    expect_error(
        .ir_fit(sim, prior, NULL, inner_refresh = 0L),
        "inner_refresh"
    )
    expect_error(
        .ir_fit(sim, prior, NULL, inner_refresh = c(2L, 3L)),
        "inner_refresh"
    )
})

test_that("factor reuse matches the every-step default (BYM2 + copy)", {
    sim <- .ir_sim_joint(seed = 11L, alpha_true = 1.2)
    prior <- c(list(type = "bym2",
                    sigma_grid = c(0.5, 0.6),
                    rho_grid   = c(0.7)),
               sim$adj)
    copy_spec <- list(arm = "pos", alpha_grid = c(1.0, 1.2))

    fit_ref   <- .ir_fit(sim, prior, copy_spec, inner_refresh = 1L)
    fit_reuse <- .ir_fit(sim, prior, copy_spec, inner_refresh = 3L)
    .ir_expect_equiv(fit_ref, fit_reuse)
})

test_that("factor reuse matches the every-step default (ICAR, no copy)", {
    sim <- .ir_sim_joint(seed = 12L)
    prior <- c(list(type = "icar", sigma_grid = c(0.5, 0.7)), sim$adj)

    fit_ref   <- .ir_fit(sim, prior, NULL, inner_refresh = 1L)
    fit_reuse <- .ir_fit(sim, prior, NULL, inner_refresh = 4L)
    .ir_expect_equiv(fit_ref, fit_reuse)
})

test_that("factor reuse with a grad-only-honoring cell-coupling spec matches", {
    # Exercises the grad-only reuse path through the cell-coupling branch: the
    # test Bernoulli spec skips its Hessian when out.grad_only, so reuse steps
    # build only the gradient. The converged mode must still match the
    # every-step default (the gradient is exact on every step).
    cpp_register_test_separable_bernoulli_coupling()
    set.seed(909L)
    n_s <- 14L; N <- 90L
    adj <- .ir_chain_adj(n_s)
    spatial_idx <- as.integer(rep(seq_len(n_s), length.out = N))
    eta_true <- 0.3 + rnorm(n_s, 0, 0.6)
    y <- rbinom(N, 1, plogis(eta_true[spatial_idx]))
    arm <- list(y = y, n_trials = rep(1L, N),
                X = matrix(1, nrow = N, ncol = 1L),
                spatial_idx = spatial_idx, family = "binomial", phi = 1,
                coupled = TRUE, cell_obs_map = seq_len(N))
    prior <- c(list(type = "icar", sigma_grid = c(0.4, 0.8, 1.5)), adj)

    fit_coupled <- function(inner_refresh) {
        tulpa_nested_laplace_joint(
            responses = list(occ = arm), prior = prior,
            cell_coupling = "test_separable_bernoulli",
            control = list(max_iter = 60L, tol = 1e-9, force_sparse = TRUE,
                           inner_refresh = inner_refresh))
    }
    fit_ref   <- fit_coupled(1L)
    fit_reuse <- fit_coupled(3L)
    expect_equal(as.numeric(fit_reuse$log_marginal),
                 as.numeric(fit_ref$log_marginal), tolerance = 1e-6,
                 info = "grad-only cell-coupling reuse changed log_marginal")
    expect_equal(as.numeric(fit_reuse$modes), as.numeric(fit_ref$modes),
                 tolerance = 1e-5,
                 info = "grad-only cell-coupling reuse changed modes")
})
