# Outer Pareto-k-hat for the joint nested-Laplace backend (gcol33/tulpa#42).
#
# The joint hyperparameter space is heterogeneous: positive scales (sigma,
# tau, phi_*) integrate on the log scale, the BYM2 mixing weight (rho) on the
# logit scale, the copy coefficient (alpha) on the identity scale, and the
# CAR_proper correlation (rho_car) has an eigenvalue-bounded support that is
# not safely guessable -> the fit declines to quad-ESS rather than report a
# wrong k-hat. These tests cover: (1) the recovery direction (proposal-covers
# -> low k, heavier-than-proposal -> high k) through the joint transform
# driver; (2) finite k-hat on all-identifiable single-block (icar / bym2) and
# multi-block fits; (3) decline on rho_car; (4) the diagnose_k / k_samples
# gating + RNG non-perturbation.

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

.jpk_chain_adj <- function(n_s) {
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

# Two-arm fixture: binomial occupancy + gaussian positive sharing one spatial
# field, the positive arm copying it with coefficient alpha_true.
.jpk_sim <- function(N = 260, n_s = 24, sigma = 0.6, alpha_true = 1.0,
                     seed = 7) {
    set.seed(seed)
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    rw    <- cumsum(rnorm(n_s, 0, sigma / sqrt(n_s)))
    phi_s <- rw - mean(rw)
    x <- rnorm(N)
    Xocc <- cbind(1, x)
    eta_occ <- as.numeric(Xocc %*% c(-0.3, 0.5)) + phi_s[spatial_idx]
    occur   <- rbinom(N, 1, plogis(eta_occ))
    is_pos  <- occur == 1L
    Xpos    <- Xocc[is_pos, , drop = FALSE]
    spi_pos <- spatial_idx[is_pos]
    eta_pos <- as.numeric(Xpos %*% c(0.2, -0.4)) + alpha_true * phi_s[spi_pos]
    y_pos   <- rnorm(sum(is_pos), eta_pos, 0.5)
    list(N = N, n_s = n_s, spatial_idx = as.integer(spatial_idx),
         Xocc = Xocc, occur = occur,
         Xpos = Xpos, y_pos = y_pos, spi_pos = as.integer(spi_pos))
}

.jpk_arms <- function(sim) {
    list(
        occ = list(y = as.numeric(sim$occur), n_trials = rep(1L, sim$N),
                   X = sim$Xocc, spatial_idx = sim$spatial_idx,
                   re_idx = rep(0, sim$N), n_re_groups = 0L, sigma_re = 1.0,
                   family = "binomial", phi = 1.0),
        pos = list(y = sim$y_pos, n_trials = rep(1L, length(sim$y_pos)),
                   X = sim$Xpos, spatial_idx = sim$spi_pos,
                   re_idx = rep(0, length(sim$y_pos)), n_re_groups = 0L,
                   sigma_re = 1.0, family = "gaussian", phi = 0.5)
    )
}

# As .jpk_arms but declares the copy coefficient on the positive arm via its
# per-arm field_coef axis (the single-block copy spec).
.jpk_arms_alpha <- function(sim, grid) {
    a <- .jpk_arms(sim)
    a$pos$field_coef <- list(name = "alpha", grid = grid)
    a
}

# --------------------------------------------------------------------------- #
# (1) Recovery direction through the joint transform driver                   #
# --------------------------------------------------------------------------- #
# A constructed single-axis (sigma, log scale) result with an analytic
# re-evaluation closure: the effective u-space target is set by subtracting
# the log-Jacobian inside `refit`, so `.joint_pareto_k` exercises the full
# forward-transform / proposal-fit / inverse-transform-plus-Jacobian / IS-PSIS
# composition with a known answer. Mirrors test-psis.R's Gaussian-covers /
# Student-t-heavy pair.

test_that(".joint_pareto_k is low when the proposal covers the target", {
    set.seed(101)
    sg <- exp(seq(-3, 3, length.out = 61))
    sd_prop <- 1.2
    lw <- stats::dnorm(log(sg), 0, sd_prop, log = TRUE)
    w  <- exp(lw - max(lw)); w <- w / sum(w)
    res <- list(theta_grid = matrix(sg, ncol = 1,
                                    dimnames = list(NULL, "sigma")),
                weights = w, prior = list(type = "icar"))
    sd_tgt <- 0.8                                  # target narrower than proposal
    refit_cover <- function(tm) {
        u <- log(tm[, "sigma"]); -0.5 * (u / sd_tgt)^2 - u  # lt -> N(0, sd_tgt^2)
    }
    kd <- .joint_pareto_k(res, refit_cover, n_samples = 3000L)
    expect_true(is.finite(kd$pareto_k))
    expect_lt(kd$pareto_k, 0.5)
    expect_gt(kd$is_ess, 0)
})

test_that(".joint_pareto_k rises when the target is heavier than the proposal", {
    set.seed(102)
    sg <- exp(seq(-3, 3, length.out = 61))
    lw <- stats::dnorm(log(sg), 0, 1.0, log = TRUE)
    w  <- exp(lw - max(lw)); w <- w / sum(w)
    res <- list(theta_grid = matrix(sg, ncol = 1,
                                    dimnames = list(NULL, "sigma")),
                weights = w, prior = list(type = "icar"))
    refit_heavy <- function(tm) {
        u <- log(tm[, "sigma"]); stats::dt(u, df = 2, log = TRUE) - u  # lt -> t_2
    }
    kd <- .joint_pareto_k(res, refit_heavy, n_samples = 4000L)
    expect_gt(kd$pareto_k, 0.5)
})

test_that(".joint_pareto_k declines a fit carrying an unguessable axis", {
    # A car_proper-typed single block with a rho_car axis: the resolver must
    # return NULL (decline) rather than guess the eigenvalue-bounded support.
    res <- list(theta_grid = matrix(c(0.5, 1.0, 0.8, 0.9), ncol = 2,
                                    dimnames = list(NULL, c("sigma", "rho_car"))),
                weights = c(0.5, 0.5), prior = list(type = "car_proper"))
    expect_null(.joint_pareto_axis_tags(res))
    kd <- .joint_pareto_k(res, function(tm) rep(0, nrow(tm)), n_samples = 200L)
    expect_true(is.na(kd$pareto_k))
})

# --------------------------------------------------------------------------- #
# (2) Finite k-hat on all-identifiable single-block fits                      #
# --------------------------------------------------------------------------- #

test_that("joint ICAR copy fit reports a finite outer Pareto-k-hat", {
    skip_if_not_slow()
    sim <- .jpk_sim(seed = 11)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.7, 1.1))
    fit <- tulpa_nested_laplace_joint(
        responses = .jpk_arms_alpha(sim, c(0, 0.5, 1.0, 1.5)), prior = prior,
        control = list(k_samples = 150L))
    # (sigma -> log, alpha -> identity): both transformable, so k-hat is a
    # finite data-dependent reading. The value is not asserted (a small-n
    # alpha posterior is skewed, so a high reading is correct) -- pin the
    # plumbing and the ESS range, as the re_cov / single-axis tests do.
    expect_true(is.finite(fit$pareto_k))
    expect_true(is.finite(fit$pareto_k_is_ess))
    expect_gt(fit$pareto_k_is_ess, 0)
    expect_lte(fit$pareto_k_is_ess, 150 + 1e-6)
    expect_equal(fit$pareto_k_scope, "outer (hyperparameter) Gaussian proposal")
})

test_that("joint BYM2 copy fit reports a finite k-hat (log + logit + identity)", {
    skip_if_not_slow()
    sim <- .jpk_sim(seed = 21)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "bym2", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, scale_factor = 1.0,
                  sigma_grid = c(0.4, 0.8), rho_grid = c(0.3, 0.7))
    fit <- tulpa_nested_laplace_joint(
        responses = .jpk_arms_alpha(sim, c(0.5, 1.0, 1.5)), prior = prior,
        control = list(k_samples = 150L))
    # Exercises all three transforms together: sigma (log), rho (logit on the
    # BYM2 mixing weight), alpha (identity).
    expect_true(is.finite(fit$pareto_k))
    expect_gt(fit$pareto_k_is_ess, 0)
    expect_lte(fit$pareto_k_is_ess, 150 + 1e-6)
})

# --------------------------------------------------------------------------- #
# (3) Decline on CAR_proper's rho_car                                         #
# --------------------------------------------------------------------------- #

test_that("joint CAR_proper copy fit declines k-hat to quad-ESS", {
    skip_if_not_slow()
    sim <- .jpk_sim(seed = 31)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "car_proper", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors,
                  sigma_grid = c(0.5, 1.0), rho_car_grid = c(0.5, 0.9))
    fit <- tulpa_nested_laplace_joint(
        responses = .jpk_arms_alpha(sim, c(0.5, 1.0)), prior = prior,
        control = list(k_samples = 100L))
    # rho_car support is the adjacency eigenvalue interval -> declined, not
    # guessed. The quad-ESS fallback (weights) is still available downstream.
    expect_true(is.na(fit$pareto_k))
    expect_true(is.na(fit$pareto_k_is_ess))
    expect_gt(length(fit$weights), 1L)
})

# --------------------------------------------------------------------------- #
# (4) Multi-block path                                                         #
# --------------------------------------------------------------------------- #

test_that("joint multi-block (ICAR copy) reports a finite outer k-hat", {
    skip_if_not_slow()
    sim <- .jpk_sim(seed = 41)
    adj <- .jpk_chain_adj(sim$n_s)
    prior_multi <- list(list(
        type = "icar", n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.8),
        spatial_idx = list(sim$spatial_idx, sim$spi_pos)))
    fit <- tulpa_nested_laplace_joint(
        responses = .jpk_arms(sim), prior = prior_multi,
        copy = list(block = 1, arm = "pos", alpha_grid = c(0.5, 1.0, 1.5)),
        control = list(k_samples = 150L))
    expect_s3_class(fit, "tulpa_nested_laplace_joint_multi")
    expect_true(is.finite(fit$pareto_k))
    expect_gt(fit$pareto_k_is_ess, 0)
    expect_lte(fit$pareto_k_is_ess, 150 + 1e-6)
})

# --------------------------------------------------------------------------- #
# (5) diagnose_k gating + RNG non-perturbation                                #
# --------------------------------------------------------------------------- #

test_that("diagnose_k = FALSE skips k-hat and leaves modes unchanged", {
    skip_if_not_slow()
    sim <- .jpk_sim(seed = 51)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.9))
    arms <- .jpk_arms_alpha(sim, c(0.5, 1.0, 1.5))

    off <- tulpa_nested_laplace_joint(
        responses = arms, prior = prior,
        control = list(diagnose_k = FALSE))
    expect_true(is.na(off$pareto_k))                 # gated off

    on <- tulpa_nested_laplace_joint(
        responses = arms, prior = prior,
        control = list(diagnose_k = TRUE, k_samples = 120L))
    expect_true(is.finite(on$pareto_k))
    # The inner solve is deterministic and the k-hat draws restore the RNG, so
    # the integration result is bit-identical with and without the diagnostic.
    expect_equal(on$log_marginal, off$log_marginal)
    expect_equal(on$modes, off$modes)
})

test_that("a sub-floor k_samples declines to NA without paying the solves", {
    skip_if_not_slow()
    # gcol33/tulpa#51: k_samples below .PSIS_MIN_EVAL can never produce a usable
    # GPD fit, so the diagnostic must decline (NA) up front rather than run every
    # one of its inner solves and discard the result. The decline path costs no
    # more than diagnose_k = FALSE.
    sim <- .jpk_sim(seed = 71)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.9))
    arms <- .jpk_arms_alpha(sim, c(0.5, 1.0, 1.5))

    fit <- tulpa_nested_laplace_joint(
        responses = arms, prior = prior,
        control = list(diagnose_k = TRUE, k_samples = 20L))
    expect_lt(20L, tulpa:::.PSIS_MIN_EVAL)
    expect_true(is.na(fit$pareto_k))
    expect_true(is.na(fit$pareto_k_is_ess))
    # The fit itself is unaffected.
    off <- tulpa_nested_laplace_joint(
        responses = arms, prior = prior,
        control = list(diagnose_k = FALSE))
    expect_equal(fit$log_marginal, off$log_marginal)
})

test_that("warm-start + capped diagnostic iters leave the k-hat unchanged", {
    skip_if_not_slow()
    # The cost bound (gcol33/tulpa#51) warm-starts each diagnostic solve from the
    # modal latent mode and caps its iterations. Converged draws keep their exact
    # log-marginal and only negligible-weight tail draws are truncated, so the
    # k-hat is unchanged. Compare the capped path against an uncapped reference
    # built by overriding the cap to the full iteration budget.
    sim <- .jpk_sim(seed = 73)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.8, 1.2))
    arms <- .jpk_arms_alpha(sim, c(0, 0.5, 1.0, 1.5))
    ctrl <- list(diagnose_k = TRUE, k_samples = 200L, max_iter = 80L,
                 adaptive_grid = FALSE)

    capped <- tulpa_nested_laplace_joint(
        responses = arms, prior = prior, control = ctrl)

    old <- tulpa:::.K_DIAG_MAX_ITER
    assignInNamespace(".K_DIAG_MAX_ITER", 10000L, ns = "tulpa")
    on.exit(assignInNamespace(".K_DIAG_MAX_ITER", old, ns = "tulpa"), add = TRUE)
    uncapped <- tulpa_nested_laplace_joint(
        responses = arms, prior = prior, control = ctrl)

    expect_true(is.finite(capped$pareto_k))
    expect_equal(capped$pareto_k, uncapped$pareto_k, tolerance = 1e-3)
})

test_that("parallel diagnostic IS re-solves leave the k-hat unchanged (single-block)", {
    skip_on_cran()
    skip_if_not(parallel::detectCores() >= 2L,
                "needs multi-core to exercise the parallel diagnostic path")
    # The outer Pareto-k re-evaluates the inner joint marginal at each importance
    # draw. Those re-solves are independent, so the diagnostic dispatches the
    # whole batch across `n_threads_outer` (warm-started from the broadcast modal
    # mode) instead of one-at-a-time. The converged per-draw log-marginal is
    # warm-start- and thread-count-independent, so the k-hat must match the serial
    # diagnostic to numerical noise (same RNG seed for the IS draws).
    n_outer <- max(2L, min(4L, parallel::detectCores() - 1L))
    sim <- .jpk_sim(seed = 83)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.8, 1.2))
    ctrl <- function(n_to) list(diagnose_k = TRUE, k_samples = 200L,
                                n_threads = 1L, n_threads_outer = n_to,
                                adaptive_grid = FALSE,
                                var_of_means_consistency = FALSE)
    serial   <- tulpa_nested_laplace_joint(
        responses = .jpk_arms_alpha(sim, c(0, 0.5, 1.0, 1.5)), prior = prior,
        control = ctrl(1L))
    parallel <- tulpa_nested_laplace_joint(
        responses = .jpk_arms_alpha(sim, c(0, 0.5, 1.0, 1.5)), prior = prior,
        control = ctrl(n_outer))
    expect_true(is.finite(serial$pareto_k))
    expect_equal(parallel$pareto_k, serial$pareto_k, tolerance = 1e-3)
    expect_equal(parallel$pareto_k_is_ess, serial$pareto_k_is_ess,
                 tolerance = 1e-3 * max(1, abs(serial$pareto_k_is_ess)))
})

test_that("parallel diagnostic IS re-solves leave the k-hat unchanged (multi-block)", {
    skip_on_cran()
    skip_if_not(parallel::detectCores() >= 2L,
                "needs multi-core to exercise the parallel diagnostic path")
    n_outer <- max(2L, min(4L, parallel::detectCores() - 1L))
    sim <- .jpk_sim(seed = 87)
    adj <- .jpk_chain_adj(sim$n_s)
    prior_multi <- list(list(
        type = "icar", n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.8),
        spatial_idx = list(sim$spatial_idx, sim$spi_pos)))
    cp <- list(block = 1, arm = "pos", alpha_grid = c(0, 0.5, 1.0, 1.5))
    ctrl <- function(n_to) list(diagnose_k = TRUE, k_samples = 200L,
                                integration = "grid",
                                n_threads = 1L, n_threads_outer = n_to,
                                var_of_means_consistency = FALSE)
    serial   <- tulpa_nested_laplace_joint(
        responses = .jpk_arms(sim), prior = prior_multi, copy = cp,
        control = ctrl(1L))
    parallel <- tulpa_nested_laplace_joint(
        responses = .jpk_arms(sim), prior = prior_multi, copy = cp,
        control = ctrl(n_outer))
    expect_true(is.finite(serial$pareto_k))
    expect_equal(parallel$pareto_k, serial$pareto_k, tolerance = 1e-3)
    expect_equal(parallel$pareto_k_is_ess, serial$pareto_k_is_ess,
                 tolerance = 1e-3 * max(1, abs(serial$pareto_k_is_ess)))
})

test_that("diagnose_k does not perturb the global RNG state", {
    skip_if_not_slow()
    sim <- .jpk_sim(seed = 61)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.9))
    set.seed(123)
    s0 <- .Random.seed
    invisible(tulpa_nested_laplace_joint(
        responses = .jpk_arms_alpha(sim, c(0.5, 1.0, 1.5)), prior = prior,
        control = list(diagnose_k = TRUE, k_samples = 100L)))
    expect_identical(.Random.seed, s0)
})

# --------------------------------------------------------------------------- #
# (5) Batched outer Pareto-k: median + Monte Carlo range (gcol33/tulpa#123)    #
# --------------------------------------------------------------------------- #
# The k-hat is a noisy GPD-tail estimator; control$k_batches > 1 reports its
# median over independent importance batches plus the observed min/max range.
# Off (k_batches = 1) is byte-identical to the prior single-value path. The
# constructed single-axis res + analytic refit (as in section 1) exercises the
# batching driver directly, fast, with a known proposal-covers target.

.jpk_cover_res <- function() {
    sg <- exp(seq(-3, 3, length.out = 61))
    lw <- stats::dnorm(log(sg), 0, 1.2, log = TRUE)
    w  <- exp(lw - max(lw)); w <- w / sum(w)
    list(theta_grid = matrix(sg, ncol = 1, dimnames = list(NULL, "sigma")),
         weights = w, prior = list(type = "icar"))
}
# lt -> N(0, 0.8^2): target narrower than the grid-moment proposal (low k).
.jpk_cover_refit <- function(tm) { u <- log(tm[, "sigma"]); -0.5 * (u / 0.8)^2 - u }

test_that("k_batches = 1 is byte-identical to the single-value path (no range fields)", {
    res <- .jpk_cover_res()
    set.seed(404)
    a <- .joint_pareto_k(res, .jpk_cover_refit, n_samples = 2000L)
    set.seed(404)
    b <- .joint_pareto_k(res, .jpk_cover_refit, n_samples = 2000L, k_batches = 1L)
    expect_identical(a, b)
    expect_null(a$pareto_k_lo)
    expect_null(a$pareto_k_hi)
    expect_null(a$pareto_k_n_batches)
})

test_that("k_batches > 1 reports a median k-hat inside its observed range", {
    res <- .jpk_cover_res()
    set.seed(405)
    kb <- .joint_pareto_k(res, .jpk_cover_refit, n_samples = 400L, k_batches = 6L)
    expect_equal(kb$pareto_k_n_batches, 6L)
    expect_true(is.finite(kb$pareto_k))
    expect_true(is.finite(kb$is_ess) && kb$is_ess > 0)
    expect_lte(kb$pareto_k_lo, kb$pareto_k)        # band classified off the median
    expect_lte(kb$pareto_k, kb$pareto_k_hi)
    expect_lte(kb$pareto_k_lo, kb$pareto_k_hi)
    # The covering target keeps every batch usable.
    expect_lt(kb$pareto_k_hi, 0.7)
})

test_that("the batched k-hat is reproducible (seeds drawn from the restored state)", {
    res <- .jpk_cover_res()
    set.seed(406)
    k1 <- .joint_pareto_k(res, .jpk_cover_refit, n_samples = 400L, k_batches = 5L)
    # No set.seed between calls: .joint_pareto_k restores the RNG on exit, so the
    # second call sees the identical entry state and derives the same per-batch
    # seeds.
    k2 <- .joint_pareto_k(res, .jpk_cover_refit, n_samples = 400L, k_batches = 5L)
    expect_identical(k1$pareto_k, k2$pareto_k)
    expect_identical(k1$pareto_k_lo, k2$pareto_k_lo)
    expect_identical(k1$pareto_k_hi, k2$pareto_k_hi)
})

test_that("control$k_batches reports the batched median + range end to end", {
    skip_if_not_slow()
    sim <- .jpk_sim(seed = 91)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.7, 1.1))
    arms <- .jpk_arms_alpha(sim, c(0, 0.5, 1.0, 1.5))
    fit <- tulpa_nested_laplace_joint(
        responses = arms, prior = prior,
        control = list(k_samples = 120L, k_batches = 5L))
    expect_equal(fit$pareto_k_n_batches, 5L)
    expect_true(is.finite(fit$pareto_k))
    expect_lte(fit$pareto_k_lo, fit$pareto_k)
    expect_lte(fit$pareto_k, fit$pareto_k_hi)
    # The default (k_batches = 1) attaches no range fields.
    base <- tulpa_nested_laplace_joint(
        responses = arms, prior = prior, control = list(k_samples = 120L))
    expect_null(base$pareto_k_n_batches)
    expect_null(base$pareto_k_lo)
})

test_that("k_batches batches the per-arm k too (multi-block by_arm)", {
    skip_if_not_slow()
    sim <- .jpk_sim(seed = 92)
    adj <- .jpk_chain_adj(sim$n_s)
    prior_multi <- list(list(
        type = "icar", n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.8),
        spatial_idx = list(sim$spatial_idx, sim$spi_pos)))
    cp <- list(block = 1, arm = "pos", alpha_grid = c(0, 0.5, 1.0, 1.5))
    fit <- tulpa_nested_laplace_joint(
        responses = .jpk_arms(sim), prior = prior_multi, copy = cp,
        control = list(diagnose_k = "by_arm", integration = "grid",
                       k_samples = 120L, k_batches = 5L))
    expect_false(is.null(fit$pareto_k_by_arm))
    expect_false(is.null(fit$pareto_k_by_arm_lo))
    expect_false(is.null(fit$pareto_k_by_arm_hi))
    ok <- is.finite(fit$pareto_k_by_arm)
    expect_true(any(ok))
    expect_true(all(fit$pareto_k_by_arm_lo[ok] <= fit$pareto_k_by_arm[ok] + 1e-12))
    expect_true(all(fit$pareto_k_by_arm[ok] <= fit$pareto_k_by_arm_hi[ok] + 1e-12))
})

test_that("batched diagnostic does not perturb the global RNG state", {
    skip_if_not_slow()
    sim <- .jpk_sim(seed = 93)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.9))
    set.seed(321)
    s0 <- .Random.seed
    invisible(tulpa_nested_laplace_joint(
        responses = .jpk_arms_alpha(sim, c(0.5, 1.0, 1.5)), prior = prior,
        control = list(k_samples = 100L, k_batches = 4L)))
    expect_identical(.Random.seed, s0)
})

test_that("control$k_batches rejects a non-integer / sub-1 value", {
    sim <- .jpk_sim(seed = 94)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.9))
    arms <- .jpk_arms_alpha(sim, c(0.5, 1.0))
    expect_error(
        tulpa_nested_laplace_joint(responses = arms, prior = prior,
                                   control = list(k_batches = 0L)),
        "k_batches")
    expect_error(
        tulpa_nested_laplace_joint(responses = arms, prior = prior,
                                   control = list(k_batches = 2.5)),
        "k_batches")
})
