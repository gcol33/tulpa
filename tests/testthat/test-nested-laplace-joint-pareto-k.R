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
# (5) Bootstrap outer Pareto-k: per-arm + RNG hygiene (gcol33/tulpa#127)       #
# --------------------------------------------------------------------------- #
# The k-hat's sampling uncertainty is bootstrapped from one batch's raw
# log-ratios. The constructed single-axis res + analytic refit exercises the
# scorer directly, fast, with a known proposal-covers target; the multi-block
# fixtures exercise the per-arm bootstrap and the global-RNG hygiene.

.jpk_cover_res <- function() {
    sg <- exp(seq(-3, 3, length.out = 61))
    lw <- stats::dnorm(log(sg), 0, 1.2, log = TRUE)
    w  <- exp(lw - max(lw)); w <- w / sum(w)
    list(theta_grid = matrix(sg, ncol = 1, dimnames = list(NULL, "sigma")),
         weights = w, prior = list(type = "icar"))
}
# lt -> N(0, 0.8^2): target narrower than the grid-moment proposal (low k).
.jpk_cover_refit <- function(tm) { u <- log(tm[, "sigma"]); -0.5 * (u / 0.8)^2 - u }

test_that("the bootstrap diagnostic carries the per-arm uncertainty (multi-block by_arm)", {
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
                       diagnose_draws = 200L, k_bootstrap = 300L))
    expect_false(is.null(fit$pareto_k_by_arm))
    expect_false(is.null(fit$pareto_k_by_arm_se_boot))
    expect_false(is.null(fit$pareto_k_by_arm_ci_low))
    expect_false(is.null(fit$pareto_k_by_arm_ci_high))
    expect_false(is.null(fit$pareto_k_by_arm_band_confident))
    expect_true(is.logical(fit$pareto_k_by_arm_band_confident))
    ok <- is.finite(fit$pareto_k_by_arm)
    expect_true(any(ok))
    expect_true(all(fit$pareto_k_by_arm_ci_low[ok] <= fit$pareto_k_by_arm_ci_high[ok] + 1e-12))
    expect_true(all(fit$pareto_k_by_arm_se_boot[ok] >= 0 | is.na(fit$pareto_k_by_arm_se_boot[ok])))
})

test_that("the bootstrap diagnostic does not perturb the global RNG state", {
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
        control = list(diagnose_draws = 150L, k_bootstrap = 300L)))
    expect_identical(.Random.seed, s0)
})

# --------------------------------------------------------------------------- #
# (6) Bootstrap outer Pareto-k uncertainty (gcol33/tulpa#127)                  #
# --------------------------------------------------------------------------- #
# The k-hat is a single fixed number for a fit + proposal; its sampling
# uncertainty is bootstrapped from ONE batch's raw importance log-ratios (no new
# inner solves). diagnose_draws is the precision knob; k_tail_points an expert
# tail-threshold guard; k_conf_bands configurable band boundaries.

test_that(".psis_tail_len: automatic PSIS rule and the 20%-of-draws cap", {
    expect_identical(.psis_tail_len(500L),  68L)        # ceil(min(100, 3*sqrt(500)))
    expect_identical(.psis_tail_len(5000L), 213L)       # ceil(min(1000, 3*sqrt(5000)))
    expect_identical(.psis_tail_len(500L, 60L),  60L)   # below ceiling: honoured
    expect_identical(.psis_tail_len(500L, 300L), 100L)  # above floor(0.2*500): capped
    expect_error(.psis_tail_len(500L, 0L), "positive integer")
})

test_that(".k_band_b / .within_one_band_b use (-Inf,0.5] (0.5,0.7] (0.7,Inf)", {
    b <- c(0.5, 0.7)
    expect_identical(.k_band_b(0.30, b), 0L)            # good
    expect_identical(.k_band_b(0.50, b), 0L)            # boundary belongs to good (strict >)
    expect_identical(.k_band_b(0.60, b), 1L)            # ok
    expect_identical(.k_band_b(0.70, b), 1L)            # boundary belongs to ok
    expect_identical(.k_band_b(0.80, b), 2L)            # unreliable
    expect_identical(.k_band_b(NA_real_, b), NA_integer_)
    expect_true(.within_one_band_b(0.20, 0.45, b))      # both good
    expect_true(.within_one_band_b(0.55, 0.69, b))      # both ok
    expect_false(.within_one_band_b(0.45, 0.55, b))     # crosses 0.5
    expect_false(.within_one_band_b(0.65, 0.75, b))     # crosses 0.7
    expect_false(.within_one_band_b(NA_real_, 0.30, b)) # non-finite endpoint
})

test_that("bootstrap SE recovers the across-batch SD of the k-hat", {
    # Heavy-tailed importance ratios (proposal N(0,1), target ~ N(0,2^2)):
    # log w = 0.375 x^2, x ~ proposal -> a heavy right tail, so k > 0.
    set.seed(127)
    gen_lr <- function(N) 0.375 * rnorm(N)^2
    N  <- 400L
    # Truth: the empirical sampling SD of the single-batch k-hat over many batches.
    ks <- vapply(seq_len(150L), function(i) tulpa_psis(gen_lr(N))$pareto_k, numeric(1))
    across_sd <- stats::sd(ks[is.finite(ks)])
    # Bootstrap SE from a SINGLE batch (median of a few single-batch runs).
    ses <- vapply(seq_len(10L), function(i)
        .tulpa_psis_k_uncertainty(gen_lr(N), n_boot = 500L)$se_boot, numeric(1))
    boot_se <- stats::median(ses, na.rm = TRUE)
    # Both estimate the SAME sampling SD; agree within bootstrap noise.
    expect_gt(boot_se / across_sd, 0.6)
    expect_lt(boot_se / across_sd, 1.6)
})

test_that(".tulpa_psis_k_uncertainty returns well-formed bootstrap + formula fields", {
    set.seed(128)
    lr <- 0.375 * rnorm(400L)^2
    u  <- .tulpa_psis_k_uncertainty(lr, n_boot = 800L, conf_bands = c(0.5, 0.7))
    expect_true(is.finite(u$pareto_k))
    expect_true(is.finite(u$se_boot) && u$se_boot >= 0)
    expect_true(is.finite(u$ci_low) && is.finite(u$ci_high))
    expect_lte(u$ci_low, u$ci_high)
    expect_true(is.finite(u$se_formula) && u$se_formula > 0)
    expect_identical(u$tail_points, .psis_tail_len(length(lr)))
    expect_true(is.logical(u$band_confident))
    # k_bootstrap = 0 -> point k only, NA uncertainty.
    u0 <- .tulpa_psis_k_uncertainty(lr, n_boot = 0L)
    expect_true(is.finite(u0$pareto_k))
    expect_true(is.na(u0$se_boot))
})

# --------------------------------------------------------------------------- #
# Sample-size-dependent reliability boundary (gcol33/tulpa#128)                #
# --------------------------------------------------------------------------- #

test_that(".ps_khat_threshold is the sample-size-dependent usable boundary", {
    expect_equal(.ps_khat_threshold(100), 0.5, tolerance = 1e-9)
    expect_equal(.ps_khat_threshold(200), 1 - 1 / log10(200), tolerance = 1e-12)
    expect_equal(round(.ps_khat_threshold(200), 4), 0.5654)
    # Cap at 0.7: 1 - 1/log10(S) = 0.7 at S ~ 2154, and the cap binds above it.
    expect_equal(.ps_khat_threshold(2154), 0.7, tolerance = 1e-3)
    expect_equal(.ps_khat_threshold(10000), 0.7)
    expect_lt(.ps_khat_threshold(200), 0.7)            # tighter than the fixed cap
})

test_that(".ps_conf_bands keeps the good cut at 0.5 with a size-dependent usable cut", {
    b200 <- .ps_conf_bands(200)
    expect_length(b200, 2L)
    expect_identical(b200[1L], 0.5)
    expect_equal(b200[2L], .ps_khat_threshold(200))
    # Very small S: the usable cut drops below 0.5, so a single reliable cut.
    bsmall <- .ps_conf_bands(50)
    expect_length(bsmall, 1L)
    expect_lt(bsmall, 0.5)
})

test_that("the size-dependent boundary changes the band verdict vs fixed c(0.5, 0.7)", {
    # A bootstrap interval [0.55, 0.62] at S = 200 crosses the size-dependent usable
    # cut (~0.565), so it is NOT confidently in one band; under the fixed ok band
    # (0.5, 0.7] it would read confident. The size-dependent default is stricter.
    bands_s <- .ps_conf_bands(200)
    expect_false(.within_one_band_b(0.55, 0.62, bands_s))
    expect_true(.within_one_band_b(0.55, 0.62, c(0.5, 0.7)))
})

test_that(".tulpa_psis_k_uncertainty defaults conf_bands to the size-dependent bands", {
    set.seed(321)
    lr <- 0.5 * rnorm(300L)^2
    S  <- sum(is.finite(lr))
    set.seed(7); a <- .tulpa_psis_k_uncertainty(lr, n_boot = 400L, conf_bands = NULL)
    set.seed(7); b <- .tulpa_psis_k_uncertainty(lr, n_boot = 400L,
                                                conf_bands = .ps_conf_bands(S))
    expect_identical(a$band_confident, b$band_confident)
})

test_that(".joint_pareto_k reports bootstrap uncertainty end to end (cover fixture)", {
    res <- .jpk_cover_res()
    set.seed(415)
    kd <- .joint_pareto_k(res, .jpk_cover_refit, n_samples = 600L, k_bootstrap = 500L)
    expect_true(is.finite(kd$pareto_k))
    expect_true(is.finite(kd$pareto_k_se_boot) && kd$pareto_k_se_boot >= 0)
    expect_true(is.finite(kd$pareto_k_ci_low) && is.finite(kd$pareto_k_ci_high))
    expect_lte(kd$pareto_k_ci_low, kd$pareto_k_ci_high)
    expect_true(is.finite(kd$pareto_k_se_formula))
    expect_gte(kd$pareto_k_tail_points, 5L)
    expect_true(is.logical(kd$pareto_k_band_confident))
    expect_true(is.na(kd$pareto_k_tail_points_requested))   # automatic (NULL) request
})

test_that(".joint_pareto_k warns when k_tail_points exceeds the 20% cap", {
    res <- .jpk_cover_res()
    set.seed(99)
    # cap = floor(0.2 * 600) = 120; a request of 250 warns and the used count caps.
    expect_warning(
        kd <- .joint_pareto_k(res, .jpk_cover_refit, n_samples = 600L,
                              k_bootstrap = 200L, k_tail_points = 250L),
        "20% PSIS tail cap")
    expect_identical(kd$pareto_k_tail_points_requested, 250L)
    expect_lte(kd$pareto_k_tail_points, 120L)
})

test_that("the bootstrap k-hat is reproducible (RNG restored on exit)", {
    res <- .jpk_cover_res()
    set.seed(417)
    a <- .joint_pareto_k(res, .jpk_cover_refit, n_samples = 600L, k_bootstrap = 300L)
    # No set.seed between: .joint_pareto_k restores the entry RNG state, so the
    # second call draws the identical scoring + bootstrap samples.
    b <- .joint_pareto_k(res, .jpk_cover_refit, n_samples = 600L, k_bootstrap = 300L)
    expect_identical(a$pareto_k, b$pareto_k)
    expect_identical(a$pareto_k_se_boot, b$pareto_k_se_boot)
    expect_identical(a$pareto_k_band_confident, b$pareto_k_band_confident)
})

test_that("control validation: k_bootstrap / k_tail_points / k_conf_bands", {
    sim <- .jpk_sim(seed = 95)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.9))
    arms <- .jpk_arms_alpha(sim, c(0.5, 1.0))
    expect_error(
        tulpa_nested_laplace_joint(responses = arms, prior = prior,
            control = list(k_bootstrap = -1L)), "k_bootstrap")
    expect_error(
        tulpa_nested_laplace_joint(responses = arms, prior = prior,
            control = list(k_tail_points = 0L)), "k_tail_points")
    expect_error(
        tulpa_nested_laplace_joint(responses = arms, prior = prior,
            control = list(k_conf_bands = c(0.7, 0.5))), "k_conf_bands")
})

test_that("control end to end carries the bootstrap uncertainty fields", {
    skip_if_not_slow()
    sim <- .jpk_sim(seed = 96)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.7, 1.1))
    arms <- .jpk_arms_alpha(sim, c(0, 0.5, 1.0, 1.5))
    fit <- tulpa_nested_laplace_joint(
        responses = arms, prior = prior,
        control = list(diagnose_draws = 300L, k_bootstrap = 400L))
    expect_true(is.finite(fit$pareto_k))
    expect_true(is.finite(fit$pareto_k_se_boot))
    expect_lte(fit$pareto_k_ci_low, fit$pareto_k_ci_high)
    expect_true(is.logical(fit$pareto_k_band_confident))
    expect_identical(fit$diagnose_draws, 300L)
    expect_true(is.na(fit$diagnose_cost_ratio) || fit$diagnose_cost_ratio >= 0)
})

# --------------------------------------------------------------------------- #
# k_quality reliability front door (gcol33/tulpa#129)                          #
# --------------------------------------------------------------------------- #

test_that("control$k_quality validation rejects unknown levels", {
    sim <- .jpk_sim(seed = 95)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.9))
    arms <- .jpk_arms_alpha(sim, c(0.5, 1.0))
    expect_error(
        tulpa_nested_laplace_joint(responses = arms, prior = prior,
            control = list(k_quality = "great")), "k_quality")
})

test_that("k_quality = none disables the diagnostic and reports no verdict", {
    skip_if_not_slow()
    sim <- .jpk_sim(seed = 97)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.9))
    arms <- .jpk_arms_alpha(sim, c(0.5, 1.0))
    fit <- tulpa_nested_laplace_joint(responses = arms, prior = prior,
                                      control = list(k_quality = "none"))
    expect_identical(fit$k_quality_requested, "none")
    expect_true(is.na(fit$pareto_k))                  # diagnostic was disabled
    expect_true(is.na(fit$k_quality_reached))
    expect_match(fit$k_quality_reason, "disabled")
})

test_that("k_quality reports an honest reached / best / reason quartet", {
    skip_if_not_slow()
    sim <- .jpk_sim(seed = 96)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.7, 1.1))
    arms <- .jpk_arms_alpha(sim, c(0, 0.5, 1.0, 1.5))

    rep_fit <- tulpa_nested_laplace_joint(responses = arms, prior = prior,
                  control = list(k_quality = "report", diagnose_draws = 300L))
    expect_identical(rep_fit$k_quality_requested, "report")
    expect_true(is.na(rep_fit$k_quality_reached))     # no target band
    expect_true(rep_fit$k_quality_best %in%
                c("good", "ok", "unreliable", "uncertain"))

    # k_max_rounds = 0 keeps it single-shot (no escalation re-fits) for a fast,
    # deterministic check of the verdict.
    good_fit <- tulpa_nested_laplace_joint(responses = arms, prior = prior,
                  control = list(k_quality = "good", k_max_rounds = 0L))
    expect_identical(good_fit$k_quality_requested, "good")
    expect_true(is.logical(good_fit$k_quality_reached) &&
                !is.na(good_fit$k_quality_reached))
    expect_true(nzchar(good_fit$k_quality_reason))
    # "good" raised the default draw budget (diagnose_draws not set by the caller).
    expect_identical(good_fit$diagnose_draws, 2000L)
    expect_identical(good_fit$k_quality_rounds, 0L)     # escalation disabled
    # When the target is reached, the achieved band must be the good band.
    if (isTRUE(good_fit$k_quality_reached))
        expect_identical(good_fit$k_quality_best, "good")
})

test_that("control$k_refine / k_max_rounds validation (gcol33/tulpa#131)", {
    sim <- .jpk_sim(seed = 95)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.4, 0.9))
    arms <- .jpk_arms_alpha(sim, c(0.5, 1.0))
    expect_error(
        tulpa_nested_laplace_joint(responses = arms, prior = prior,
            control = list(k_refine = "bogus")), "k_refine")
    expect_error(
        tulpa_nested_laplace_joint(responses = arms, prior = prior,
            control = list(k_max_rounds = -1L)), "k_max_rounds")
})

test_that("k_refine = 'grid' refines the integration grid driven by bad k and lowers the outer k (gcol33/tulpa#131)", {
    skip_if_not_slow()
    # A grid that sits below the field's posterior support: the first fit does not
    # reach the good band, so the k_quality ladder must REFINE THE INTEGRATION GRID
    # (not spend more diagnostic draws) and re-integrate, driven by the bad k.
    sim <- .jpk_sim(seed = 21, sigma = 1.2)
    adj <- .jpk_chain_adj(sim$n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, sigma_grid = c(0.2, 0.35, 0.5))
    arms <- .jpk_arms_alpha(sim, c(0, 0.5, 1.0))
    ctrl <- list(k_quality = "good", diagnose_draws = 800L, k_max_rounds = 3L)

    # k_refine = "none": the band is reported but not chased -- no escalation, the
    # grid is left untouched (k_quality_rounds stays 0).
    fit0 <- tulpa_nested_laplace_joint(responses = arms, prior = prior,
              control = modifyList(ctrl, list(k_refine = "none")))
    expect_identical(fit0$k_quality_rounds, 0L)
    base_cells <- nrow(fit0$theta_grid)

    # k_refine = "grid" (the default): the bad k drives integration-grid
    # refinement, re-fitting and re-diagnosing until the band is reached.
    fit1 <- tulpa_nested_laplace_joint(responses = arms, prior = prior,
              control = modifyList(ctrl, list(k_refine = "grid")))
    expect_gte(fit1$k_quality_rounds, 1L)              # escalation ran
    expect_false(is.null(fit1$adaptive_grid_info))     # the grid was refined ...
    expect_gt(nrow(fit1$theta_grid), base_cells)       # ... by adding cells

    # diagnose_draws is identical across the two fits (set, not auto-raised), so the
    # k improvement is the refined grid's, not the diagnostic's: the separation the
    # feature is built on. Here it crosses into the good band.
    expect_lt(fit1$pareto_k, fit0$pareto_k)
    expect_true(isTRUE(fit1$k_quality_reached))
    expect_identical(fit1$k_quality_best, "good")
})
