# Auto mode-Hessian outer-grid recentering, standalone registry path
# (gcol33/tulpa#290) -- the .NL_REGISTRY generalization of the joint
# driver's #289 fix (test-nested-laplace-joint-auto-grid.R). Scope: icar's
# `tau_grid` and bym2's `sigma_grid`, one recenter attempt (no PC-prior
# fallback -- see the scope note on `.nl_registry_grid_rescue()`).

.chain_adj_reg <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn),
         n_spatial_units = n_s)
}

test_that("icar auto-recenters a collapsed tau axis away from the fixed ceiling", {
    skip_on_cran()
    set.seed(11)
    n_s <- 20L; n_per <- 6L
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    base_p <- rep(0.02, n_s); base_p[1:5] <- 0.95
    y <- rbinom(length(spatial_idx), 1, base_p[spatial_idx])
    X <- cbind(1, rnorm(length(y), 0, 0.05))
    adj <- .chain_adj_reg(n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, spatial_idx = spatial_idx)

    fit_fixed <- tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X,
        prior = modifyList(prior, list(tau_grid = .default_tau_grid())),
        family = "binomial")
    fit_auto <- tulpa_nested_laplace(y = y, n_trials = rep(1L, length(y)), X = X,
                                     prior = prior, family = "binomial")

    expect_identical(fit_fixed$outer_grid_placement, "fixed")
    expect_identical(fit_auto$outer_grid_placement, "auto_recentered")
    expect_identical(fit_auto$outer_grid_recenter_attempts, 1L)
    # The recentered axis brackets a smaller tau (larger field SD) than the
    # fixed default's floor of 0.3 -- the mode moved off the low-tau
    # ceiling instead of being silently truncated there.
    expect_lt(min(fit_auto$theta_grid), 0.3)
})

test_that("an explicit tau_grid on icar is never touched by the rescue", {
    skip_on_cran()
    set.seed(11)
    n_s <- 20L; n_per <- 6L
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    base_p <- rep(0.02, n_s); base_p[1:5] <- 0.95
    y <- rbinom(length(spatial_idx), 1, base_p[spatial_idx])
    X <- cbind(1, rnorm(length(y), 0, 0.05))
    adj <- .chain_adj_reg(n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, spatial_idx = spatial_idx,
                  tau_grid = c(1, 5, 10, 20, 30))

    fit <- tulpa_nested_laplace(y = y, n_trials = rep(1L, length(y)), X = X,
                                prior = prior, family = "binomial")
    expect_identical(fit$outer_grid_placement, "fixed")
    expect_null(fit$outer_grid_recenter_attempts)
    expect_identical(fit$theta_grid, c(1, 5, 10, 20, 30))
})

test_that("a well-identified icar fit is byte-stable across repeated calls", {
    skip_on_cran()
    set.seed(42)
    n_s <- 20L; n_per <- 60L
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    rw    <- cumsum(rnorm(n_s, 0, 1.0 / sqrt(n_s)))
    phi_s <- rw - mean(rw)
    X <- cbind(1, rnorm(length(spatial_idx)))
    eta <- as.numeric(X %*% c(-0.2, 0.4)) + phi_s[spatial_idx]
    y   <- rbinom(length(eta), 1, plogis(eta))
    adj <- .chain_adj_reg(n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, spatial_idx = spatial_idx)

    fit_a <- tulpa_nested_laplace(y = y, n_trials = rep(1L, length(y)), X = X,
                                  prior = prior, family = "binomial")
    fit_b <- tulpa_nested_laplace(y = y, n_trials = rep(1L, length(y)), X = X,
                                  prior = prior, family = "binomial")

    expect_identical(fit_a$outer_grid_placement, "fixed")
    expect_null(fit_a$outer_grid_recenter_attempts)
    expect_identical(fit_a$theta_grid, fit_b$theta_grid)
    expect_identical(fit_a$weights, fit_b$weights)
    expect_identical(fit_a$log_marginal, fit_b$log_marginal)
})

test_that("bym2's rescue mechanics recentre (sigma, rho) and re-cross rho's nodes", {
    skip_on_cran()
    # Direct unit test of .nl_registry_grid_rescue()'s mechanics on bym2's
    # 2-axis (sigma, rho) grid, using a synthetic quadratic-in-log-sigma
    # target so the FD-Hessian stencil has a well-defined mode/curvature --
    # exercising the "other axis" re-crossing path without depending on
    # finding real data whose fitted posterior collapses (bym2's iid mixing
    # component makes that empirically hard to trigger; the fixed-ceiling
    # default construction -- confirmed in gcol33/tulpa#290 -- is identical
    # in code to icar/car_proper regardless).
    sg <- exp(seq(log(0.1), log(3), length.out = 5))
    rg <- c(0.2, 0.5, 0.8, 0.95)
    gr <- expand.grid(sigma = sg, rho = rg, KEEP.OUT.ATTRS = FALSE)
    theta_grid <- cbind(sigma = gr$sigma, rho = gr$rho)
    # A synthetic log-marginal peaked at (sigma = 8, rho = 0.8) -- well past
    # the sigma ceiling on one axis, sharply resolved on the other -- so
    # weight concentrates onto a single cell: the "collapsed_edge" signature
    # #289/#290 describe. (A synthetic target flat in rho instead spreads
    # weight evenly across that axis's 4 nodes and never collapses -- ess_grid
    # is a JOINT quantity over the whole grid, not per-axis.)
    true_log_sigma_mode <- log(8)
    true_rho_mode <- 0.8
    log_marg <- -0.5 * ((log(theta_grid[, "sigma"]) - true_log_sigma_mode) / 0.6)^2 +
                -0.5 * ((theta_grid[, "rho"] - true_rho_mode) / 0.05)^2
    w <- exp(log_marg - max(log_marg)); w <- w / sum(w)

    res <- list(theta_grid = theta_grid, theta_names = c("sigma", "rho"),
               weights = w, log_marginal = log_marg)
    res <- .joint_attach_pareto_k_regime(res)
    expect_identical(res$pareto_k_regime, "collapsed_edge")
    expect_true("sigma" %in% res$pareto_k_grid_edge_axes)

    synthetic_lm <- function(theta_mat) {
        -0.5 * ((log(theta_mat[, 1L]) - true_log_sigma_mode) / 0.6)^2 +
        -0.5 * ((theta_mat[, 2L] - true_rho_mode) / 0.05)^2
    }
    refit_log_marginal <- function(prior_i, theta_mat) synthetic_lm(theta_mat)
    refit <- function(prior_i) {
        new_grid <- cbind(sigma = prior_i$sigma_grid, rho = prior_i$rho_grid)
        lm_new <- synthetic_lm(new_grid)
        w_new  <- exp(lm_new - max(lm_new)); w_new <- w_new / sum(w_new)
        r <- list(theta_grid = new_grid, theta_names = c("sigma", "rho"),
                 weights = w_new, log_marginal = lm_new)
        .joint_attach_pareto_k_regime(r)
    }

    rescue <- .nl_registry_grid_rescue(res, "bym2", list(type = "bym2"),
                                       refit, refit_log_marginal)
    expect_identical(rescue$res$outer_grid_placement, "auto_recentered")
    expect_identical(rescue$res$outer_grid_recenter_attempts, 1L)
    # The recentered sigma axis brackets the true mode at 8, well past the
    # retired 3.0 ceiling.
    new_sigma <- sort(unique(rescue$prior$sigma_grid))
    expect_gt(max(new_sigma), 3.0)
    expect_lt(min(new_sigma), 8.0)
    expect_gt(max(new_sigma), 8.0 * 0.5)
    # rho's nodes are the SAME 4 default values, re-crossed with the new
    # sigma axis (not altered, not collapsed to one value).
    expect_identical(sort(unique(rescue$prior$rho_grid)), rg)
    expect_identical(rescue$res$pareto_k_regime, "spread")
})
