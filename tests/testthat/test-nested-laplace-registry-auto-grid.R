# Auto mode-Hessian outer-grid recentering on tulpa_nested_laplace()
# (gcol33/tulpa#290): the joint door's placement pass laying a registry
# block's default axis (test-nested-laplace-joint-auto-grid.R).
#
# gcol33/tulpa#293: what counts as a PIN (never moved) is decided by
# `.nl_axis_is_pinned()`, not by field presence -- an axis equal to the engine's
# own default axis, or marked with `auto_grid()`, is a default whoever wrote it.
# `control$auto_recenter = FALSE` is the opt-out that holds any grid as given.

skip_on_cran()

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

    # `control$auto_recenter = FALSE` is the opt-out that holds ANY grid where it
    # is (the default axis included -- gcol33/tulpa#293: handing the engine's own
    # default back in is a default, not a pin, so it does NOT hold the grid).
    fit_fixed <- tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X,
        prior = modifyList(prior, list(tau_grid = .nl_grid_axis("gmrf_tau"))),
        family = "binomial", control = list(auto_recenter = FALSE))
    fit_auto <- tulpa_nested_laplace(y = y, n_trials = rep(1L, length(y)), X = X,
                                     prior = prior, family = "binomial")
    # Same data, same axis, arriving explicitly instead of by default.
    fit_explicit_default <- tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X,
        prior = modifyList(prior, list(tau_grid = .nl_grid_axis("gmrf_tau"))),
        family = "binomial")

    expect_identical(fit_fixed$outer_grid_placement, "fixed")
    expect_identical(fit_fixed$outer_grid_recenter_declined,
                     "auto_recenter_disabled")
    expect_identical(fit_auto$outer_grid_placement, "auto_recentered")
    expect_identical(fit_auto$outer_grid_recenter_attempts, 1L)
    expect_identical(fit_explicit_default$outer_grid_placement, "auto_recentered")
    expect_identical(fit_explicit_default$theta_grid, fit_auto$theta_grid)
    # The recentered axis brackets a smaller tau (larger field SD) than the
    # fixed default's floor of 0.3 -- the mode moved off the low-tau
    # ceiling instead of being silently truncated there.
    expect_lt(min(fit_auto$theta_grid), 0.3)
})

test_that("a recentred fit reads its k-hat once, on the grid it reports", {
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
    rec <- new.env()
    rec$grids <- list()
    trace(".joint_pareto_k", print = FALSE, where = asNamespace("tulpa"),
          tracer = bquote(assign("grids", c(get("grids", envir = .(rec)),
                                            list(as.numeric(res$theta_grid))),
                                 envir = .(rec))))
    on.exit(untrace(".joint_pareto_k", where = asNamespace("tulpa")),
            add = TRUE)
    fit <- tulpa_nested_laplace(y = y, n_trials = rep(1L, length(y)), X = X,
                                prior = prior, family = "binomial")
    expect_identical(fit$outer_grid_placement, "auto_recentered")
    # The grid the placement replaced is never scored; the reported one is.
    expect_length(rec$grids, 1L)
    expect_equal(rec$grids[[1L]], as.numeric(fit$theta_grid))
    expect_true(is.finite(fit$pareto_k))
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
    expect_identical(fit$outer_grid_recenter_declined, "axis_pinned")
    expect_null(fit$outer_grid_recenter_attempts)
    expect_identical(as.numeric(fit$theta_grid[, "b1.tau"]), c(1, 5, 10, 20, 30))
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
