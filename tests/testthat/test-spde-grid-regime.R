# Outer-grid collapse visibility on fit_spde() (gcol33/tulpa#276, #290): the
# fit carries the pareto_k_regime diagnostic its joint-door outer grid reports,
# as every other family does.

skip_on_cran()

test_that("fit_spde() attaches pareto_k_regime", {
    skip_on_cran()
    set.seed(42)
    n_obs <- 150
    coords <- cbind(runif(n_obs), runif(n_obs))
    spec <- spatial_spde(coords)
    y <- rbinom(n_obs, 1, 0.4)
    X <- matrix(1, nrow = n_obs, ncol = 1)

    fit <- suppressWarnings(fit_spde(y, X, spec, family = "binomial",
                                     n_trials = rep(1L, n_obs),
                                     control = list(progress = FALSE)))

    expect_true(fit$converged)
    expect_true(fit$pareto_k_regime %in% c("spread", "collapsed_interior", "collapsed_edge"))
    expect_type(fit$pareto_k_grid_edge_axes, "character")
    expect_type(fit$pareto_k_grid_edge_sides, "character")
})
