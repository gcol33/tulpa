# Outer-grid collapse visibility for fit_spde()'s explicit method = "grid"
# path (gcol33/tulpa#276, #290). fit_spde()'s default `control$method` is
# already "ccd" (the mode-Hessian path #289/#290 ask for); "grid" is an
# explicit opt-in whose fixed span (prior mode * 0.3 to * 3) can still rail,
# so it gets the same pareto_k_regime diagnostic every other family carries
# -- visibility, not a silent override of the user's chosen method.

test_that("fit_spde(method = 'grid') attaches pareto_k_regime", {
    skip_on_cran()
    set.seed(42)
    n_obs <- 150
    coords <- cbind(runif(n_obs), runif(n_obs))
    spec <- spatial_spde(coords)
    y <- rbinom(n_obs, 1, 0.4)
    X <- matrix(1, nrow = n_obs, ncol = 1)

    fit <- fit_spde(y, X, spec, family = "binomial", n_trials = rep(1L, n_obs),
                    control = list(method = "grid", n_grid = 5L))

    expect_true(fit$converged)
    expect_true(fit$pareto_k_regime %in% c("spread", "collapsed_interior", "collapsed_edge"))
    expect_type(fit$pareto_k_grid_edge_axes, "character")
    expect_type(fit$pareto_k_grid_edge_sides, "character")
})
