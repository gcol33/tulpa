# The registry arm's cell solve (`.nl_solve_cells()`, the joint driver) and
# its per-row predictive variance switch. Every nested-Laplace grid fit runs on
# the joint driver; the
# SPDE grid entry is the one `cpp_nested_laplace_*` entry left and carries its
# own tests.

skip_on_cran()

test_that("registry probes omit fitted variances and preserve the density and modes", {
  skip_on_cran()
  args <- list(y = c(.2, -.4, .8), n = rep(1L, 3L), X = matrix(1, 3L, 1L),
    re_idx = rep(0, 3L), n_re_groups = 0L, sigma_re = 1,
    family = "gaussian", phi = .3, max_iter = 50L, tol = 1e-10,
    n_threads = 1L, store_Q = FALSE, hyperprior = "proper")
  block <- list(type = "iid", n_units = 3L, obs_idx = 1:3,
                sigma_grid = c(.3, .8))
  cells <- as.matrix(expand.grid(c(.3, .8), c(.3, .8)))
  on <- tulpa:::.nl_solve_cells(args, list(block, block), cells)
  off <- tulpa:::.nl_solve_cells(modifyList(args,
    list(compute_fitted_var = FALSE)), list(block, block), cells)
  expect_true("fitted_eta_var" %in% names(on))
  expect_false("fitted_eta_var" %in% names(off))
  expect_identical(off$log_marginal, on$log_marginal)
  expect_identical(off$modes, on$modes)
  expect_identical(off$fitted_eta, on$fitted_eta)
})
