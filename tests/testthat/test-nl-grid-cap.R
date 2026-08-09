# test-nl-grid-cap.R
# Caller override for the multi-block outer-grid cell ceiling
# (`control$max_grid_cells`, gcol33/tulpa#343). The default refuses a grid whose
# per-block axes multiplied out past 2048 cells; a deliberate converged tensor
# reference grid (4 axes at 7 levels is 2401 cells) raises it. Both enforcement
# sites -- the multi-block nested-Laplace dispatch and the joint multi-block
# dispatch -- go through one guard reading one resolved value.

# --------------------------------------------------------------------------- #
# (1) Resolver                                                                 #
# --------------------------------------------------------------------------- #

test_that(".nl_max_grid_cells() defaults to 2048 and reads the control knob", {
  expect_identical(.nl_max_grid_cells(), 2048)
  expect_identical(.nl_max_grid_cells(list()), 2048)
  expect_identical(.nl_max_grid_cells(list(max_grid_cells = 4096L)), 4096)
})

test_that(".nl_max_grid_cells() reads the scoped option, and control wins", {
  op <- options(tulpa.nl_max_grid_cells = 3000)
  on.exit(options(op), add = TRUE)
  expect_identical(.nl_max_grid_cells(), 3000)
  expect_identical(.nl_max_grid_cells(list()), 3000)
  expect_identical(.nl_max_grid_cells(list(max_grid_cells = 12)), 12)
})

test_that(".nl_max_grid_cells() rejects a malformed knob", {
  expect_error(.nl_max_grid_cells(list(max_grid_cells = 0)), "max_grid_cells")
  expect_error(.nl_max_grid_cells(list(max_grid_cells = -5)), "max_grid_cells")
  expect_error(.nl_max_grid_cells(list(max_grid_cells = c(10, 20))),
               "max_grid_cells")
  expect_error(.nl_max_grid_cells(list(max_grid_cells = "many")),
               "max_grid_cells")
  expect_error(.nl_max_grid_cells(list(max_grid_cells = NA_real_)),
               "max_grid_cells")
})

test_that("max_grid_cells is an accepted control knob on both front doors", {
  expect_silent(tulpa_check_control(list(max_grid_cells = 4096),
                                    .CONTROL_KEYS$nested_laplace,
                                    "tulpa_nested_laplace"))
  expect_silent(tulpa_check_control(list(max_grid_cells = 4096),
                                    .CONTROL_KEYS$nested_laplace_joint,
                                    "tulpa_nested_laplace_joint"))
  expect_silent(tulpa_check_control(list(max_grid_cells = 4096),
                                    .CONTROL_KEYS$tulpa, "tulpa"))
})

# --------------------------------------------------------------------------- #
# (2) The single guard                                                         #
# --------------------------------------------------------------------------- #

test_that("the guard refuses at the default and permits at the override", {
  # 4 axes x 7 levels = 2401: the reference grid the bare default refuses.
  expect_error(
    .nl_check_grid_cap(2401, .nl_max_grid_cells(list()), "Reduce it."),
    "2401 cells \\(hard cap 2048\\)")
  expect_silent(.nl_check_grid_cap(
    2401, .nl_max_grid_cells(list(max_grid_cells = 2401)), "Reduce it."))
  expect_silent(.nl_check_grid_cap(
    2401, .nl_max_grid_cells(list(max_grid_cells = 1e5)), "Reduce it."))
  # A lowered ceiling refuses a grid the default would have taken.
  expect_error(
    .nl_check_grid_cap(9, .nl_max_grid_cells(list(max_grid_cells = 4)),
                       "Reduce it."),
    "9 cells \\(hard cap 4\\)")
})

test_that("the cap error names the override and keeps the accidental remedy", {
  msg <- tryCatch(
    .nl_check_grid_cap(2401, 2048,
                       "Reduce per-block grid sizes or set control$integration = \"ccd\"."),
    error = conditionMessage)
  expect_true(grepl("control$max_grid_cells = 2401", msg, fixed = TRUE))
  expect_true(grepl("Reduce per-block grid sizes or set control$integration",
                    msg, fixed = TRUE))
})

# --------------------------------------------------------------------------- #
# (3) Site one: the multi-block nested-Laplace dispatch                        #
# --------------------------------------------------------------------------- #

.cap_iid_data <- function(seed = 11L, N = 40L, n_a = 5L, n_b = 4L) {
  set.seed(seed)
  ia <- rep_len(seq_len(n_a), N)
  ib <- rep_len(seq_len(n_b), N)
  x  <- rnorm(N)
  eta <- -0.2 + 0.5 * x + rnorm(n_a, 0, 0.3)[ia] + rnorm(n_b, 0, 0.3)[ib]
  list(y = rbinom(N, 1L, plogis(eta)), n = rep(1L, N), X = cbind(1, x),
       ia = as.integer(ia), ib = as.integer(ib), n_a = n_a, n_b = n_b)
}

.cap_iid_prior <- function(d, g_a, g_b) {
  list(
    list(type = "iid", obs_idx = d$ia, n_units = d$n_a, sigma_grid = g_a),
    list(type = "iid", obs_idx = d$ib, n_units = d$n_b, sigma_grid = g_b)
  )
}

test_that("multi-block dispatch refuses past the default ceiling", {
  d <- .cap_iid_data()
  # 46 x 46 = 2116 cells, just past the 2048 default. The guard fires before
  # the first inner solve, so nothing here is fitted.
  prior <- .cap_iid_prior(d, seq(0.05, 2, length.out = 46L),
                             seq(0.05, 2, length.out = 46L))
  expect_error(
    tulpa_nested_laplace(y = d$y, n_trials = d$n, X = d$X, prior = prior,
                         family = "binomial"),
    "2116 cells \\(hard cap 2048\\)")
})

test_that("control$max_grid_cells reaches the multi-block dispatch", {
  d <- .cap_iid_data()
  prior <- .cap_iid_prior(d, c(0.2, 0.5, 1.0), c(0.2, 0.5, 1.0))   # 9 cells
  expect_error(
    tulpa_nested_laplace(y = d$y, n_trials = d$n, X = d$X, prior = prior,
                         family = "binomial",
                         control = list(max_grid_cells = 4)),
    "9 cells \\(hard cap 4\\)")
})

test_that("the same grid fits under a ceiling that admits it", {
  skip_on_cran()
  d <- .cap_iid_data()
  prior <- .cap_iid_prior(d, c(0.2, 0.5, 1.0), c(0.2, 0.5, 1.0))   # 9 cells
  fit <- tulpa_nested_laplace(y = d$y, n_trials = d$n, X = d$X, prior = prior,
                              family = "binomial",
                              control = list(max_grid_cells = 9,
                                              max_iter = 30L, tol = 1e-6))
  expect_equal(length(fit$weights), 9L)
  expect_true(all(is.finite(fit$log_marginal)))
})

test_that("a raised ceiling integrates a grid past the default, end to end", {
  skip_on_cran()
  d <- .cap_iid_data()
  prior <- .cap_iid_prior(d, seq(0.05, 2, length.out = 46L),
                             seq(0.05, 2, length.out = 46L))       # 2116 cells
  fit <- suppressWarnings(tulpa_nested_laplace(
    y = d$y, n_trials = d$n, X = d$X, prior = prior, family = "binomial",
    control = list(max_grid_cells = 2116, max_iter = 30L, tol = 1e-6,
                   progress = FALSE)))
  expect_equal(length(fit$weights), 2116L)
  expect_equal(sum(fit$weights), 1)
  expect_true(all(is.finite(fit$theta_mean)))
})

# --------------------------------------------------------------------------- #
# (4) Site two: the joint multi-block dispatch                                 #
# --------------------------------------------------------------------------- #

.cap_chain_adj <- function(n) {
  nb <- lapply(seq_len(n), function(s) setdiff(c(s - 1L, s + 1L), c(0L, n + 1L)))
  list(adj_row_ptr = as.integer(c(0L, cumsum(lengths(nb)))),
       adj_col_idx = as.integer(unlist(nb) - 1L),
       n_neighbors = as.integer(lengths(nb)))
}

.cap_joint_fixture <- function(g_a, g_b, seed = 19L, n_s = 6L, N = 60L) {
  set.seed(seed)
  adjA <- .cap_chain_adj(n_s); adjB <- .cap_chain_adj(n_s)
  iA <- sample.int(n_s, N, replace = TRUE)
  iB <- sample.int(n_s, N, replace = TRUE)
  fA <- as.numeric(scale(cumsum(rnorm(n_s, 0, 0.4))))
  fB <- as.numeric(scale(cumsum(rnorm(n_s, 0, 0.4))))
  x  <- rnorm(N); X <- cbind(1, x)
  eta <- as.numeric(X %*% c(-0.2, 0.4)) + fA[iA] + fB[iB]
  y1 <- rbinom(N, 1L, plogis(eta))
  y2 <- rnorm(N, eta, 0.5)
  responses <- list(
    a = list(y = as.numeric(y1), n_trials = rep(1L, N), X = X,
             spatial_idx = as.integer(iA), re_idx = rep(0, N),
             n_re_groups = 0L, sigma_re = 1.0, family = "binomial", phi = 1.0),
    b = list(y = y2, n_trials = rep(1L, N), X = X,
             spatial_idx = as.integer(iA), re_idx = rep(0, N),
             n_re_groups = 0L, sigma_re = 1.0, family = "gaussian", phi = 0.5))
  prior <- list(
    list(type = "icar", n_spatial_units = n_s,
         adj_row_ptr = adjA$adj_row_ptr, adj_col_idx = adjA$adj_col_idx,
         n_neighbors = adjA$n_neighbors, tau_grid = g_a,
         spatial_idx = list(as.integer(iA), as.integer(iA))),
    list(type = "icar", n_spatial_units = n_s,
         adj_row_ptr = adjB$adj_row_ptr, adj_col_idx = adjB$adj_col_idx,
         n_neighbors = adjB$n_neighbors, tau_grid = g_b,
         spatial_idx = list(as.integer(iB), as.integer(iB))))
  list(responses = responses, prior = prior)
}

test_that("joint multi-block dispatch refuses past the default ceiling", {
  f <- .cap_joint_fixture(seq(0.5, 20, length.out = 46L),
                          seq(0.5, 20, length.out = 46L))        # 2116 cells
  expect_error(
    tulpa_nested_laplace_joint(responses = f$responses, prior = f$prior,
                               control = list(diagnose_k = FALSE)),
    "2116 cells \\(hard cap 2048\\)")
})

test_that("control$max_grid_cells reaches the joint multi-block dispatch", {
  f <- .cap_joint_fixture(c(0.5, 2.0), c(0.5, 2.0))               # 4 cells
  expect_error(
    tulpa_nested_laplace_joint(responses = f$responses, prior = f$prior,
                               control = list(diagnose_k = FALSE,
                                               max_grid_cells = 3)),
    "4 cells \\(hard cap 3\\)")
})

test_that("the same joint grid fits under a ceiling that admits it", {
  skip_on_cran()
  f <- .cap_joint_fixture(c(0.5, 2.0), c(0.5, 2.0))               # 4 cells
  fit <- tulpa_nested_laplace_joint(
    responses = f$responses, prior = f$prior,
    control = list(diagnose_k = FALSE, max_grid_cells = 4,
                   max_iter = 60L, tol = 1e-6))
  expect_s3_class(fit, "tulpa_nested_laplace_joint_multi")
  expect_equal(length(fit$weights), 4L)
})
