# test-nl-grid-cap.R
# A multi-block outer grid is solved at whatever size its axes multiply out to
# (gcol33/tulpa#916). A cell count does not measure the run's cost -- that is
# cells times the cost of one inner solve -- so the cost signals are the running
# grid ETA and the post-solve timing warning, and the warning names the axes
# that produced the count (gcol33/tulpa#913) including the dispersion axes
# crossed on top (gcol33/tulpa#915).

# --------------------------------------------------------------------------- #
# (1) No cell-count knob                                                       #
# --------------------------------------------------------------------------- #

test_that("max_grid_cells is refused as an unknown control knob", {
  expect_error(tulpa_check_control(list(max_grid_cells = 4096),
                                   .CONTROL_KEYS$nested_laplace,
                                   "tulpa_nested_laplace"),
               "Unknown control knob")
  expect_error(tulpa_check_control(list(max_grid_cells = 4096),
                                   .CONTROL_KEYS$nested_laplace_joint,
                                   "tulpa_nested_laplace_joint"),
               "Unknown control knob")
  expect_error(tulpa_check_control(list(max_grid_cells = 4096),
                                   .CONTROL_KEYS$tulpa, "tulpa"),
               "Unknown control knob")
})

# --------------------------------------------------------------------------- #
# (2) The layout the timing warning carries                                    #
# --------------------------------------------------------------------------- #

test_that("the layout names each block's rows and per-axis levels", {
  g1 <- as.matrix(expand.grid(sigma = c(0.1, 1, 3), alpha = c(0, 0.5, 1, 2)))
  g2 <- matrix(c(1, 2), ncol = 1, dimnames = list(NULL, "tau"))
  expect_identical(.nl_grid_layout(list(g1, g2)),
                   "b1 (12 rows: sigma 3 x alpha 4) x b2 (2 rows: tau 2)")
  expect_identical(.nl_grid_crossing(list(g1, g2)),
                   " It crosses b1 (12 rows: sigma 3 x alpha 4) x b2 (2 rows: tau 2).")
  expect_identical(.nl_grid_crossing(), "")
})

test_that("the crossing names the dispersion axes crossed on top", {
  g <- matrix(1:6, ncol = 1, dimnames = list(NULL, "tau"))
  expect_identical(.nl_grid_crossing(list(g), list(b = c(0.1, 0.4, 1, 3))),
                   " It crosses b1 (6 rows: tau 6) x phi_b 4.")
})

test_that("the timing warning carries the count, the layout and the remedy", {
  over <- .NL_MULTI_GRID_WARN_SECONDS + 1
  expect_warning(
    .nl_multi_grid_warn(over, 10000, "Reduce per-block or phi grid sizes.",
                        " It crosses b1 (50 rows: sigma 5 x alpha 10) x phi_pos 4."),
    paste0("Multi-block outer grid \\(10000 cells\\) took .* to solve\\. It crosses ",
           "b1 \\(50 rows: sigma 5 x alpha 10\\) x phi_pos 4\\. Reduce per-block ",
           "or phi grid sizes\\."))
})

# --------------------------------------------------------------------------- #
# (3) Grids past the former 2048-cell default are solved                       #
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

test_that("the multi-block dispatch integrates a 2116-cell grid end to end", {
  skip_on_cran()
  d <- .cap_iid_data()
  prior <- .cap_iid_prior(d, seq(0.05, 2, length.out = 46L),
                             seq(0.05, 2, length.out = 46L))       # 2116 cells
  fit <- suppressWarnings(tulpa_nested_laplace(
    y = d$y, n_trials = d$n, X = d$X, prior = prior, family = "binomial",
    control = list(max_iter = 30L, tol = 1e-6, progress = FALSE)))
  expect_equal(length(fit$weights), 2116L)
  expect_equal(sum(fit$weights), 1)
  expect_true(all(is.finite(fit$theta_mean)))
})

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
             n_re_groups = 0L, sigma_re = 1.0, family = "gaussian", phi = 0.25))
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

test_that("the joint dispatch solves every latent x phi cell of the dense tensor", {
  skip_on_cran()
  # 2 x 2 latent cells crossed with three dispersion values: 12 inner solves.
  f <- .cap_joint_fixture(c(0.5, 2.0), c(0.5, 2.0))
  fit <- tulpa_nested_laplace_joint(
    responses = f$responses, prior = f$prior,
    phi_grid = list(b = c(0.1, 0.25, 0.6)),
    control = list(diagnose_k = FALSE, integration = "grid",
                   max_iter = 60L, tol = 1e-6))
  expect_s3_class(fit, "tulpa_nested_laplace_joint_multi")
  expect_equal(length(fit$weights), 12L)
})
