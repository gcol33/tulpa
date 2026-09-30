# The cell-coupled sparse scatter splits its cell loop into a fixed number of
# chunks set by the cell count alone (coupled_chunk_count), writes an entry one
# chunk owns in place and adds the chunk partials of a shared one in chunk
# order. Which thread runs a chunk therefore cannot move a number: a fit at one
# inner thread and at four must agree bit for bit, not to tolerance
# (gcol33/tulpa#921). 600 cells is nine chunks, so the shared-entry reduction
# is exercised; the outer width is held at one so the warm-start route is the
# same on both sides.

.cci_fit <- function(n_threads) {
  cpp_register_test_occupancy_mixture_coupling()
  set.seed(921)
  n_cells <- 600L
  n_visits <- 3L
  z <- stats::rbinom(n_cells, 1L, stats::plogis(0.3))
  xo <- stats::rnorm(n_cells)
  xd <- stats::rnorm(n_cells * n_visits)
  y <- as.numeric(stats::rbinom(
    n_cells * n_visits, 1L,
    stats::plogis(-0.4 + 0.5 * xd) * rep(z, each = n_visits)))
  arm <- function(yy, N, map, X) list(
    y = yy, n_trials = rep(1L, N), X = X, family = "binomial", phi = 1,
    coupled = TRUE, cell_obs_map = map, beta_prior_prec = rep(0.25, ncol(X)))
  grp <- ((seq_len(n_cells) - 1L) %% 10L) + 1L
  prior <- list(list(type = "iid", n_units = 10L, sigma_grid = c(0.6, 1.2),
                     obs_idx = list(grp, rep(grp, each = n_visits))))
  tulpa_nested_laplace_joint(
    responses = list(
      occ = arm(rep(0, n_cells), n_cells, seq_len(n_cells), cbind(1, xo)),
      det = arm(y, n_cells * n_visits,
                rep(seq_len(n_cells), each = n_visits), cbind(1, xd))),
    prior = prior, cell_coupling = "test_occupancy_mixture",
    control = list(max_iter = 100L, tol = 1e-10, diagnose_k = FALSE,
                   n_threads = n_threads, n_threads_outer = 1L,
                   force_sparse = TRUE))
}

test_that("the coupled scatter returns the same fit at every thread count", {
  skip_on_cran()
  one  <- .cci_fit(1L)
  four <- .cci_fit(4L)
  expect_true(all(is.finite(one$log_marginal)))
  expect_identical(as.numeric(one$log_marginal), as.numeric(four$log_marginal))
  expect_identical(as.matrix(one$modes), as.matrix(four$modes))
})
