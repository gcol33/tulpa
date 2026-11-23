# Every posterior summary a nested grid fit reports for a latent coordinate is
# centred where its full-latent draws are: each cell at its mode plus its
# leading-order mean shift (`.nested_grid_latent_means()`). Under a skewed
# conditional (a log-link count with few observations per coordinate) the
# shift is nonzero, so a summary still read at the modes sits off the draws by
# several Monte Carlo standard errors.

centre_draws <- function(fit, cols, n = 20000L) {
  src <- .field_cells(fit)
  src$point <- FALSE
  set.seed(11)
  D <- .field_latent_draws(fit, src, n)[, cols, drop = FALSE]
  list(mean = colMeans(D), mcse = apply(D, 2L, stats::sd) / sqrt(nrow(D)))
}

grid_mix <- function(fit, M, cols) {
  w <- fit$weights / sum(fit$weights)
  as.numeric(crossprod(w, M[, cols, drop = FALSE]))
}

test_that("temporal() on a skewed Poisson RW1 fit is centred at the draws", {
  skip_on_cran()
  set.seed(3)
  n_t <- 30L
  f <- cumsum(stats::rnorm(n_t, 0, 0.35))
  d <- data.frame(t = rep(seq_len(n_t), each = 3L), x = stats::rnorm(3L * n_t))
  d$y <- stats::rpois(nrow(d), exp(-1.2 + 0.3 * d$x + f[d$t]))
  fit <- suppressMessages(tulpa(y ~ x, data = d, family = "poisson",
                                temporal = temporal_rw1("t"),
                                mode = "nested_laplace",
                                control = list(progress = FALSE)))
  expect_true(is.matrix(fit$grid_mean_shift))
  cols <- .nl_temporal_field_cols(fit)
  centred <- grid_mix(fit, .nested_grid_latent_means(fit), cols)
  at_mode <- grid_mix(fit, fit$modes, cols)
  # The shift is what separates the two reads; measured 0.011 at its largest.
  expect_gt(max(abs(centred - at_mode)), 5e-3)

  set.seed(7)
  sm <- summary(temporal(fit))
  expect_equal(sm$mean, centred, tolerance = 1e-10)
  dr <- centre_draws(fit, cols)
  # 30 coordinates at 20000 draws: measured 1.79 MCSE at the largest (4.92
  # with the summary at the modes).
  expect_lt(max(abs(sm$mean - dr$mean) / dr$mcse), 4)
})

test_that("fit_st_nested() field means and temporal() read the cell centres", {
  skip_on_cran()
  set.seed(21)
  n_s <- 12L; n_t <- 10L; N <- 240L
  adj <- matrix(0, n_s, n_s)
  for (i in 1:(n_s - 1L)) adj[i, i + 1L] <- adj[i + 1L, i] <- 1
  s <- sample(n_s, N, TRUE); tt <- sample(n_t, N, TRUE)
  us <- 0.5 * as.numeric(scale(cumsum(stats::rnorm(n_s))))
  vt <- 0.5 * as.numeric(scale(cumsum(stats::rnorm(n_t))))
  x <- stats::rnorm(N)
  y <- stats::rpois(N, exp(-1.3 + 0.3 * x + us[s] + vt[tt]))
  fit <- fit_st_nested(y, cbind(1, x), s, adj, tt, n_t, spatial_type = "icar",
                       temporal_type = "rw1", family = "poisson",
                       control = list(progress = FALSE))
  expect_true(is.matrix(fit$grid_mean_shift))
  n_x <- ncol(fit$modes)
  C <- .nested_grid_latent_means(fit)
  te <- .nl_role_cols(fit$prior, fit$block_latent_offsets, n_x, "temporal")
  sp <- .nl_role_cols(fit$prior, fit$block_latent_offsets, n_x, "spatial")
  expect_gt(max(abs(grid_mix(fit, C, te) - grid_mix(fit, fit$modes, te))), 5e-3)
  expect_equal(fit$temporal_effects, grid_mix(fit, C, te), tolerance = 1e-10)
  expect_equal(fit$spatial_effects, grid_mix(fit, C, sp), tolerance = 1e-10)
  set.seed(7)
  expect_equal(summary(temporal(fit))$mean, grid_mix(fit, C, te),
               tolerance = 1e-10)

  # Measured 1.77 (temporal) and 1.89 (spatial) MCSE at the largest; 8.80 and
  # 5.81 with the field means at the modes.
  dt <- centre_draws(fit, te)
  ds <- centre_draws(fit, sp)
  expect_lt(max(abs(fit$temporal_effects - dt$mean) / dt$mcse), 4)
  expect_lt(max(abs(fit$spatial_effects - ds$mean) / ds$mcse), 4)

  g <- tulpa_outer_grid(fit)
  expect_equal(g$means, C)
})
