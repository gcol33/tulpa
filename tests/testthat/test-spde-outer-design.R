# fit_spde()'s outer grid over (range, sigma) is the joint door's: laid from
# the outer mode of (log range, log sigma) by the placement pass and read as a
# density over its cells.

skip_on_cran()

.spde_design_fixture <- fixture_once(function() {
  set.seed(1)
  n <- 200
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  sp <- spatial_spde(~ lon + lat, data = d, max_edge = c(0.1, 0.3))
  d$y <- rpois(n, exp(1 + 0.5 * d$x + sin(3 * d$lon)))
  list(d = d, sp = sp, X = model.matrix(~ x, d))
})

.spde_fit <- function(fx) {
  suppressWarnings(fit_spde(fx$d$y, fx$X, fx$sp, family = "poisson",
                            control = list(diagnose_k = FALSE, progress = FALSE)))
}

test_that("the outer grid is laid at the outer mode and carries the evidence", {
  skip_on_cran()
  fx <- .spde_design_fixture()
  fa <- .spde_fit(fx)
  expect_identical(fa$nested$outer_grid_placement, "auto_recentered")
  expect_equal(sum(fa$nested$weights), 1)
  expect_true(is.finite(fa$log_evidence))
})

test_that("the outer grid reads the upper tail a dense quadrature reads", {
  skip_if_not_slow()
  fx <- .spde_design_fixture()
  fa <- .spde_fit(fx)
  n <- length(fx$d$y)
  L <- log(fa$nested$theta_grid)
  wg <- fa$nested$weights
  ctr <- as.numeric(crossprod(wg, L))
  sdl <- sqrt(as.numeric(crossprod(wg, sweep(L, 2L, ctr)^2)))
  g1 <- seq(ctr[1] - 6 * sdl[1], ctr[1] + 6 * sdl[1], length.out = 61)
  g2 <- seq(ctr[2] - 6 * sdl[2], ctr[2] + 6 * sdl[2], length.out = 61)
  G <- expand.grid(a = g1, b = g2)
  solve <- list(re_idx = rep(0L, n), n_re_groups = 0L, sigma_re = 1,
                max_iter = 100L, tol = 1e-6, beta_prior = NULL)
  lm <- tulpa:::.spde_grid_solve(fx$sp, y = fx$d$y, n_trials = rep(1L, n),
                                 X = fx$X, family = "poisson", phi = 1,
                                 offset = NULL, solve = solve,
                                 range_vec = exp(G$a), sigma_vec = exp(G$b),
                                 extra = list(progress = FALSE))$log_marginal
  lp <- lm + tulpa:::.spde_log_hyperprior(exp(G$a), exp(G$b), fx$sp)
  w <- exp(lp - max(lp[is.finite(lp)])); w[!is.finite(w)] <- 0; w <- w / sum(w)
  hi <- function(x) {
    o <- order(x); cw <- cumsum(w[o]) - w[o] / 2
    exp(stats::approx(cw, x[o], 0.975, ties = mean)$y)
  }
  ref <- c(range = hi(G$a), sigma = hi(G$b))
  ra <- spatial_range(fa)[c("range", "sigma"), "q97.5"]
  expect_true(all(abs(ra / ref - 1) < 0.1),
              info = sprintf("grid %s / dense %s", paste(signif(ra, 3), collapse = ", "),
                             paste(signif(ref, 3), collapse = ", ")))
})
