# fit_spde()'s default outer design over (log range, log sigma) is the aligned
# tensor (`.nl_outer_design()`, `.nl_aligned_tensor()`): uniform cells on the
# whitened coordinates of the mode's Gaussian. The 9-point CCD stays the
# explicit opt-in `control$method = "ccd"`.

skip_on_cran()

.spde_design_fixture <- fixture_once(function() {
  set.seed(1)
  n <- 200
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  sp <- spatial_spde(~ lon + lat, data = d, max_edge = c(0.1, 0.3))
  d$y <- rpois(n, exp(1 + 0.5 * d$x + sin(3 * d$lon)))
  list(d = d, sp = sp, X = model.matrix(~ x, d))
})

.spde_fit <- function(fx, method = NULL) {
  ctl <- list(diagnose_k = FALSE)
  if (!is.null(method)) ctl$method <- method
  suppressWarnings(fit_spde(fx$d$y, fx$X, fx$sp, family = "poisson",
                            control = ctl))
}

test_that("the default lays the aligned tensor and the CCD is an opt-in", {
  skip_on_cran()
  fx <- .spde_design_fixture()
  fa <- .spde_fit(fx)
  des <- .nl_outer_design(2L)
  expect_identical(fa$nested$method, "aligned_grid")
  expect_identical(fa$nested$integration, "grid")
  expect_identical(nrow(fa$nested$theta_grid), as.integer(des$n_per_axis^2L))
  expect_equal(sum(fa$nested$weights), 1)
  expect_true(is.finite(fa$log_evidence))
  fc <- .spde_fit(fx, "ccd")
  expect_identical(fc$nested$method, "ccd")
  expect_identical(nrow(fc$nested$theta_grid), 9L)
})

test_that("the aligned tensor reads the upper tail a dense quadrature reads", {
  skip_if_not_slow()
  fx <- .spde_design_fixture()
  # The fit's own (range, sigma) log posterior, captured from the door.
  cap <- new.env()
  trace("fit_spde_nested_mode", where = asNamespace("tulpa"), print = FALSE,
        at = 1, tracer = bquote(assign("f", list(lm = spde_log_marginal,
                                                 sp = sp, hp = hyperprior),
                                       envir = .(cap))))
  on.exit(untrace("fit_spde_nested_mode", where = asNamespace("tulpa")),
          add = TRUE)
  fa <- .spde_fit(fx)
  fc <- .spde_fit(fx, "ccd")
  S <- solve(fa$nested$hessian_logsc)
  ctr <- log(c(fa$nested$range_mode, fa$nested$sigma_mode))
  g1 <- seq(ctr[1] - 6 * sqrt(S[1, 1]), ctr[1] + 6 * sqrt(S[1, 1]),
            length.out = 61)
  g2 <- seq(ctr[2] - 6 * sqrt(S[2, 2]), ctr[2] + 6 * sqrt(S[2, 2]),
            length.out = 61)
  G <- expand.grid(a = g1, b = g2)
  lp <- cap$f$lm(exp(G$a), exp(G$b))$log_marginal +
    .spde_log_hyperprior(exp(G$a), exp(G$b), cap$f$sp, cap$f$hp)
  w <- exp(lp - max(lp[is.finite(lp)])); w[!is.finite(w)] <- 0; w <- w / sum(w)
  hi <- function(x) {
    o <- order(x); cw <- cumsum(w[o]) - w[o] / 2
    exp(stats::approx(cw, x[o], 0.975, ties = mean)$y)
  }
  ref <- c(range = hi(G$a), sigma = hi(G$b))
  ra <- spatial_range(fa)[c("range", "sigma"), "q97.5"]
  rc <- spatial_range(fc)[c("range", "sigma"), "q97.5"]
  # Reference (3.47, 1.20); the CCD read (2.68, 0.83), the tensor (3.60, 1.20).
  expect_true(all(abs(ra / ref - 1) < 0.1))
  expect_true(all(rc / ref < 0.8))
})
