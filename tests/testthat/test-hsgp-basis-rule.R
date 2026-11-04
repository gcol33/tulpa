# The HSGP basis size and boundary factor follow Riutort-Mayol et al. (2023,
# Eqs. 19-24) for the kernel the basis approximates, and a basis left to the
# data grows until it represents the lengthscales its posterior holds (Phase A)
# and then until a refit with more functions is stable (Phase B).

test_that("the basis rule reproduces the paper's worked values", {
  se <- "squared_exponential"
  # Appendix C of the paper, squared exponential, S = 1: an initial guess
  # ell = 0.5 gives c = 1.60 and m = 6; ell = 1 gives c = 3.20 and m = 6;
  # ell = 0.17 gives c = 1.20 and m = 13.
  a <- tulpa:::.hsgp_basis_size(1, 0.5, kernel = se)
  expect_equal(a$c, 1.6)
  expect_identical(a$m, 6L)
  b <- tulpa:::.hsgp_basis_size(1, 1, kernel = se)
  expect_equal(b$c, 3.2)
  expect_identical(b$m, 6L)
  d <- tulpa:::.hsgp_basis_size(1, 0.17, kernel = se)
  expect_equal(d$c, 1.2)
  expect_identical(d$m, 13L)
  # Table 2 of the paper, Matern 3/2, S = 1: ell = 0.5 gives c = 2.25 and
  # m = 16; ell = 0.12 gives c = 1.20 and m = 35.
  m1 <- tulpa:::.hsgp_basis_size(1, 0.5, kernel = "matern32")
  expect_equal(m1$c, 2.25)
  expect_identical(m1$m, 16L)
  m2 <- tulpa:::.hsgp_basis_size(1, 0.12, kernel = "matern32")
  expect_equal(m2$c, 1.2)
  expect_identical(m2$m, 35L)
  # Eqs. 21-22, Matern 5/2: c = 4.1 ell / S, m = 2.65 c / (ell / S).
  m5 <- tulpa:::.hsgp_basis_size(1, 0.5, kernel = "matern52")
  expect_equal(m5$c, 2.05)
  expect_identical(m5$m, as.integer(ceiling(2.65 * 2.05 / 0.5)))
})

test_that("a door's cov / nu name the kernel, one default for every door", {
  expect_identical(tulpa:::.hsgp_kernel(NULL), "matern32")
  expect_identical(tulpa:::.hsgp_kernel("matern", 1.5), "matern32")
  expect_identical(tulpa:::.hsgp_kernel("matern", 2.5), "matern52")
  expect_identical(tulpa:::.hsgp_kernel("squared_exponential"),
                   "squared_exponential")
  expect_error(tulpa:::.hsgp_kernel("exponential"), "no HSGP basis rule")
  expect_error(tulpa:::.hsgp_kernel("matern", 0.5), "nu = 1.5 or 2.5")
  expect_identical(spatial_gp(~ lon + lat, approx = "hsgp")$kernel, "matern32")
  expect_identical(spatial_gp(~ lon + lat, approx = "hsgp",
                              cov = "squared_exponential")$kernel,
                   "squared_exponential")
  expect_identical(spatial_svc(~ lon + lat, approx = "hsgp", cov = "matern",
                               nu = 2.5)$kernel, "matern52")
  expect_identical(suppressMessages(spatial_multiscale(
    ~ lon + lat, approx = "hsgp", range_regional = c(1, 10)))$kernel,
    "matern32")
  # The NNGP branches keep their own default.
  expect_identical(spatial_gp(~ lon + lat)$cov, "exponential")
  expect_null(spatial_gp(~ lon + lat)$kernel)
})

test_that("the lengthscale window inverts the rule", {
  S <- c(1.7, 1.2)
  for (k in names(tulpa:::.HSGP_KERNELS)) {
    sz <- tulpa:::.hsgp_basis_size(S, 0.6, 0.9, kernel = k)
    expect_lt(sz$m, 50L)
    win <- tulpa:::.hsgp_lengthscale_window(sz$m, sz$c, S, k)
    expect_lte(win[["lo"]], 0.6)
    expect_gte(win[["hi"]], 0.9 - 1e-12)
    # One basis function fewer no longer reaches the smallest lengthscale.
    expect_gt(tulpa:::.hsgp_lengthscale_window(sz$m - 1L, sz$c, S, k)[["lo"]],
              0.6)
    # m is bounded by what the constructor accepts.
    big <- tulpa:::.hsgp_basis_size(1, 1e-3, kernel = k)
    expect_identical(big$m, 50L)
    expect_gt(big$m_req, 50)
  }
  # A rougher kernel needs more functions for the same lengthscale.
  m_of <- function(k) tulpa:::.hsgp_basis_size(1, 0.3, kernel = k)$m
  expect_lt(m_of("squared_exponential"), m_of("matern52"))
  expect_lt(m_of("matern52"), m_of("matern32"))
})

test_that("the diagnostic reads both ends of the window with the slack", {
  S <- c(1.5, 1.2)
  for (k in names(tulpa:::.HSGP_KERNELS)) {
    win <- tulpa:::.hsgp_lengthscale_window(30L, 2, S, k)
    holds <- function(ell) tulpa:::.hsgp_basis_holds(30L, 2, S, ell, k)
    lo_slack <- 0.01 * max(S)
    hi_slack <- 0.01 * min(S)
    expect_true(all(holds(win[["lo"]])))
    expect_true(holds(win[["lo"]] - 0.99 * lo_slack)[["lower"]])
    expect_false(holds(win[["lo"]] - 1.01 * lo_slack)[["lower"]])
    expect_true(holds(win[["hi"]] + 0.99 * hi_slack)[["upper"]])
    expect_false(holds(win[["hi"]] + 1.01 * hi_slack)[["upper"]])
  }
})

test_that("a refit step follows the lower end of the window only", {
  set.seed(1)
  d <- data.frame(lon = runif(80), lat = runif(80))
  sp <- validate_hsgp(spatial_gp(~ lon + lat, approx = "hsgp"), d)
  S <- sp$basis_rule$half_width
  win <- tulpa:::.hsgp_lengthscale_window(sp$m, sp[["c"]], S)
  post_at <- function(ell) {
    local_mocked_bindings(
      .hsgp_lengthscale_posterior = function(fit, axes = NULL)
        rbind(c(mean = ell, lo = ell, hi = ell)),
      .package = "tulpa")
    tulpa:::.hsgp_refit_spec(sp, list())
  }
  above <- post_at(10 * win[["hi"]])
  expect_false(above$refit)
  expect_identical(above$spec$m, sp$m)
  below <- post_at(0.5 * win[["lo"]])
  expect_true(below$refit)
  expect_lte(tulpa:::.hsgp_lengthscale_window(below$spec$m, below$spec[["c"]],
                                              S)[["lo"]], 0.5 * win[["lo"]])
  expect_identical(below$spec$basis_rule$history$source,
                   c("prior", "posterior"))

  # A supplied m leaves only c to move; once c sits at its floor of 1.2 the
  # window cannot reach further down, and the loop stops instead of refitting
  # the same basis.
  fixed_m <- validate_hsgp(spatial_gp(~ lon + lat, approx = "hsgp", m = 6), d)
  floor_c <- tulpa:::.hsgp_resolve_basis(fixed_m, 1e-3, "posterior")
  expect_identical(floor_c[["c"]], 1.2)
  sp <- floor_c
  stuck <- post_at(1e-3)
  expect_false(stuck$refit)
  expect_identical(stuck$spec[["c"]], 1.2)
})

# A sequence of fits whose lengthscale posterior and linear-predictor summary
# the test sets, read through mocked accessors: each element of `seq` is the
# (lengthscale mean, interval half-width, linear-predictor shift) of one fit.
.phase_b_run <- function(spec, seq) {
  k <- 0L
  local_mocked_bindings(
    .hsgp_lengthscale_posterior = function(fit, axes = NULL)
      rbind(c(mean = fit$ell, lo = fit$ell - fit$hw, hi = fit$ell + fit$hw)),
    posterior_linpred = function(object, ...)
      data.frame(estimate = object$shift + c(0, 1),
                 conf.low = object$shift + c(0, 1) - 0.5,
                 conf.high = object$shift + c(0, 1) + 0.5),
    .package = "tulpa")
  run <- tulpa:::.hsgp_fit_basis_loop(spec, function(s) {
    k <<- k + 1L
    e <- seq[[min(k, length(seq))]]
    list(ell = e[1], hw = e[2], shift = e[3], m = s$m)
  })
  list(run = run, n_fit = k)
}

test_that("Phase B adds five functions and stops at the first stable refit", {
  set.seed(1)
  d <- data.frame(lon = runif(80), lat = runif(80))
  sp <- validate_hsgp(spatial_gp(~ lon + lat, approx = "hsgp"), d)
  S <- sp$basis_rule$half_width
  ell <- 2 * tulpa:::.hsgp_lengthscale_window(sp$m, sp[["c"]], S,
                                              "matern32")[["lo"]]
  # Phase A holds at once; the Phase B refit moves inside both intervals.
  r <- .phase_b_run(sp, list(c(ell, 0.1, 0), c(ell * 1.01, 0.1, 0.1)))
  expect_identical(r$n_fit, 2L)
  expect_identical(r$run$spec$m, sp$m + 5L)
  expect_identical(tail(r$run$spec$basis_rule$history$source, 1L), "phase B")
  expect_identical(r$run$fit$m, sp$m + 5L)

  # A refit whose linear predictor leaves the previous intervals is not
  # stable: five more functions, then stable.
  r2 <- .phase_b_run(sp, list(c(ell, 0.1, 0), c(ell, 0.1, 0.8),
                              c(ell, 0.1, 0.9)))
  expect_identical(r2$n_fit, 3L)
  expect_identical(r2$run$spec$m, sp$m + 10L)

  # So is one whose lengthscale leaves the previous interval.
  r3 <- .phase_b_run(sp, list(c(ell, 0.01, 0), c(ell * 1.5, 0.01, 0),
                              c(ell * 1.5, 0.01, 0)))
  expect_identical(r3$n_fit, 3L)

  # A supplied m ends the procedure after Phase A.
  fixed <- validate_hsgp(spatial_gp(~ lon + lat, approx = "hsgp", m = 8), d)
  r4 <- .phase_b_run(fixed, list(c(ell, 0.1, 0), c(ell, 0.1, 5)))
  expect_identical(r4$n_fit, 1L)
  expect_identical(r4$run$spec$m, 8L)
})

test_that("Phase B stops at the basis-size bound", {
  set.seed(1)
  d <- data.frame(lon = runif(80), lat = runif(80))
  sp <- validate_hsgp(spatial_gp(~ lon + lat, approx = "hsgp"), d)
  sp$m <- 48L
  ell <- 10
  r <- .phase_b_run(sp, list(c(ell, 0.1, 0), c(ell, 0.1, 5), c(ell, 0.1, 9)))
  expect_identical(r$run$spec$m, 50L)
  expect_identical(r$n_fit, 2L)
})

test_that("the PC range quantile is the inverse of its CDF", {
  U <- 0.7; alpha <- 0.1
  for (p in c(0.05, 0.5, 0.9)) {
    r <- tulpa:::.pc_range_quantile(p, U, alpha, 2)
    lambda <- -log(alpha) * U
    expect_equal(exp(-lambda / r), p)
  }
  expect_equal(tulpa:::.pc_range_quantile(alpha, U, alpha, 2), U)
})

test_that("an unset basis is chosen at validation and a supplied one is kept", {
  set.seed(1)
  d <- data.frame(lon = runif(80), lat = runif(80))
  auto <- validate_hsgp(spatial_gp(~ lon + lat, approx = "hsgp"), d)
  S <- tulpa:::.hsgp_half_widths(auto$coords_matrix)
  ell0 <- tulpa:::.hsgp_prior_lengthscale(auto$coords_matrix)
  expect_identical(auto$m, tulpa:::.hsgp_basis_size(S, ell0)$m)
  expect_equal(auto[["c"]], tulpa:::.hsgp_basis_size(S, ell0)$c)
  expect_identical(auto$basis_rule$history$source, "prior")

  fixed <- validate_hsgp(spatial_gp(~ lon + lat, approx = "hsgp", m = 9,
                                    c = 2), d)
  expect_identical(fixed$m, 9L)
  expect_identical(fixed[["c"]], 2)
  expect_false(any(fixed$basis_rule$auto))

  half <- validate_hsgp(spatial_gp(~ lon + lat, approx = "hsgp", c = 2.5), d)
  expect_identical(half[["c"]], 2.5)
  expect_identical(half$m, tulpa:::.hsgp_basis_size(S, ell0, c = 2.5)$m)

  expect_error(spatial_gp(~ lon + lat, approx = "hsgp", m = 2), "between 3")
  expect_error(spatial_gp(~ lon + lat, approx = "hsgp", c = 0.5), ">= 1")
})

# A squared-exponential field with a lengthscale of 0.1 on the unit square,
# below the 0.22 that a basis of m = 6 at c = 1.5 represents there.
.hsgp_rough_data <- function(seed, n = 250L, ell = 0.1) {
  set.seed(seed)
  d <- data.frame(lon = runif(n + 60L), lat = runif(n + 60L))
  H <- as.matrix(dist(d))
  K <- 0.8^2 * exp(-H^2 / (2 * ell^2))
  w <- as.numeric(t(chol(K + diag(1e-8, nrow(K)))) %*% rnorm(nrow(K)))
  d$x <- rnorm(nrow(d))
  d$eta <- 0.5 + 0.4 * d$x + w
  d$y <- rpois(nrow(d), exp(d$eta))
  list(obs = d[seq_len(n), ], out = d[n + seq_len(60L), ])
}

test_that("a basis left to the data grows to hold the posterior lengthscale", {
  skip_on_cran()
  dd <- .hsgp_rough_data(3L)
  fit <- suppressMessages(tulpa(y ~ x, data = dd$obs, family = "poisson",
    spatial = spatial_gp(~ lon + lat, approx = "hsgp"),
    control = list(n_threads = 2L)))
  chk <- tulpa:::.hsgp_basis_check(fit)
  expect_true(chk$holds[["lower"]])
  expect_gt(chk$m, 6L)
  expect_gt(nrow(fit$spatial$basis_rule$history), 1L)
  expect_false(grepl("down to", attr(diagnostics(fit), "hsgp_basis_note") %||% ""))

  small <- suppressMessages(tulpa(y ~ x, data = dd$obs, family = "poisson",
    spatial = spatial_gp(~ lon + lat, approx = "hsgp", m = 6, c = 1.5),
    control = list(n_threads = 2L)))
  expect_false(tulpa:::.hsgp_basis_check(small)$holds[["lower"]])
  expect_match(attr(diagnostics(small), "hsgp_basis_note"),
               "represents lengthscales down to")
})

test_that("held-out coverage of a rough field reaches nominal", {
  skip_if_not_slow()
  cov <- vapply(1:6, function(s) {
    dd <- .hsgp_rough_data(10L + s)
    fit <- suppressMessages(tulpa(y ~ x, data = dd$obs, family = "poisson",
      spatial = spatial_gp(~ lon + lat, approx = "hsgp"),
      control = list(n_threads = 2L)))
    p <- predict(fit, newdata = dd$out, se.fit = TRUE)
    mean(p$lower <= dd$out$eta & dd$out$eta <= p$upper)
  }, numeric(1))
  expect_gte(mean(cov), 0.88)
})
.hsgp_upper_miss <- function(kernel) {
  list(holds = c(lower = TRUE, upper = FALSE), kernel = kernel,
       window = c(lo = .1, hi = 2), lengthscale = c(mean = 1),
       lengthscale_hi = c(mean = 3), capped = FALSE,
       auto = c(m = FALSE, c = FALSE), m = 10L, c = 2)
}

test_that("a fixed HSGP boundary miss has a readable basis diagnostic", {
  for (kernel in c("squared_exponential", "matern32", "matern52")) {
    note <- tulpa:::.hsgp_basis_note(.hsgp_upper_miss(kernel))
    expect_type(note, "character")
    expect_length(note, 1L)
    expect_match(note, "2-D boundary variance criterion")
  }
})
