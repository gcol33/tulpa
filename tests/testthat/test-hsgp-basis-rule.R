# The HSGP basis size and boundary factor follow Riutort-Mayol et al. (2023,
# Eqs. 19-20) for the squared-exponential kernel, and a basis left to the data
# grows until it represents the lengthscales its posterior holds.

test_that("the basis rule reproduces the paper's worked values", {
  # Appendix C of the paper, S = 1: an initial guess ell = 0.5 gives c = 1.60
  # and m = 6; ell = 1 gives c = 3.20 and m = 6; ell = 0.17 gives c = 1.20 and
  # m = 13.
  a <- tulpa:::.hsgp_basis_size(1, 0.5)
  expect_equal(a$c, 1.6)
  expect_identical(a$m, 6L)
  b <- tulpa:::.hsgp_basis_size(1, 1)
  expect_equal(b$c, 3.2)
  expect_identical(b$m, 6L)
  d <- tulpa:::.hsgp_basis_size(1, 0.17)
  expect_equal(d$c, 1.2)
  expect_identical(d$m, 13L)
})

test_that("the lengthscale window inverts the rule", {
  S <- c(1.7, 1.2)
  sz <- tulpa:::.hsgp_basis_size(S, 0.3, 0.9)
  win <- tulpa:::.hsgp_lengthscale_window(sz$m, sz$c, S)
  expect_lte(win[["lo"]], 0.3)
  expect_gte(win[["hi"]], 0.9 - 1e-12)
  # One basis function fewer no longer reaches the smallest lengthscale.
  expect_gt(tulpa:::.hsgp_lengthscale_window(sz$m - 1L, sz$c, S)[["lo"]], 0.3)
  # m is bounded by what the constructor accepts.
  big <- tulpa:::.hsgp_basis_size(1, 1e-3)
  expect_identical(big$m, 50L)
  expect_gt(big$m_req, 50)
})

test_that("the diagnostic reads both ends of the window with the slack", {
  S <- c(1.5, 1.2)
  win <- tulpa:::.hsgp_lengthscale_window(10L, 1.2, S)
  holds <- function(ell) tulpa:::.hsgp_basis_holds(10L, 1.2, S, ell)
  lo_slack <- 0.01 * max(S)
  hi_slack <- 0.01 * min(S)
  expect_true(all(holds(win[["lo"]])))
  expect_true(holds(win[["lo"]] - 0.99 * lo_slack)[["lower"]])
  expect_false(holds(win[["lo"]] - 1.01 * lo_slack)[["lower"]])
  expect_true(holds(win[["hi"]] + 0.99 * hi_slack)[["upper"]])
  expect_false(holds(win[["hi"]] + 1.01 * hi_slack)[["upper"]])
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
