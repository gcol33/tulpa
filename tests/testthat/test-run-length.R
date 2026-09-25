# gcol33/tulpa#872: `warmup >= n_iter` reached every sampler but mala /
# imh_laplace unchecked, and returned an empty fit (hmc, ess, sghmc,
# tulpa_nuts_beta, tulpa_gibbs), NaN (mode = "gibbs") or a C++ vector error
# (warmup > n_iter). One check, `.check_run_length()`, now refuses it before
# any kernel runs. Every case errors at the door, so none of these samples.

run_length_data <- function() {
  set.seed(1)
  n <- 200
  g <- factor(sample(1:10, n, TRUE))
  x <- stats::rnorm(n)
  data.frame(g, x,
             yp = stats::rpois(n, exp(0.3 + 0.8 * x + stats::rnorm(10, 0, 0.7)[g])),
             yb = stats::rbinom(n, 1, 0.5))
}

test_that("the shared check refuses warmup >= n_iter where n_iter counts it", {
  chk <- tulpa:::.check_run_length
  expect_silent(chk(400, 300, "f"))
  expect_error(chk(300, 300, "f"), "Need 0 <= warmup < n_iter")
  expect_error(chk(300, 400, "f"), "keeps n_iter - warmup draws")
  expect_error(chk(1, 0, "f"), "n_iter >= 2")
  expect_error(chk(300, -1, "f"), "Need 0 <= warmup")
  expect_error(chk(NA, 10, "f"), "single whole number")
  expect_error(chk(c(10, 20), 5, "f"), "single whole number")
  # The caller's own spelling is named.
  expect_error(chk(300, 300, "f", warmup_name = "n_warmup"),
               "Need 0 <= n_warmup < n_iter")
  # Where warmup runs on top of the kept iterations, any warmup is valid.
  expect_silent(chk(300, 300, "f", counts = "post"))
  expect_silent(chk(1, 5000, "f", counts = "post"))
  expect_error(chk(0, 10, "f", counts = "post"), "n_iter >= 1")
})

test_that("tulpa() refuses warmup >= n_iter on every total-count sampler", {
  d <- run_length_data()
  for (m in c("hmc", "ess", "sghmc", "sgld")) {
    expect_error(
      tulpa(yp ~ x + (1 | g), d, "poisson", mode = m,
            control = list(n_iter = 300, warmup = 300, seed = 1)),
      "Need 0 <= warmup < n_iter", info = m)
  }
  # warmup > n_iter aborted inside C++ with vector::_M_default_append.
  expect_error(
    tulpa(yp ~ x + (1 | g), d, "poisson", mode = "hmc",
          control = list(n_chains = 1, n_iter = 300, warmup = 400)),
    "Need 0 <= warmup < n_iter")
  # The n_warmup alias reaches the same check.
  expect_error(
    tulpa(yp ~ x + (1 | g), d, "poisson", mode = "hmc",
          control = list(n_iter = 300, n_warmup = 300)),
    "Need 0 <= warmup < n_iter")
  # mode = "gibbs" returned NaN.
  expect_error(
    tulpa(yb ~ x + (1 | g), d, "binomial", mode = "gibbs",
          control = list(n_iter = 300, warmup = 300)),
    "Need 0 <= warmup < n_iter")
  # mode = "mala" is unchanged in substance.
  expect_error(
    suppressWarnings(tulpa(yp ~ x + (1 | g), d, "poisson", mode = "mala",
                           sigma_re = 1,
                           control = list(n_iter = 300, warmup = 300))),
    "Need 0 <= warmup < n_iter")
})

test_that("the direct sampler fitters refuse it too", {
  set.seed(1)
  y <- stats::rbinom(100, 1, 0.5)
  X <- cbind(1, stats::rnorm(100))
  expect_error(tulpa_gibbs(y, rep(1L, 100), X, rep(1:10, 10), 10L,
                           control = list(n_iter = 500, warmup = 500)),
               "Need 0 <= warmup < n_iter")
  # A bare n_iter below the 1000-iteration default warmup is the same mistake.
  expect_error(tulpa_gibbs(y, rep(1L, 100), X, rep(1:10, 10), 10L,
                           control = list(n_iter = 500)),
               "Need 0 <= warmup < n_iter")
  yb <- stats::rbeta(100, 2, 3)
  expect_error(tulpa_nuts_beta(yb, X, control = list(n_iter = 200,
                                                     n_warmup = 200, seed = 1)),
               "Need 0 <= n_warmup < n_iter")
})
