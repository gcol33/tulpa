# geweke_test(): a within-chain diagnostic, computed per chain, on validated
# windows (gcol33/tulpa#895).

.geweke_fixture <- function() {
  set.seed(7)
  n <- 400L
  # Two stationary chains at different levels: each one's start agrees with
  # its own end, so the per-chain z is unremarkable, while concatenating them
  # puts chain 1 in the first window and chain 2 in the last.
  arr <- array(NA_real_, dim = c(n, 2L, 2L),
               dimnames = list(NULL, NULL, c("a", "b")))
  arr[, 1L, "a"] <- as.numeric(stats::arima.sim(list(ar = 0.5), n))
  arr[, 2L, "a"] <- 3 + as.numeric(stats::arima.sim(list(ar = 0.5), n))
  arr[, 1L, "b"] <- stats::rnorm(n)
  arr[, 2L, "b"] <- stats::rnorm(n)
  structure(list(draws = arr, draws_kind = "chain", n_chains = 2L),
            class = "tulpa_fit")
}

test_that("geweke_test() reports each chain, never the concatenation (#895)", {
  fit <- .geweke_fixture()
  g <- geweke_test(fit, pars = c("a", "b"))
  expect_s3_class(g, "tulpa_geweke")
  expect_named(g, c("chain", "parameter", "z_score", "p_value"))
  expect_equal(nrow(g), 4L)
  expect_equal(g$chain, c(1L, 1L, 2L, 2L))

  # Each row is the one-chain statistic on that chain's own series.
  one <- function(x, f1 = 0.1, f2 = 0.5) {
    n <- length(x); x1 <- x[seq_len(floor(f1 * n))]
    x2 <- x[(n - floor(f2 * n) + 1):n]
    (mean(x1) - mean(x2)) /
      sqrt(spectrum0_ar(x1) / length(x1) + spectrum0_ar(x2) / length(x2))
  }
  for (k in 1:2) for (p in c("a", "b")) {
    expect_equal(g$z_score[g$chain == k & g$parameter == p],
                 one(fit$draws[, k, p]), tolerance = 1e-12)
  }
  # The level shift between chains does not register as non-convergence.
  expect_true(all(abs(g$z_score[g$parameter == "a"]) < 4))
})

test_that("geweke_test() per-chain z agrees with coda::geweke.diag (#895)", {
  skip_if_not_installed("coda")
  fit <- .geweke_fixture()
  g <- geweke_test(fit, pars = "a")
  ref <- vapply(1:2, function(k)
    unname(coda::geweke.diag(coda::mcmc(fit$draws[, k, "a"]))$z), numeric(1))
  # coda fits its spectral AR by Yule-Walker, tulpa by Burg: same statistic,
  # slightly different spectral estimate.
  expect_equal(g$z_score, ref, tolerance = 0.1)
})

test_that("geweke_test() validates its windows (#895)", {
  fit <- .geweke_fixture()
  expect_error(geweke_test(fit, frac1 = 0), "`frac1`")
  expect_error(geweke_test(fit, frac2 = 1), "`frac2`")
  expect_error(geweke_test(fit, frac1 = NA), "`frac1`")
  expect_error(geweke_test(fit, frac1 = c(0.1, 0.2)), "`frac1`")
  expect_error(geweke_test(fit, frac1 = 0.7, frac2 = 0.6), "overlap")
  expect_error(geweke_test(fit, frac1 = 0.001), "at least 2")
})

test_that("print() tolerates an undefined z-score (#895)", {
  fit <- .geweke_fixture()
  fit$draws[, 1L, "b"] <- 1                  # constant within chain 1
  g <- geweke_test(fit, pars = "b")
  expect_true(is.na(g$z_score[g$chain == 1L]))
  expect_output(print(g), "Geweke")
})
