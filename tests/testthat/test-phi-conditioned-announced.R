# An unsupplied dispersion is a CHOICE, and it is announced (gcol33/tulpa#849).
#
# `phi` defaults to 1 and is conditioned on, which is the stated design
# (`?tulpa`, `@param estimate_phi`) and the same shape `sigma_re` has. The
# difference was that `sigma_re` warned and `phi` did not, so a gaussian fit
# silently ran at a residual variance of 1 against a truth of 0.2025 -- a 4.9x
# overstatement that `posterior_predict()`, `bayes_R2()`, `pp_check()` and
# WAIC / LOO all read off `fit$phi`.
#
# And `fit$phi_estimated`, the one field that tells an estimate from a
# conditioned value, was set on `tulpa_eb()` alone, so nowhere else could a
# consumer even ask.

.pca_data <- function(n = 200L, seed = 5L) {
  set.seed(seed)
  d <- data.frame(x = rnorm(n), g = factor(sample(12L, n, TRUE)))
  b <- rnorm(12L, 0, 0.6)
  d$y <- rnorm(n, 0.4 + 0.8 * d$x + b[as.integer(d$g)], 0.45)
  d
}

.pca_warnings <- function(expr) {
  ws <- character(0)
  withCallingHandlers(suppressMessages(expr),
                      warning = function(w) {
                        ws <<- c(ws, conditionMessage(w))
                        invokeRestart("muffleWarning")
                      })
  ws
}


test_that("a defaulted phi warns for the families that read one, and only those", {
  skip_on_cran()
  d <- .pca_data()
  said <- function(ws) any(grepl("`phi` not supplied", ws, fixed = TRUE))

  # Reads a dispersion -> announced, naming the family and both ways out.
  ws <- .pca_warnings(tulpa(y ~ x, data = d, family = "gaussian",
                            mode = "laplace"))
  expect_true(said(ws))
  hit <- grep("`phi` not supplied", ws, fixed = TRUE, value = TRUE)
  expect_match(hit, "gaussian")
  expect_match(hit, "estimate_phi = TRUE")

  # Supplied -> the caller made the choice, nothing to say.
  expect_false(said(.pca_warnings(
    tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace", phi = 0.2))))

  # Reads no dispersion -> silent. The warning must not become noise on the
  # two most common families.
  dp <- d; dp$y <- rpois(nrow(d), exp(0.4 + 0.8 * d$x))
  db <- d; db$y <- rbinom(nrow(d), 1L, plogis(0.4 + 0.8 * d$x))
  expect_false(said(.pca_warnings(
    tulpa(y ~ x, data = dp, family = "poisson", mode = "laplace"))))
  expect_false(said(.pca_warnings(
    tulpa(y ~ x, data = db, family = "binomial", mode = "laplace"))))
})


test_that("every fit says whether its dispersion was estimated", {
  skip_on_cran()
  d <- .pca_data()

  # Conditioned: FALSE, on a door that is not `eb`.
  cond <- suppressWarnings(suppressMessages(
    tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace")))
  expect_false(cond$phi_estimated)
  expect_equal(cond$phi, 1)

  # Estimated: TRUE, and the value is the estimate rather than the start.
  est <- suppressWarnings(suppressMessages(
    tulpa(y ~ x + (1 | g), data = d, family = "gaussian", mode = "eb",
          estimate_phi = TRUE)))
  expect_true(est$phi_estimated)
  expect_gt(est$phi, 0.1)
  expect_lt(est$phi, 0.4)              # truth 0.45^2 = 0.2025
  # The field is not merely present: it disagrees with the conditioned fit,
  # which is the whole point of having it.
  expect_false(isTRUE(all.equal(est$phi, cond$phi)))

  # `mode = "eb"` WITHOUT estimate_phi still conditions, and says so.
  eb_cond <- suppressWarnings(suppressMessages(
    tulpa(y ~ x + (1 | g), data = d, family = "gaussian", mode = "eb")))
  expect_false(eb_cond$phi_estimated)
  expect_equal(eb_cond$phi, 1)

  # A family with no dispersion has nothing to have estimated, so the field is
  # absent rather than a meaningless FALSE.
  dp <- d; dp$y <- rpois(nrow(d), exp(0.4 + 0.8 * d$x))
  pois <- suppressWarnings(suppressMessages(
    tulpa(y ~ x, data = dp, family = "poisson", mode = "laplace")))
  expect_null(pois$phi_estimated)
})


test_that("the inner solve's conditioned phi does not answer for the outer fit", {
  skip_on_cran()
  # tulpa_eb() assembles its result as c(fit_hat, list(...)), and `fit_hat` is
  # a finalized tulpa_laplace fit -- so a name set on both lands TWICE and `$`
  # answers with whichever came first. The inner solve CONDITIONS on a phi, so
  # its `phi_estimated = FALSE` would answer for the outer fit that estimated
  # one. `phi` was already cleared for exactly this reason; `phi_estimated`
  # joins it.
  d <- .pca_data()
  X <- cbind(1, d$x); colnames(X) <- c("(Intercept)", "x")
  rt <- list(list(idx = as.integer(d$g), n_groups = nlevels(d$g),
                  n_coefs = 1L))
  fit <- suppressWarnings(suppressMessages(tulpa_eb(
    y = d$y, n_trials = rep(1L, nrow(d)), X = X, re_terms = rt,
    family = "gaussian", phi = 1, estimate_phi = TRUE)))

  expect_equal(sum(names(fit) == "phi_estimated"), 1L)
  expect_equal(sum(names(fit) == "phi"), 1L)
  expect_true(fit$phi_estimated)
})
