# The marginal fixed-effect precision H_beta of an inner Laplace solve, and how
# a driver that solves many cells reports the cells where it is absent.
#
# A covariance far past the data's scale (a log-Cholesky axis laid across a
# collapsed slope SD reaches L_22 = exp(+500)) leaves the penalized log
# posterior non-finite at the latent start, so the solve never starts: there is
# no mode, no curvature, and no H_beta to form. That cell carries no mass. A
# solve that ran and has no Schur complement is the condition about H_beta,
# signalled per cell and reported once per fit.

.hb_data <- function(seed = 2) {
  set.seed(seed)
  g <- rep(1:20, each = 6); x <- rnorm(120)
  y <- rbinom(120, 1, plogis(-0.3 + 0.8 * x + rnorm(20)[g]))
  list(y = y, x = x, g = g, X = cbind(1, x))
}

.hb_re_list <- function(d, L) {
  list(list(idx = d$g, n_groups = 20L, n_coefs = 2L, Z = cbind(1, d$x), L = L))
}

test_that("a solve that never starts carries no mass and no H_beta warning", {
  d <- .hb_data()
  L <- matrix(c(0.07, -0.002, 0, exp(371)), 2)
  expect_no_warning(
    f <- tulpa_laplace(d$y, rep(1L, 120), d$X, re_list = .hb_re_list(d, L),
                       family = "binomial", return_hessian = TRUE))
  expect_true(isTRUE(f$start_infeasible))
  expect_identical(f$log_marginal, -Inf)
  expect_null(f$H_beta)
})

test_that("an unformed H_beta is a classed per-cell condition", {
  d <- .hb_data()
  local_mocked_bindings(.laplace_marginal_H_fixed = function(...) NULL,
                        .package = "tulpa")
  expect_warning(
    f <- tulpa_laplace(d$y, rep(1L, 120), d$X,
                       re_list = .hb_re_list(d, diag(c(0.8, 0.5))),
                       family = "binomial", return_hessian = TRUE),
    class = "tulpa_hbeta_unformed")
  expect_true(is.finite(f$log_marginal))
  expect_null(f$H_beta)
})

test_that("the integration nodes' unformed H_beta are reported once per fit", {
  w <- c(0.5, 0.2, 0.3)
  expect_silent(tulpa:::.re_cov_report_hbeta(c(FALSE, FALSE, FALSE), w, "f"))
  expect_warning(tulpa:::.re_cov_report_hbeta(c(TRUE, FALSE, TRUE), w, "f"),
                 "at 2 of 3 integration nodes.*held 0.8 of the posterior")

  skip_on_cran()
  d <- .hb_data()
  local_mocked_bindings(.laplace_marginal_H_fixed = function(...) NULL,
                        .package = "tulpa")
  msgs <- character(0)
  withCallingHandlers(
    tulpa_re_cov_nested(d$y, rep(1L, 120), d$X,
                        re_terms = list(list(idx = d$g, n_groups = 20L, n_coefs = 2L,
                                             Z = cbind(1, d$x),
                                             correlated = TRUE)),
                        family = "binomial",
                        control = list(integration = "ccd", diagnose_k = FALSE)),
    warning = function(w) {
      msgs <<- c(msgs, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  hb <- grep("marginal fixed-effect precision", msgs, value = TRUE)
  expect_length(hb, 1L)
  expect_match(hb, "integration nodes")
})
