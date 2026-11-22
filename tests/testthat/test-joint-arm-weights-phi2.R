# The joint arm's built-in family response carries observation weights and a
# second dispersion (BuiltinFamilyResponse). At one fixed random-effect scale
# the joint door and the conditional Laplace backend solve the same inner
# problem -- the joint `iid` block is the non-centred form of the same random
# intercept, and the Laplace approximation is invariant under that linear
# change of variables -- so the log marginal and the mode agree.

.jw_data <- function() {
  set.seed(11)
  G <- 12L; per <- 8L; n <- G * per
  g <- rep(seq_len(G), each = per); x <- rnorm(n)
  b <- rnorm(G, 0, 0.6)
  eta <- 0.3 + 0.5 * x + b[g]
  list(n = n, G = G, g = g, X = cbind(1, x), eta = eta,
       yp = rpois(n, exp(eta)), yt = eta + rt(n, 3) * 0.5,
       w = runif(n, 0.3, 2))
}

.jw_pair <- function(d, family, y, s = 0.7, phi = 1, phi2 = NULL,
                     weights = NULL) {
  lap <- tulpa_laplace(y, rep(1L, d$n), d$X,
                       re_list = list(list(idx = d$g, n_groups = d$G, sigma = s)),
                       family = family, phi = phi, phi2 = phi2,
                       weights = weights, tol = 1e-10, max_iter = 200L)
  jt <- tulpa_nested_laplace_joint(
    responses = list(y = list(y = y, X = d$X, family = family, phi = phi,
                              weights = weights, phi2 = phi2)),
    prior = list(list(type = "iid", obs_idx = d$g, n_units = d$G,
                      sigma_grid = s)),
    hyperprior = "flat",
    control = list(axis_refine = "none", tol = 1e-10, max_iter = 200L,
                   progress = FALSE, diagnose_k = FALSE))
  p <- ncol(d$X)
  list(lap_lm = lap$log_marginal, jt_lm = jt$log_marginal[1L],
       lap_mode = lap$mode[seq_len(p + d$G)],
       jt_mode = c(jt$modes[1L, seq_len(p)],
                   s * jt$modes[1L, p + seq_len(d$G)]))
}

.jw_expect_same <- function(r) {
  expect_equal(r$jt_lm, r$lap_lm, tolerance = 1e-6)
  expect_equal(unname(r$jt_mode), unname(r$lap_mode), tolerance = 1e-6)
}

test_that("weighted poisson: the joint arm matches the conditional Laplace", {
  skip_if_fast()
  d <- .jw_data()
  plain <- .jw_pair(d, "poisson", d$yp)
  wtd <- .jw_pair(d, "poisson", d$yp, weights = d$w)
  .jw_expect_same(plain)
  .jw_expect_same(wtd)
  expect_gt(abs(wtd$jt_lm - plain$jt_lm), 1)
})

test_that("student-t phi2: the joint arm matches the conditional Laplace", {
  skip_if_fast()
  d <- .jw_data()
  df3 <- .jw_pair(d, "t", d$yt, phi = 0.5, phi2 = 3)
  df10 <- .jw_pair(d, "t", d$yt, phi = 0.5, phi2 = 10)
  .jw_expect_same(df3)
  .jw_expect_same(df10)
  .jw_expect_same(.jw_pair(d, "t", d$yt, phi = 0.5, phi2 = 3, weights = d$w))
  expect_gt(abs(df3$jt_lm - df10$jt_lm), 1e-3)
})

test_that("the joint arm validates weights and phi2", {
  d <- .jw_data()
  arm <- list(y = d$yp, X = d$X, family = "poisson")
  blk <- list(list(type = "iid", obs_idx = d$g, n_units = d$G, sigma_grid = 1))
  fit <- function(a) tulpa_nested_laplace_joint(list(y = a), blk,
                                               control = list(progress = FALSE))
  expect_error(fit(c(arm, list(weights = rep(-1, d$n)))), "non-negative")
  expect_error(fit(c(arm, list(weights = 1))), "length")
  expect_error(fit(c(arm, list(phi2 = 3))), "second dispersion")
  expect_error(fit(utils::modifyList(arm, list(family = "tweedie",
                                               y = abs(d$yt)))),
               "variance power")
})

test_that("tulpa() integrates a weighted gaussian's variance on the joint door", {
  skip_if_fast()
  d <- .jw_data()
  dat <- data.frame(y = d$eta + rnorm(d$n, 0, 0.4), x = d$X[, 2],
                    g = factor(d$g))
  f1 <- tulpa(y ~ x + (1 | g), dat, family = "gaussian",
              weights = rep(1, d$n), mode = "structured")
  f0 <- tulpa(y ~ x + (1 | g), dat, family = "gaussian", mode = "structured")
  expect_identical(f1$backend, "nested_laplace_joint")
  expect_equal(coef(f1), coef(f0), tolerance = 1e-8)
  f2 <- tulpa(y ~ x + (1 | g), dat, family = "gaussian",
              weights = d$w, mode = "structured")
  expect_identical(f2$backend, "nested_laplace_joint")
  expect_false(isTRUE(all.equal(coef(f2), coef(f0))))
})
