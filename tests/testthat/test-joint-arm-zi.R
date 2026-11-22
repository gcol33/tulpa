# Zero inflation on the joint nested-Laplace door. The structural-zero logit is
# a second arm on the ziformula's design, coupled row by row to the response's
# arm through the "zero_inflated" cell coupling, which evaluates the shared
# mixture density and its 2 x 2 curvature. At one fixed random-intercept scale
# the joint door and the conditional Laplace backend (the two-process spec)
# solve the same inner problem -- the joint `iid` block is the non-centred form
# of the same random intercept, and the Laplace approximation is invariant
# under that linear change of variables -- so the log marginal and the mode
# agree.

.jz_data <- function(seed = 11L, G = 12L, per = 10L) {
  set.seed(seed)
  n <- G * per
  g <- rep(seq_len(G), each = per); x <- rnorm(n); w <- rnorm(n)
  b <- rnorm(G, 0, 0.6)
  eta <- 0.3 + 0.5 * x + b[g]
  struct0 <- runif(n) < plogis(-0.5 + 0.8 * w)
  list(n = n, G = G, g = g, x = x, w = w, X = cbind(1, x), Xz = cbind(1, w),
       yp = ifelse(struct0, 0, rpois(n, exp(eta))),
       yb = ifelse(struct0, 0, rbinom(n, 4L, plogis(eta))))
}

.jz_pair <- function(d, family, y, nt, s = 0.7, force_sparse = FALSE) {
  lap <- tulpa_laplace(y, nt, d$X,
                       re_list = list(list(idx = d$g, n_groups = d$G, sigma = s)),
                       family = family, X_zi = d$Xz, zi_prior_sd = 2.5,
                       tol = 1e-10, max_iter = 200L)
  responses <- .joint_arm_zi(
    list(y = list(y = y, X = d$X, family = family, n_trials = nt)),
    d$Xz, zi_prior_sd = 2.5)
  jt <- tulpa_nested_laplace_joint(
    responses = responses,
    prior = list(list(type = "iid", obs_idx = d$g, n_units = d$G,
                      sigma_grid = s)),
    cell_coupling = "zero_inflated", hyperprior = "flat",
    control = list(axis_refine = "none", tol = 1e-10, max_iter = 200L,
                   progress = FALSE, diagnose_k = FALSE,
                   force_sparse = force_sparse))
  p <- 2L * ncol(d$X)
  list(lap_lm = lap$log_marginal, jt_lm = jt$log_marginal[1L],
       lap_mode = lap$mode[seq_len(p + d$G)],
       jt_mode = c(jt$modes[1L, seq_len(p)], s * jt$modes[1L, p + seq_len(d$G)]))
}

.jz_expect_same <- function(r) {
  expect_equal(r$jt_lm, r$lap_lm, tolerance = 1e-6)
  expect_equal(unname(r$jt_mode), unname(r$lap_mode), tolerance = 1e-6)
}

test_that("the zero-inflated coupling is registered", {
  expect_true(cpp_cell_coupling_registry_has("zero_inflated"))
})

test_that("zi poisson and zi binomial: the joint door matches the conditional Laplace", {
  skip_if_fast()
  d <- .jz_data()
  .jz_expect_same(.jz_pair(d, "poisson", d$yp, rep(1L, d$n)))
  .jz_expect_same(.jz_pair(d, "binomial", d$yb, rep(4L, d$n)))
})

test_that("the zero-inflated coupling matches on the sparse joint Newton", {
  skip_if_fast()
  d <- .jz_data()
  .jz_expect_same(.jz_pair(d, "poisson", d$yp, rep(1L, d$n),
                           force_sparse = TRUE))
  big <- .jz_data(seed = 3L, G = 240L, per = 4L)
  .jz_expect_same(.jz_pair(big, "poisson", big$yp, rep(1L, big$n)))
})

test_that("tulpa(): a conditioned zi fit on the joint door matches mode = 'laplace'", {
  skip_if_fast()
  d <- .jz_data()
  dat <- data.frame(y = d$yp, x = d$x, w = d$w, g = factor(d$g))
  ctl <- list(tol = 1e-10, max_iter = 200L)
  fl <- tulpa(y ~ x + (1 | g), dat, family = "poisson", ziformula = ~ w,
              sigma_re = 0.7, mode = "laplace", control = ctl)
  fj <- tulpa(y ~ x + (1 | g), dat, family = "poisson", ziformula = ~ w,
              sigma_re = 0.7, mode = "nested_laplace_joint",
              control = c(ctl, list(axis_refine = "none")))
  expect_identical(fj$backend, "nested_laplace_joint")
  expect_identical(names(coef(fj)), names(coef(fl)))
  expect_equal(coef(fj), coef(fl), tolerance = 1e-6)
  expect_equal(fj$log_marginal[1L], fl$log_marginal, tolerance = 1e-6)
  expect_identical(fj$N, d$n)
})

test_that("the joint door refuses observation weights with a zero-inflation formula", {
  d <- .jz_data()
  dat <- data.frame(y = d$yp, x = d$x, w = d$w, g = factor(d$g))
  expect_error(tulpa(y ~ x + (1 | g), dat, family = "poisson", ziformula = ~ w,
                     weights = rep(1, d$n), mode = "nested_laplace_joint"),
               "observation weights with a zero-inflation formula")
})

test_that("a block stated for one arm is widened to the zero arm", {
  blk <- list(type = "miid", obs_idx = 1:4, n_groups = 2L, n_fields = 2L,
              field_weight = list(list(rep(1, 4)), list(c(1, 2, 3, 4))))
  wb <- .joint_block_widen_arms(blk, 2L)
  expect_identical(wb$obs_idx, 1:4)
  expect_length(wb$field_weight[[2L]], 2L)
  expect_identical(wb$field_weight[[2L]][[2L]], c(1, 2, 3, 4))
  hs <- list(type = "hsgp", phi = list(diag(3)), n_obs_per_arm = 3L)
  wh <- .joint_block_widen_arms(hs, 2L)
  expect_length(wh$phi, 2L)
  expect_identical(wh$n_obs_per_arm, c(3L, 3L))
})

test_that("tulpa(): zi poisson with a random intercept recovers the truth on the joint door", {
  skip_on_cran()
  set.seed(2026)
  G <- 40L; per <- 25L; n <- G * per
  g <- rep(seq_len(G), each = per); x <- rnorm(n); w <- rnorm(n)
  truth <- c("(Intercept)" = 0.4, x = 0.6, "zi_(Intercept)" = -0.7, zi_w = 0.9)
  sigma <- 0.5
  b <- rnorm(G, 0, sigma)
  eta <- truth[1] + truth[2] * x + b[g]
  struct0 <- runif(n) < plogis(truth[3] + truth[4] * w)
  dat <- data.frame(y = ifelse(struct0, 0, rpois(n, exp(eta))), x = x, w = w,
                    g = factor(g))
  fit <- tulpa(y ~ x + (1 | g), dat, family = "poisson", ziformula = ~ w,
               mode = "nested_laplace_joint")
  expect_identical(fit$backend, "nested_laplace_joint")
  ci <- confint(fit)
  expect_true(all(ci[names(truth), 1] < truth & truth < ci[names(truth), 2]))
  expect_lt(max(abs(coef(fit)[names(truth)] - truth)), 0.25)
  vc <- VarCorr(fit)
  expect_lt(abs(vc$sd[1L] - sigma), 0.2)
})

test_that("tulpa(): an uncorrelated slope term widens its row weight to the zero arm", {
  skip_on_cran()
  d <- .jz_data(seed = 5L, G = 20L, per = 15L)
  dat <- data.frame(y = d$yp, x = d$x, w = d$w, g = factor(d$g))
  fit <- tulpa(y ~ x + (1 + x || g), dat, family = "poisson", ziformula = ~ w,
               mode = "nested_laplace_joint")
  expect_identical(fit$backend, "nested_laplace_joint")
  expect_true(all(is.finite(coef(fit))))
  expect_identical(length(coef(fit)), 4L)
})
