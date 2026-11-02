# predict() / posterior_predict() at new data condition on the fitted groups:
# a group the fit saw keeps its effect, a group it did not see draws one from
# the term's covariance, and re_form = NA is the population level.

skip_on_cran()

.prc_sim <- fixture_once(function() {
  set.seed(31)
  G <- 30L; npg <- 10L; N <- G * npg
  g <- rep(seq_len(G), each = npg); x <- rnorm(N)
  b <- rnorm(G, 0, 0.8)
  eta <- 0.3 + 0.5 * x + b[g]
  list(d = data.frame(y = rpois(N, exp(eta)), x = x, g = factor(g)),
       b = b, G = G)
})

.prc_fit <- fixture_once(function() {
  tulpa(y ~ x + (1 | g), data = .prc_sim()$d, family = "poisson",
        mode = "eb")
})

test_that("predict() at new data reads the fitted groups' effects", {
  sim <- .prc_sim(); fit <- .prc_fit()
  d <- sim$d
  # The training rows as new data reproduce the conditional fitted predictor.
  expect_equal(predict(fit, newdata = d),
               tulpa:::.tulpa_fitted_linpred(fit, "conditional", "t")$eta,
               tolerance = 1e-10)
  expect_equal(predict(fit, newdata = d), predict(fit), tolerance = 1e-10)
  # re_form = NA is the fixed effects alone.
  X <- cbind(1, d$x)
  expect_equal(predict(fit, newdata = d, re_form = NA),
               as.numeric(X %*% coef(fit)), tolerance = 1e-10)
  expect_equal(predict(fit, newdata = d, re_form = ~0),
               predict(fit, newdata = d, re_form = NA))

  # Against the simulated truth: at new x values of the fitted groups the
  # conditional prediction follows each group's own level.
  nd <- data.frame(x = rep(0, sim$G), g = factor(seq_len(sim$G)))
  truth <- 0.3 + sim$b
  cond <- predict(fit, newdata = nd)
  pop  <- predict(fit, newdata = nd, re_form = NA)
  expect_gt(cor(cond, truth), 0.9)
  expect_lt(sqrt(mean((cond - truth)^2)), 0.5 * sqrt(mean((pop - truth)^2)))
})

test_that("a new group predicts at the population mean with the between-group spread", {
  fit <- .prc_fit()
  nd <- data.frame(x = c(-1, 0, 1), g = factor("unseen"))
  p_new <- predict(fit, newdata = nd, se.fit = TRUE)
  p_pop <- predict(fit, newdata = nd, re_form = NA, se.fit = TRUE)
  expect_equal(p_new$fit, p_pop$fit, tolerance = 1e-10)
  # se^2 of a new group = fixed-effect variance + the group variance.
  s2 <- VarCorr(fit)$sd[1]^2
  expect_equal(p_new$se.fit^2, p_pop$se.fit^2 + s2, tolerance = 0.06)
  # The interval is a pinned read: the same on every call.
  expect_identical(predict(fit, newdata = nd, se.fit = TRUE), p_new)

  # posterior_predict(): rows of one new group share its effect.
  nd2 <- data.frame(x = c(0, 0), g = factor(c("a", "a")))
  e <- posterior_linpred(fit, newdata = nd2, ndraws = 2000, seed = 4)
  expect_equal(e[, 1], e[, 2])
  expect_equal(stats::var(e[, 1]), p_pop$se.fit[2]^2 + s2, tolerance = 0.1)
})

test_that("a sampled covariance gives each replicate its own new-group variance", {
  sim <- .prc_sim()
  fit <- suppressWarnings(tulpa(y ~ x + (1 | g), data = sim$d,
                                family = "poisson", mode = "laplace",
                                control = list(re_cov = "gibbs", n_iter = 2000L,
                                               warmup = 500L, seed = 2L)))
  nd <- data.frame(x = 0, g = factor("unseen"))
  e <- posterior_linpred(fit, newdata = nd, seed = 5)
  s2 <- vapply(fit$Sigma_draws, function(S) S[1, 1], numeric(1))
  vb <- stats::var(fit$draws[, "(Intercept)"])
  # var(eta_new) = var(beta_0) + E[Sigma_11], the draws read at their own Sigma.
  expect_equal(stats::var(e[, 1]), vb + mean(s2), tolerance = 0.12)
  # A fitted group's draws are its sampled effect plus the fixed effects.
  nd1 <- data.frame(x = 0, g = factor("3", levels = levels(sim$d$g)))
  e1 <- posterior_linpred(fit, newdata = nd1)
  expect_equal(as.numeric(e1[, 1]),
               as.numeric(fit$draws[, "(Intercept)"] + fit$re[, 3]))
})

test_that("re_form selects terms, and an unknown grouping is named", {
  set.seed(8)
  N <- 240L
  d <- data.frame(x = rnorm(N), g = factor(sample(1:12, N, TRUE)),
                  h = factor(sample(1:8, N, TRUE)))
  d$y <- rpois(N, exp(0.2 + 0.4 * d$x + rnorm(12, 0, 0.6)[d$g] +
                        rnorm(8, 0, 0.6)[d$h]))
  fit <- tulpa(y ~ x + (1 | g) + (1 | h), data = d, family = "poisson",
               mode = "eb")
  nd <- d[1:5, ]
  full <- predict(fit, newdata = nd)
  pop  <- predict(fit, newdata = nd, re_form = NA)
  only_g <- predict(fit, newdata = nd, re_form = ~ (1 | g))
  only_h <- predict(fit, newdata = nd, re_form = ~ (1 | h))
  expect_equal(only_g + only_h - pop, full, tolerance = 1e-10)
  expect_error(predict(fit, newdata = nd, re_form = ~ (1 | site)), "site")
  # The population prediction needs no grouping column.
  expect_silent(predict(fit, newdata = data.frame(x = 0), re_form = NA))
})

test_that("a fit that integrates its groups out refuses the conditional read and gives the population", {
  sim <- .prc_sim()
  fit <- tulpa(y ~ x + (1 | g), data = sim$d, family = "poisson",
               mode = "laplace", control = list(re_cov = "aghq", n_quad = 3L))
  nd <- data.frame(x = 0, g = factor("2", levels = levels(sim$d$g)))
  expect_error(predict(fit, newdata = nd), "re_form = NA")
  expect_error(posterior_predict(fit, newdata = nd, ndraws = 10), "re_form = NA")
  expect_equal(predict(fit, newdata = nd, re_form = NA),
               sum(coef(fit) * c(1, 0)))
  # A group the fit did not see needs only the covariance.
  expect_length(predict(fit, newdata = data.frame(x = 0, g = "new"),
                        se.fit = TRUE)$fit, 1L)
})

test_that("K-fold scores a held-out row at its group's effect", {
  sim <- .prc_sim(); fit <- .prc_fit()
  kf <- tulpa_kfold(fit, data = sim$d, K = 5L, seed = 1)
  lo <- suppressWarnings(loo::loo(pointwise_loglik(fit)))$estimates["elpd_loo", 1]
  # Every fold keeps rows of every group in training, so the held-out density
  # is the conditional one PSIS-LOO approximates. Scored at the population
  # level instead it reads `pop` nats lower.
  expect_lt(abs(kf$elpd_kfold - lo), 10)
  pop <- sum(vapply(seq_len(nrow(sim$d)), function(i) {
    eta <- sum(coef(fit) * c(1, sim$d$x[i]))
    stats::dpois(sim$d$y[i], exp(eta), log = TRUE)
  }, numeric(1)))
  expect_gt(kf$elpd_kfold - pop, 20)
})
