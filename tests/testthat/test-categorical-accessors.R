# Observation-level accessors on categorical fits (family = "multinomial" /
# "ordinal") and the observation record of a joint nested-Laplace fit. The class
# probabilities are scored against the baseline-category softmax and the
# cumulative-link differences written out by hand.

cat_data <- function(n = 240, seed = 1) {
  set.seed(seed)
  d <- data.frame(x = stats::rnorm(n))
  d$ycat <- factor(sample(c("a", "b", "c"), n, TRUE))
  d$yord <- factor(cut(0.6 * d$x + stats::rlogis(n), c(-Inf, -0.5, 0.8, Inf),
                       labels = c("lo", "mid", "hi")), ordered = TRUE)
  d
}

test_that("multinomial fitted / residuals / predict read the softmax by hand", {
  skip_on_cran()
  d <- cat_data()
  fit <- tulpa(ycat ~ x, d, family = "multinomial")
  expect_identical(nobs(fit), nrow(d))
  expect_equal(attr(logLik(fit), "nobs"), nrow(d))

  th <- as.numeric(fit$means)
  X <- cbind(1, d$x)
  E <- cbind(X %*% th[1:2], X %*% th[3:4], 0)
  P <- exp(E) / rowSums(exp(E))
  expect_equal(unname(fitted(fit)), P, tolerance = 1e-12)
  expect_identical(colnames(fitted(fit)), c("a", "b", "c"))
  expect_equal(unname(predict(fit)), unname(E[, 1:2]), tolerance = 1e-12)
  expect_equal(predict(fit, type = "response"), fitted(fit))
  expect_equal(unname(predict(fit, newdata = d[1:4, ], type = "response")),
               P[1:4, ], tolerance = 1e-12)

  Y <- outer(as.integer(d$ycat), 1:3, "==") * 1
  expect_equal(unname(residuals(fit, type = "response")), Y - P,
               tolerance = 1e-12)
  expect_equal(unname(residuals(fit)), (Y - P) / sqrt(P * (1 - P)),
               tolerance = 1e-10)

  se <- predict(fit, newdata = d[1:4, ], se.fit = TRUE)
  V <- vcov(fit)
  Xn <- X[1:4, ]
  expect_equal(unname(se$se.fit[, 1]),
               sqrt(rowSums((Xn %*% V[1:2, 1:2]) * Xn)), tolerance = 1e-10)
  pr <- predict(fit, newdata = d[1:4, ], type = "response", se.fit = TRUE)
  expect_true(all(pr$lower <= pr$fit & pr$fit <= pr$upper))
})

test_that("ordinal fitted / residuals / predict read the cumulative logit by hand", {
  skip_on_cran()
  d <- cat_data()
  fit <- tulpa(yord ~ x, d, family = "ordinal")
  expect_identical(nobs(fit), nrow(d))

  th <- as.numeric(fit$means)
  eta <- th[1] * d$x
  Fm <- stats::plogis(outer(-eta, th[2:3], "+"))
  P <- cbind(Fm, 1) - cbind(0, Fm)
  expect_equal(unname(fitted(fit)), P, tolerance = 1e-12)
  expect_identical(colnames(fitted(fit)), c("lo", "mid", "hi"))
  expect_equal(predict(fit), eta, tolerance = 1e-12)
  Y <- outer(as.integer(d$yord), 1:3, "==") * 1
  expect_equal(unname(residuals(fit, type = "response")), Y - P,
               tolerance = 1e-12)
})

test_that("categorical replicates are class draws, simulated as factors", {
  skip_on_cran()
  d <- cat_data()
  for (fam in c("multinomial", "ordinal")) {
    f <- if (fam == "multinomial") ycat ~ x else yord ~ x
    fit <- tulpa(f, d, family = fam)
    lev <- if (fam == "multinomial") c("a", "b", "c") else c("lo", "mid", "hi")
    yrep <- posterior_predict(fit, ndraws = 300, seed = 1)
    expect_equal(dim(yrep), c(300L, nrow(d)))
    expect_true(all(yrep %in% 1:3))
    expect_identical(attr(yrep, "levels"), lev)
    # Class frequencies of 72000 draws against the mean fitted probabilities.
    expect_equal(as.numeric(table(factor(yrep, 1:3))) / length(yrep),
                 unname(colMeans(fitted(fit))), tolerance = 0.03)

    s <- simulate(fit, nsim = 2, seed = 4)
    expect_equal(dim(s), c(nrow(d), 2L))
    expect_true(is.factor(s$sim_1))
    expect_identical(levels(s$sim_1), lev)
    expect_identical(is.ordered(s$sim_1), fam == "ordinal")
    expect_error(test_dispersion(fit, nsim = 5L), class(fit)[1])
    expect_error(pit_residuals(fit, nsim = 5L), "categorical")
  }
})

test_that("criteria on categorical fits score the log probability of each observed class", {
  skip_on_cran()
  skip_if_not_installed("loo")
  d <- cat_data()
  for (fam in c("multinomial", "ordinal")) {
    f <- if (fam == "multinomial") ycat ~ x else yord ~ x
    fit <- tulpa(f, d, family = fam)
    D <- fit$draws
    cls <- as.integer(if (fam == "multinomial") d$ycat else d$yord)
    prob_obs <- function(th) {
      P <- if (fam == "multinomial") {
        E <- cbind(th[1] + th[2] * d$x, th[3] + th[4] * d$x, 0)
        exp(E) / rowSums(exp(E))
      } else {
        Fm <- stats::plogis(outer(-th[1] * d$x, th[2:3], "+"))
        cbind(Fm, 1) - cbind(0, Fm)
      }
      log(P[cbind(seq_len(nrow(d)), cls)])
    }
    hand <- t(apply(D, 1L, prob_obs))

    expect_equal(tulpa:::.tulpa_pointwise_loglik(fit), hand, tolerance = 1e-10)
    expect_equal(cpo(fit), cpo(hand), tolerance = 1e-10)
    expect_equal(suppressWarnings(loo::waic(fit))$estimates,
                 suppressWarnings(loo::waic(hand))$estimates, tolerance = 1e-10)
    lo <- suppressWarnings(loo::loo(fit))
    expect_s3_class(lo, "psis_loo")
    expect_equal(lo$estimates, suppressWarnings(loo::loo(hand))$estimates,
                 tolerance = 1e-10)

    dc <- dic(fit)
    expect_equal(dc, dic(hand, loglik_at_mean = prob_obs(colMeans(D))),
                 tolerance = 1e-10)
    expect_true(is.finite(compare_models(a = fit, criterion = "waic")$elpd))
  }
})

test_that("a joint fit records its observations and refuses by class", {
  skip_on_cran()
  S <- 16L
  adj <- matrix(0L, S, S)
  for (i in seq_len(S - 1L)) adj[i, i + 1L] <- adj[i + 1L, i] <- 1L
  nb <- rowSums(adj)
  cols <- lapply(seq_len(S), function(i) which(adj[i, ] == 1L))
  set.seed(3)
  X1 <- cbind(1, stats::rnorm(S)); X2 <- cbind(1, stats::rnorm(S))
  fit <- tulpa_nested_laplace_joint(
    responses = list(
      occ = list(y = stats::rbinom(S, 1, 0.5), n_trials = rep(1L, S), X = X1,
                 spatial_idx = seq_len(S), family = "binomial"),
      cover = list(y = stats::rnorm(S), n_trials = rep(1L, S), X = X2,
                   spatial_idx = seq_len(S), family = "gaussian")),
    prior = list(type = "icar", n_spatial_units = S,
                 adj_row_ptr = c(0L, cumsum(nb)),
                 adj_col_idx = unlist(cols) - 1L,
                 n_neighbors = as.integer(nb), sigma_grid = c(0.5, 1.0)))
  expect_identical(nobs(fit), 2L * S)
  expect_identical(names(fit$y), c("occ", "cover"))
  for (acc in list(fitted, residuals, predict, posterior_predict)) {
    msg <- tryCatch(acc(fit), error = conditionMessage)
    expect_match(msg, "tulpa_nested_laplace_joint")
    expect_no_match(msg, "refit")
  }
})

test_that("simulate() carries the stats::simulate seed attribute", {
  skip_on_cran()
  set.seed(2)
  d <- data.frame(x = stats::rnorm(80))
  d$y <- stats::rpois(80, exp(0.3 + 0.4 * d$x))
  fit <- tulpa(y ~ x, d, family = "poisson", mode = "laplace")

  s <- simulate(fit, nsim = 2, seed = 11)
  expect_identical(as.numeric(attr(s, "seed")), 11)
  expect_identical(attr(attr(s, "seed"), "kind"), as.list(RNGkind()))
  expect_identical(simulate(fit, nsim = 2, seed = attr(s, "seed")), s)

  set.seed(99)
  state <- .Random.seed
  s0 <- simulate(fit, nsim = 2)
  expect_identical(attr(s0, "seed"), state)
  assign(".Random.seed", attr(s0, "seed"), envir = globalenv())
  expect_identical(simulate(fit, nsim = 2), s0)
})

test_that("bayes_R2() and test_dispersion() refuse a categorical fit with the same message", {
  skip_on_cran()
  d <- cat_data()
  fit_m <- tulpa(ycat ~ x, d, family = "multinomial")
  fit_o <- tulpa(yord ~ x, d, family = "ordinal")
  for (fit in list(fit_m, fit_o)) {
    err_r2 <- tryCatch(bayes_R2(fit), error = conditionMessage)
    err_td <- tryCatch(test_dispersion(fit), error = conditionMessage)
    expect_match(err_r2, "categorical response")
    expect_match(err_td, "categorical response")
  }
})

test_that("pp_check() on a categorical fit plots class-replicate bars instead of erroring", {
  skip_on_cran()
  skip_if_not_installed("bayesplot")
  d <- cat_data()
  fit_m <- tulpa(ycat ~ x, d, family = "multinomial")
  fit_o <- tulpa(yord ~ x, d, family = "ordinal")
  for (fit in list(fit_m, fit_o)) {
    p <- pp_check(fit, ndraws = 20)
    expect_s3_class(p, "ggplot")
  }
  expect_identical(
    utils::getS3method("pp_check", "tulpa_categorical", envir = asNamespace("tulpa")),
    pp_check.tulpa_categorical
  )
})
