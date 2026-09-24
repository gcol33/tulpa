# A ModelData sampler fit stores its random effects NON-CENTERED: the `re[...]`
# columns are the standardized `z`, and the effects are `b = sigma * L z`
# (`re_term_group_effect()`, inst/include/tulpa/re_term_prior.h). ranef() read
# `z` as `b` (gcol33/tulpa#866) and VarCorr() dropped the sampled `L_re`
# correlation (gcol33/tulpa#867).

test_that(".re_chol_from_raw is build_L_from_raw: tanh stick-breaking rows", {
  raw <- rbind(c(0.3, -1.1, 0.7), c(-2, 0.1, 1.5))
  L <- .re_chol_from_raw(raw, 3L)
  for (s in 1:2) {
    Ls <- L[s, , ]
    expect_equal(Ls[upper.tri(Ls)], rep(0, 3))
    expect_equal(rowSums(Ls^2), rep(1, 3))          # unit-norm rows: a correlation
    expect_equal(Ls[2, 1], tanh(raw[s, 1]))         # the 2x2 correlation itself
    expect_equal(Ls[3, 1], tanh(raw[s, 2]))
    expect_equal(Ls[3, 2], tanh(raw[s, 3]) * sqrt(1 - tanh(raw[s, 2])^2))
  }
})

test_that("a scalar term's z is scaled by its own sigma, per draw", {
  tail <- cbind(`re[1]` = c(1, -2), `re[2]` = c(0.5, 3),
                log_sigma_re = log(c(2, 0.1)))
  b <- .re_noncentered_effects(tail)
  expect_equal(unname(b), cbind(c(2, -0.2), c(1, 0.3)))
  expect_equal(colnames(b), c("re[1]", "re[2]"))
})

test_that("several scalar terms each read their own sigma", {
  tail <- cbind(`re[t1.g1]` = 1, `re[t1.g2]` = 2, `re[t2.g1]` = 1,
                `log_sigma_re[t1]` = log(3), `log_sigma_re[t2]` = log(5))
  expect_equal(as.numeric(.re_noncentered_effects(tail)), c(3, 6, 5))
})

test_that("a correlated term mixes z through L before scaling", {
  set.seed(1)
  S <- 4L
  z <- matrix(rnorm(S * 4), S)              # g1.c1, g1.c2, g2.c1, g2.c2
  raw <- rnorm(S); ls <- matrix(rnorm(S * 2, 0, 0.3), S)
  tail <- cbind(z, raw, ls)
  colnames(tail) <- c("re[t1.g1.c1]", "re[t1.g1.c2]", "re[t1.g2.c1]",
                      "re[t1.g2.c2]", "L_re[t1.1]",
                      "log_sigma_re[t1.c1]", "log_sigma_re[t1.c2]")
  b <- .re_noncentered_effects(tail)
  for (s in seq_len(S)) {
    L <- matrix(c(1, tanh(raw[s]), 0, sqrt(1 - tanh(raw[s])^2)), 2)
    for (g in 1:2) {
      zg <- z[s, (2 * g - 1):(2 * g)]
      expect_equal(unname(b[s, (2 * g - 1):(2 * g)]),
                   exp(ls[s, ]) * as.numeric(L %*% zg))
    }
  }
})

test_that("a missing scale column refuses rather than returning z", {
  expect_null(.re_noncentered_effects(cbind(`re[1]` = 1, `re[2]` = 2)))
})

test_that("VarCorr() reports the sampled L_re correlation, not an identity", {
  set.seed(2)
  S <- 500L
  raw <- rnorm(S, -0.6, 0.2)
  ls  <- cbind(rnorm(S, log(1.2), 0.1), rnorm(S, log(0.5), 0.1))
  draws <- cbind(`(Intercept)` = rnorm(S), x = rnorm(S),
                 ls, raw)
  colnames(draws)[3:5] <- c("log_sigma_re[t1.c1]", "log_sigma_re[t1.c2]",
                            "L_re[t1.1]")
  fit <- structure(list(
    draws = draws, n_fixed = 2L,
    re_layout = list(list(group_var = "g", n_coefs = 2L,
                          coef_labels = c("(Intercept)", "x")))),
    class = "tulpa_fit")
  vc <- VarCorr(fit)
  expect_equal(vc$source, rep("sampled", 2))
  expect_equal(vc$sd, unname(colMeans(exp(ls))))   # still the posterior-mean SD
  S_g <- attr(vc, "cov")$g
  expect_equal(attr(S_g, "correlation")[1, 2], mean(tanh(raw)))
  expect_equal(S_g[1, 2], mean(tanh(raw)) * prod(vc$sd))
})

test_that("ranef() on an hmc fit reports b, the effects eta is built from", {
  skip_on_cran()
  set.seed(1)
  J <- 12L; n <- 180L
  gi <- sample(J, n, TRUE); x <- rnorm(n)
  B <- cbind(rnorm(J, 0, 0.8), rnorm(J, 0, 0.4))
  d <- data.frame(y = 1 + 0.5 * x + B[gi, 1] + B[gi, 2] * x + rnorm(n, 0, 0.3),
                  x = x, g = factor(gi))
  fit <- tulpa(y ~ x + (1 + x | g), d, phi = 0.09, mode = "hmc",
               control = list(n_chains = 1L, n_iter = 300L, warmup = 150L,
                              seed = 1L))
  b <- .re_coef_draws(fit)
  eta <- .tulpa_eta_draws(fit)
  beta <- .fixed_draws_mat(fit)
  expect_equal(eta,
               beta %*% t(fit$model_inputs$X) + as.matrix(b %*% .tulpa_re_map(fit)),
               ignore_attr = TRUE, tolerance = 1e-10)
  re <- ranef(fit)
  expect_equal(re$estimate, unname(colMeans(b)))
  expect_false(isTRUE(all.equal(unname(colMeans(b)),
                                unname(colMeans(fit$draws[, colnames(b)])))))
  vc <- attr(VarCorr(fit), "cov")$g
  expect_true(abs(attr(vc, "correlation")[1, 2]) > 0)
})
