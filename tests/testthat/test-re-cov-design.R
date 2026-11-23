# The random-effect covariance integrator runs on the joint nested-Laplace
# door, which lays the outer grid at the posterior. A scale's posterior on its
# log coordinate reaches several SDs of the mode's Gaussian into its left tail;
# these tests hold the default read to dense references.

skip_on_cran()

test_that("a slope SD's left tail past the 4.5-SD span is integrated", {
  # (1 + x | g), 50 binary groups x 6. The log slope SD's posterior has a left
  # tail heavier than the Gaussian at the mode: the 2.5% bound of a dense
  # (log sigma_1, log sigma_2, atanh rho) tensor (0.05 / 0.15 / 0.15 apart,
  # about a million cells trimmed to all but 1e-9 of each marginal) is 0.0116.
  set.seed(4)
  gg <- rep(1:50, each = 6); xx <- rnorm(300)
  b0 <- rnorm(50, 0, 1); b1 <- rnorm(50, 0, 0.7)
  yy <- rbinom(300, 1, plogis(-0.3 + 0.8 * xx + b0[gg] + b1[gg] * xx))
  rt <- list(idx = gg, n_groups = 50L, n_coefs = 2L, Z = cbind(1, xx),
             correlated = TRUE)
  fit <- suppressWarnings(tulpa_re_cov_nested(
    yy, NULL, cbind(1, xx), rt, family = "binomial",
    control = list(n_threads = 2L, diagnose_k = FALSE)))
  p <- fit$posterior
  lo <- p$ci_lo[p$parameter == "sigma_2"]
  expect_lt(abs(log(lo / 0.0116)), 0.25)
  hi <- p$ci_hi[p$parameter == "sigma_2"]
  expect_lt(abs(log(hi / 0.880)), 0.1)
})

test_that("a within-cell coupling whose zeroed precision is near singular is read", {
  set.seed(29)
  gg <- rep(1:50, each = 6); xx <- rnorm(300)
  b0 <- rnorm(50, 0, 1); b1 <- rnorm(50, 0, 0.7)
  yy <- rbinom(300, 1, plogis(-0.3 + 0.8 * xx + b0[gg] + b1[gg] * xx))
  rt <- list(idx = gg, n_groups = 50L, n_coefs = 2L, Z = cbind(1, xx),
             correlated = TRUE)
  fit <- suppressWarnings(tulpa_re_cov_nested(
    yy, NULL, cbind(1, xx), rt, family = "binomial",
    control = list(n_threads = 2L, diagnose_k = FALSE)))
  p <- fit$posterior
  expect_true(all(is.finite(c(p$ci_lo, p$median, p$ci_hi))))
  expect_true(all(p$ci_lo <= p$median & p$median <= p$ci_hi))
})

test_that("a (1 | g) scale's default interval matches a dense reference", {
  set.seed(20261102)
  G <- 40L; npg <- 8L; N <- G * npg
  g <- rep(seq_len(G), each = npg)
  x <- rnorm(N)
  u <- rnorm(G, sd = 0.8)
  y <- rbinom(N, 1, plogis(-0.4 + 0.9 * x + u[g]))
  X <- cbind(1, x)
  def <- tulpa_re_cov_nested(y, NULL, X, list(idx = g, n_groups = G),
                             control = list(diagnose_k = FALSE))$posterior
  # The reference: the same model on a stated 401-node log-SD axis over
  # [0.01, 4], every cell solved.
  blk <- list(type = "miid", obs_idx = g, n_groups = G, n_fields = 1L,
              field_weight = list(list(rep(1, N))),
              logchol_prior = re_cov_pc_lkj_prior(1L),
              logchol_grid = matrix(seq(log(0.01), log(4), length.out = 401L),
                                    ncol = 1L, dimnames = list(NULL, "L11")))
  ref <- tulpa_nested_laplace_joint(
    list(y = list(y = y, X = X, family = "binomial", n_trials = rep(1L, N))),
    prior = list(blk),
    control = list(progress = FALSE, diagnose_k = FALSE, axis_refine = "none",
                   var_of_means_consistency = FALSE, prune = FALSE,
                   auto_recenter = FALSE))$hyper_derived
  r <- ref[ref$parameter == "b1.sigma_1", ]
  d <- def[def$parameter == "sigma_1", ]
  expect_lt(abs(d$ci_lo - r$ci_lo), 0.005)
  expect_lt(abs(d$ci_hi - r$ci_hi), 0.02)
  expect_lt(abs(d$median - r$median), 0.002)
  expect_lt(abs(d$sd / r$sd - 1), 0.01)
})

test_that("a two-term covariance matches its dense 4-axis reference", {
  skip_if_not_slow()
  # (1 + x | g) + (1 | h), 40 g-groups and 25 crossed h-groups, 480 binary
  # rows, seed 21. The reference integrates (log sigma_1, log sigma_2,
  # atanh rho, log sigma_h) on lattices of halving spacing, down to
  # 0.1 / 0.125 / 0.15 / 0.1, each level solved within 16 nats of the
  # maximum (3.7 million cells); consecutive levels agree to 0.003 on every
  # median and to 0.004 on every interval end, and each axis's outermost level
  # holds under 2e-4 of the mass. Reference: g.sigma_2 median 0.192
  # [0.0089, 0.596], g.rho_12 -0.099 [-0.820, 0.742], g.sigma_1 0.600
  # [0.215, 0.974], h.sigma_1 0.551 [0.212, 0.935], log evidence -308.944.
  set.seed(21L)
  G <- 40L; H <- 25L; npg <- 12L; N <- G * npg
  g <- rep(seq_len(G), each = npg); h <- sample.int(H, N, replace = TRUE)
  x <- rnorm(N); X <- cbind(1, x)
  Sg <- matrix(c(0.8^2, 0.4 * 0.8 * 0.5, 0.4 * 0.8 * 0.5, 0.5^2), 2)
  ug <- t(t(chol(Sg)) %*% matrix(rnorm(2 * G), 2)); uh <- rnorm(H, 0, 0.6)
  y <- rbinom(N, 1L, plogis(as.numeric(X %*% c(-0.2, 0.6)) +
                              rowSums(X * ug[g, ]) + uh[h]))
  re <- list(list(idx = g, n_groups = G, n_coefs = 2L, Z = X, label = "g"),
             list(idx = h, n_groups = H, label = "h"))
  fit <- suppressWarnings(tulpa_re_cov_nested(
    y, NULL, X, re, control = list(diagnose_k = FALSE)))
  p <- fit$posterior
  at <- function(nm) p[p$parameter == nm, ]
  ref <- list(g.sigma_1 = c(0.600, 0.215, 0.974),
              g.sigma_2 = c(0.192, 0.0089, 0.596),
              g.rho_12 = c(-0.099, -0.820, 0.742),
              h.sigma_1 = c(0.551, 0.212, 0.935))
  for (nm in names(ref)) {
    r <- at(nm)
    expect_lt(abs(r$median - ref[[nm]][1]), 0.1 * r$sd, label = nm)
    expect_lt(abs(r$ci_hi - ref[[nm]][3]), 0.15 * r$sd, label = nm)
  }
  expect_lt(abs(log(at("g.sigma_2")$ci_lo / 0.0089)), 0.5)
  expect_lt(abs(fit$log_evidence - (-308.944)), 0.1)
})
