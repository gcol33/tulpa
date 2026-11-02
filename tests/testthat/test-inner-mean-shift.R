# The inner Laplace is a Gaussian at the joint mode of the latent vector. Its
# leading-order mean, mode + Sigma A' (l3 * s) / 2, is what every coordinate's
# gamma_1 + gamma_3 / 2 location term adds up to, and is what a nested fit over
# a random-effect covariance centres each node at.

skip_on_cran()

test_that("the mean shift is sigma_i (gamma_1 + gamma_3 / 2) at every probed index", {
  set.seed(3)
  G <- 12L; npg <- 6L; N <- G * npg
  g <- rep(seq_len(G), each = npg); x <- rnorm(N)
  y <- rpois(N, exp(0.1 + 0.4 * x + rnorm(G, 0, 0.6)[g]))
  r <- tulpa_laplace(
    y = y, n_trials = rep(1L, N), X = cbind(1, x),
    re_list = list(list(idx = g, n_groups = G, n_coefs = 2L,
                        Z = cbind(1, x), L = diag(c(0.7, 0.4)))),
    family = "poisson", compute_skew = TRUE, skew_idx = 1:8,
    return_joint_hessian = TRUE)
  expect_length(r$inner_mean_shift, 2L + 2L * G)
  sg <- sqrt(diag(solve(as.matrix(r$H_latent))))[1:8]
  expect_equal(r$inner_mean_shift[1:8],
               sg * (r$inner_skew_gamma1 + r$inner_skew / 2),
               tolerance = 1e-8)
})

test_that("the centred node reaches the exact posterior mean (3-D quadrature)", {
  y <- c(0, 1, 0, 2, 3, 1, 4, 2)
  g <- rep(1:2, each = 4)
  sig <- 0.8; sb <- 2.5
  r <- tulpa_laplace(
    y = y, n_trials = rep(1L, 8), X = matrix(1, 8, 1),
    re_list = list(list(idx = g, n_groups = 2L, sigma = sig)),
    family = "poisson", beta_prior = list(mean = 0, sd = sb),
    compute_skew = TRUE, skew_idx = 1L, return_joint_hessian = TRUE,
    tol = 1e-12)
  m <- r$mode
  sd0 <- sqrt(diag(solve(as.matrix(r$H_latent))))
  ax <- function(c, s) seq(c - 7 * s, c + 7 * s, length.out = 81)
  grid <- expand.grid(b0 = ax(m[1], sd0[1]), u1 = ax(m[2], sd0[2]),
                      u2 = ax(m[3], sd0[3]))
  S <- tapply(y, g, sum); n <- tapply(y, g, length)
  lp <- with(grid, {
    e1 <- b0 + u1; e2 <- b0 + u2
    S[1] * e1 - n[1] * exp(e1) + S[2] * e2 - n[2] * exp(e2) -
      0.5 * (b0 / sb)^2 - 0.5 * (u1 / sig)^2 - 0.5 * (u2 / sig)^2
  })
  w <- exp(lp - max(lp)); w <- w / sum(w)
  exact <- colSums(as.matrix(grid) * w)
  # Mode error 0.074 / 0.027 / 0.020; the centred node is within 1e-3.
  expect_gt(max(abs(m - exact)), 0.05)
  expect_lt(max(abs(m + r$inner_mean_shift - exact)), 2e-3)
})

test_that("a nested random-slope fit centres its nodes, and skew_correct = FALSE keeps the mode", {
  set.seed(20260716)
  G <- 40L; npg <- 12L; N <- G * npg
  grp <- rep(seq_len(G), each = npg)
  x <- rnorm(N)
  S <- matrix(c(0.49, 0.14, 0.14, 0.25), 2)
  u <- t(t(chol(S)) %*% matrix(rnorm(2 * G), 2))
  y <- rpois(N, exp(0.2 + 0.5 * x + u[grp, 1] + u[grp, 2] * x))
  d <- data.frame(y = y, x = x, g = factor(grp))
  ctl <- list(diagnose_k = FALSE)
  on  <- tulpa(y ~ x + (1 + x | g), data = d, family = "poisson",
               mode = "laplace", control = ctl)
  off <- tulpa(y ~ x + (1 + x | g), data = d, family = "poisson",
               mode = "laplace", control = c(ctl, list(skew_correct = FALSE)))
  expect_true(all(on$mean_shift == "applied"))
  expect_null(off$mean_shift)
  # Exact Metropolis-within-Gibbs on the same data, 12000 kept sweeps: the
  # intercept's posterior mean is -0.019 (SD 0.134). The mode-centred mixture
  # reads 0.045, half a posterior SD off; the centred one -0.020.
  expect_gt(coef(off)[["(Intercept)"]], 0.02)
  expect_lt(abs(coef(on)[["(Intercept)"]] - (-0.019)), 0.02)
})
