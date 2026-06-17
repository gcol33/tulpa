# Random-slope recovery on the joint nested-Laplace multi-block driver
# (gcol33/tulpa#114): the two RE capabilities added for random *slopes* on an
# observation arm --
#   (1) a per-row design weight (svc_weight) on the `iid` block, so an
#       uncorrelated slope (0 + x | g) is one weighted iid block per coefficient;
#   (2) the multivariate `miid` block (mcar with Q = I), so a correlated slope
#       (1 + x | g) is one block over a free cross-coefficient Sigma.
# Statistical validation (parameter recovery vs simulated truth) plus the two
# byte-identical / equivalence reductions in the acceptance criteria.

# Build the column-major lower-tri log-Cholesky grid for a p = 2 Sigma from
# interpretable (sigma_1, sigma_2, rho) nodes (same mapping as the engine's
# .mcar_default_logchol_grid, just a coarser, test-controlled bracket).
.miid_logchol_p2 <- function(sig1, sig2, rho) {
  g <- expand.grid(s1 = sig1, s2 = sig2, rho = rho, KEEP.OUT.ATTRS = FALSE)
  out <- cbind(L11 = log(g$s1),
               L21 = g$rho * g$s2,
               L22 = log(g$s2 * sqrt(1 - g$rho^2)))
  colnames(out) <- c("L11", "L21", "L22")
  out
}

# Weighted empirical quantile (marginalize a derived quantity over the grid).
.wquantile <- function(v, w, probs) {
  o <- order(v); v <- v[o]; w <- w[o]
  cw <- cumsum(w) / sum(w)
  stats::approx(cw, v, xout = probs, ties = "ordered", rule = 2)$y
}

# One gaussian arm carrying a correlated per-group coefficient vector
# (b0_g, b1_g) ~ N(0, Sigma). eta_i = beta0 + beta1 x_i + b0_{g_i} + x_i b1_{g_i}.
.sim_miid <- function(seed, G = 40L, n_per = 20L,
                      beta = c(0.3, -0.4),
                      s0 = 0.8, s1 = 0.6, rho = 0.5, sd_noise = 0.3) {
  set.seed(seed)
  Sigma <- matrix(c(s0^2, rho * s0 * s1, rho * s0 * s1, s1^2), 2, 2)
  L <- t(chol(Sigma))
  Z <- matrix(rnorm(2 * G), G, 2)
  B <- Z %*% t(L)                         # G x 2: (b0_g, b1_g)
  grp <- rep(seq_len(G), each = n_per)
  N <- length(grp)
  x <- rnorm(N)
  eta <- beta[1] + beta[2] * x + B[grp, 1] + x * B[grp, 2]
  y <- rnorm(N, eta, sd_noise)
  list(y = y, x = x, grp = as.integer(grp), G = G, N = N,
       X = cbind(1, x), sd_noise = sd_noise,
       truth = list(s0 = s0, s1 = s1, rho = rho, beta = beta), B = B)
}

.gauss_arm <- function(sim) {
  list(y = sim$y, n_trials = rep(1L, sim$N), X = sim$X,
       re_idx = rep(0, sim$N), n_re_groups = 0L, sigma_re = 1.0,
       family = "gaussian", phi = sim$sd_noise)
}

# --------------------------------------------------------------------------- #
# 1. svc_weight on the iid block: an uncorrelated random slope (0 + x | g).    #
# --------------------------------------------------------------------------- #

test_that("weighted iid block recovers a random-slope variance (svc_weight)", {
  skip_on_cran()
  set.seed(101)
  G <- 50L; n_per <- 20L; sigma_s <- 0.7; sd_noise <- 0.3
  grp <- rep(seq_len(G), each = n_per); N <- length(grp)
  x <- rnorm(N)
  b <- rnorm(G, 0, sigma_s)               # slope-only RE
  eta <- 0.2 - 0.3 * x + x * b[grp]
  y <- rnorm(N, eta, sd_noise)

  arm <- list(y = y, n_trials = rep(1L, N), X = cbind(1, x),
              re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
              family = "gaussian", phi = sd_noise)
  prior <- list(list(
    type = "iid", n_units = G, obs_idx = list(grp),
    svc_weight = list(x),                 # the slope's covariate column
    sigma_grid = c(0.3, 0.5, 0.7, 0.9, 1.2)
  ))
  fit <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(arm = arm), prior = prior,
    control = list(max_iter = 60L, tol = 1e-6, diagnose_k = FALSE)))

  sigma_hat <- fit$block_moments[[1L]]$mean[["sigma"]]
  expect_lt(abs(sigma_hat - sigma_s) / sigma_s, 0.30)
})

# --------------------------------------------------------------------------- #
# 2. svc_weight unset == all-ones: the new row_weight path reduces exactly.    #
# --------------------------------------------------------------------------- #

test_that("iid svc_weight all-ones is identical to the unweighted iid block", {
  skip_on_cran()
  set.seed(202)
  G <- 30L; n_per <- 15L
  grp <- rep(seq_len(G), each = n_per); N <- length(grp)
  x <- rnorm(N)
  iota <- rnorm(G, 0, 0.5)
  y <- rnorm(N, 0.1 + 0.4 * x + iota[grp], 0.3)
  arm <- list(y = y, n_trials = rep(1L, N), X = cbind(1, x),
              re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
              family = "gaussian", phi = 0.3)

  base <- list(type = "iid", n_units = G, obs_idx = list(grp),
               sigma_grid = c(0.3, 0.5, 0.8))
  fit_unset <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(arm = arm), prior = list(base),
    control = list(diagnose_k = FALSE)))
  base$svc_weight <- list(rep(1, N))      # unit weight => weight 1 in the scatter
  fit_ones <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(arm = arm), prior = list(base),
    control = list(diagnose_k = FALSE)))

  # x * 1.0 == x bit-for-bit, so the weighted path with unit weights is the
  # unweighted block: the inner log-marginal is unchanged at every grid cell.
  expect_equal(fit_ones$log_marginal, fit_unset$log_marginal, tolerance = 1e-12)
})

# --------------------------------------------------------------------------- #
# 3. miid free-Sigma block: a correlated random slope (1 + x | g).            #
# --------------------------------------------------------------------------- #

test_that("miid block recovers a correlated random-slope Sigma", {
  skip_on_cran()
  sim <- .sim_miid(seed = 11L)
  logchol <- .miid_logchol_p2(sig1 = c(0.5, 0.8, 1.1),
                              sig2 = c(0.4, 0.6, 0.9),
                              rho  = c(0, 0.4, 0.7))
  prior <- list(list(
    type = "miid", n_groups = sim$G, n_fields = 2L,
    obs_idx = list(sim$grp),
    field_weight = list(list(rep(1, sim$N)), list(sim$x)),
    logchol_grid = logchol
  ))
  fit <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(arm = .gauss_arm(sim)), prior = prior,
    control = list(max_iter = 80L, tol = 1e-6, diagnose_k = FALSE)))

  # Marginalize the derived (sigma_1, sigma_2, rho) over the grid weights:
  # L = [[L11,0],[L21,L22]], Sigma = L L'.
  tg <- fit$theta_grid
  L11 <- exp(tg[, "b1.L11"]); L21 <- tg[, "b1.L21"]; L22 <- exp(tg[, "b1.L22"])
  s0_cell <- L11
  s1_cell <- sqrt(L21^2 + L22^2)
  rho_cell <- L21 / s1_cell
  w <- fit$weights

  s0_med  <- .wquantile(s0_cell, w, 0.5)
  s1_med  <- .wquantile(s1_cell, w, 0.5)
  rho_med <- .wquantile(rho_cell, w, 0.5)

  expect_lt(abs(s0_med  - sim$truth$s0) / sim$truth$s0, 0.40)
  expect_lt(abs(s1_med  - sim$truth$s1) / sim$truth$s1, 0.45)
  # The cross-correlation recovers as clearly positive near the truth (0.5).
  expect_gt(rho_med, 0.2)
  expect_lt(rho_med, 0.9)
})

# --------------------------------------------------------------------------- #
# 4. miid p = 1 == scalar iid (centered vs non-centered, Laplace-invariant).  #
# --------------------------------------------------------------------------- #

test_that("miid p=1 matches the scalar iid block on the inner log-marginal", {
  skip_on_cran()
  set.seed(303)
  G <- 35L; n_per <- 15L
  grp <- rep(seq_len(G), each = n_per); N <- length(grp)
  x <- rnorm(N)
  iota <- rnorm(G, 0, 0.6)
  y <- rnorm(N, -0.2 + 0.5 * x + iota[grp], 0.3)
  arm <- list(y = y, n_trials = rep(1L, N), X = cbind(1, x),
              re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
              family = "gaussian", phi = 0.3)

  s_vals <- c(0.3, 0.45, 0.6, 0.8, 1.1)

  fit_iid <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(arm = arm),
    prior = list(list(type = "iid", n_units = G, obs_idx = list(grp),
                      sigma_grid = s_vals)),
    control = list(max_iter = 80L, tol = 1e-7, diagnose_k = FALSE)))

  # miid with p = 1 on the matched log-Cholesky grid logL11 = log(sigma): the
  # centered N(0, sigma^2) reparameterization of the non-centered iid field.
  logchol <- matrix(log(s_vals), ncol = 1L, dimnames = list(NULL, "L11"))
  fit_miid <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(arm = arm),
    prior = list(list(type = "miid", n_groups = G, n_fields = 1L,
                      obs_idx = list(grp),
                      field_weight = list(list(rep(1, N))),
                      logchol_grid = logchol)),
    control = list(max_iter = 80L, tol = 1e-7, diagnose_k = FALSE)))

  # Same model, affine reparameterization of the latent => the Laplace
  # marginal is invariant: per-cell log-marginal (sigma = s_vals) matches.
  expect_equal(fit_miid$log_marginal, fit_iid$log_marginal, tolerance = 1e-5)
})
