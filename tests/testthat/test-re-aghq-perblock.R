# Per-block quadrature order in tulpa_re_aghq(): `n_quad` may be a scalar
# (broadcast to every covariance block, the historical behaviour) or an integer
# vector of length length(re_terms) giving one node count per block. A scalar
# must reproduce the uniform tensor grid byte-for-byte; a per-block grid with
# fewer nodes on the cheap blocks must still recover a known truth.

l1pe <- function(x) ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))

make_binom_site <- function(X, y, nt) {
  function(theta) {
    eta_fixed <- as.numeric(X %*% theta)
    list(
      eta_re = eta_fixed,
      deriv = function(rows, eta) {
        p <- plogis(eta)
        list(logL = y[rows] * eta - nt[rows] * l1pe(eta),
             d1 = y[rows] - nt[rows] * p,
             d2 = -nt[rows] * p * (1 - p))
      },
      lmat = function(rows, ETA) y[rows] * ETA - nt[rows] * l1pe(ETA))
  }
}

# Two diagonal 1-D blocks sharing one grouping factor: an intercept block
# (1 | g) and a slope block (0 + x | g). This is the smallest heterogeneous
# stack that distinguishes a scalar n_quad from a per-block vector.
sim_two_block <- function(seed, ng, n_per, sd_int, sd_slp) {
  set.seed(seed)
  N <- ng * n_per
  group <- rep(seq_len(ng), each = n_per)
  x <- rnorm(N); X <- cbind(1, x); nt <- rep(3L, N)
  b_int <- rnorm(ng, 0, sd_int); b_slp <- rnorm(ng, 0, sd_slp)
  y <- rbinom(N, nt, plogis(0.3 + 0.7 * x + b_int[group] + b_slp[group] * x))
  list(group = group, x = x, X = X, nt = nt, y = y, ng = ng, N = N)
}

two_block_terms <- function(d) {
  list(list(idx = d$group, n_groups = d$ng, n_coefs = 1L, Z = matrix(1, d$N, 1)),
       list(idx = d$group, n_groups = d$ng, n_coefs = 1L, Z = matrix(d$x, d$N, 1)))
}

test_that("scalar n_quad equals a uniform per-block vector byte-for-byte", {
  skip_on_cran()
  d <- sim_two_block(101, ng = 30L, n_per = 8L, sd_int = 0.8, sd_slp = 0.5)
  re_terms <- two_block_terms(d)
  Sigma0 <- list(matrix(0.25, 1, 1), matrix(0.25, 1, 1))
  site <- make_binom_site(d$X, d$y, d$nt)

  fit_s <- tulpa_re_aghq(c(0, 0), re_terms, Sigma0, make_site = site,
                         n_obs = d$N, n_quad = 3L)
  fit_v <- tulpa_re_aghq(c(0, 0), re_terms, Sigma0, make_site = site,
                         n_obs = d$N, n_quad = c(3L, 3L))

  expect_false(is.null(fit_s)); expect_false(is.null(fit_v))
  expect_equal(fit_v$theta, fit_s$theta, tolerance = 1e-12)
  expect_equal(fit_v$Sigma_list[[1]], fit_s$Sigma_list[[1]], tolerance = 1e-12)
  expect_equal(fit_v$Sigma_list[[2]], fit_s$Sigma_list[[2]], tolerance = 1e-12)
  expect_equal(fit_v$log_marginal, fit_s$log_marginal, tolerance = 1e-12)
  expect_equal(fit_v$theta_se, fit_s$theta_se, tolerance = 1e-12)
  expect_equal(fit_v$blup[[1]], fit_s$blup[[1]], tolerance = 1e-12)
  expect_equal(fit_v$blup[[2]], fit_s$blup[[2]], tolerance = 1e-12)
})

test_that("a per-block grid (fewer nodes on the slope block) recovers truth", {
  skip_on_cran()
  sd_int <- 0.9; sd_slp <- 0.6
  d <- sim_two_block(7, ng = 60L, n_per = 8L, sd_int = sd_int, sd_slp = sd_slp)
  re_terms <- two_block_terms(d)
  Sigma0 <- list(matrix(0.25, 1, 1), matrix(0.25, 1, 1))
  site <- make_binom_site(d$X, d$y, d$nt)

  fit_pb <- tulpa_re_aghq(c(0, 0), re_terms, Sigma0, make_site = site,
                          n_obs = d$N, n_quad = c(7L, 3L))
  fit_u  <- tulpa_re_aghq(c(0, 0), re_terms, Sigma0, make_site = site,
                          n_obs = d$N, n_quad = 7L)

  expect_false(is.null(fit_pb))
  sd_int_hat <- sqrt(fit_pb$Sigma_list[[1]][1, 1])
  sd_slp_hat <- sqrt(fit_pb$Sigma_list[[2]][1, 1])
  # Per-block grid recovers both SDs near truth.
  expect_lt(abs(sd_int_hat - sd_int), 0.2)
  expect_lt(abs(sd_slp_hat - sd_slp), 0.2)
  # Reduced slope-block resolution barely moves the estimate vs the uniform grid.
  expect_lt(abs(sd_int_hat - sqrt(fit_u$Sigma_list[[1]][1, 1])), 0.03)
  expect_lt(abs(sd_slp_hat - sqrt(fit_u$Sigma_list[[2]][1, 1])), 0.03)
  expect_true(all(is.finite(fit_pb$theta_se)))
})

test_that("n_quad of a wrong length is rejected", {
  d <- sim_two_block(1, ng = 5L, n_per = 4L, sd_int = 0.5, sd_slp = 0.5)
  re_terms <- two_block_terms(d)
  Sigma0 <- list(matrix(0.25, 1, 1), matrix(0.25, 1, 1))
  site <- make_binom_site(d$X, d$y, d$nt)
  expect_error(
    tulpa_re_aghq(c(0, 0), re_terms, Sigma0, make_site = site,
                  n_obs = d$N, n_quad = c(3L, 3L, 3L)),
    "one per RE block")
  expect_error(
    tulpa_re_aghq(c(0, 0), re_terms, Sigma0, make_site = site,
                  n_obs = d$N, n_quad = c(3L, 0L)),
    "positive integers")
})
