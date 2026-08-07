# Statistical gate for the subspace debias (gcol33/tulpa#304).
#
# The correction is APPROXIMATE and is not self-evidently correct: it samples
# x_S exactly while holding x_{-S} at its GAUSSIAN conditional, and that
# conditional is not the true one. The error it leaves lives in exactly the
# couplings being corrected, so whether it helps -- and whether S has to be
# closed under strong posterior coupling before it does -- is a measured
# question. This file measures it against ground truth rather than asserting it.
#
# GROUND TRUTH. For a Bernoulli random-intercept model at a FIXED random-effect
# SD the exact posterior of beta is available without MCMC: the groups are
# conditionally independent given beta, so each group's random intercept
# integrates out in one dimension by Gauss-Hermite quadrature, and the resulting
# p(beta | y) is evaluated on a fine two-dimensional grid and marginalized. That
# is the same target tulpa_laplace() approximates by a Gaussian, so the three
# marginals -- exact, Laplace, subspace-corrected -- are directly comparable.

sdr_fixture <- function(seed = 11L, G = 12L, per = 4L, b = c(-2.5, 0.8),
                        su = 0.7) {
  set.seed(seed)
  n <- G * per
  grp <- rep(seq_len(G), each = per)
  x <- rnorm(n)
  y <- rbinom(n, 1L, plogis(b[1] + b[2] * x + rnorm(G, 0, su)[grp]))
  list(y = y, X = cbind(1, x), n_trials = rep(1L, n), grp = grp, G = G, su = su,
       re = list(list(idx = grp, n_groups = G, n_coefs = 1L, sigma = su)),
       rt = list(idx = grp, n_groups = G, n_coefs = 1L))
}

# Gauss-Hermite nodes and weights for the physicists' weight exp(-t^2), by
# Golub-Welsch: the eigenvalues of the Jacobi matrix are the nodes and the
# squared first components of its eigenvectors give the weights. Computed here
# rather than taken from a package -- it is ten lines and this is the only
# quadrature the test needs.
sdr_gauss_hermite <- function(nq) {
  k <- seq_len(nq - 1L)
  J <- matrix(0, nq, nq)
  J[cbind(k, k + 1L)] <- sqrt(k / 2)
  J[cbind(k + 1L, k)] <- sqrt(k / 2)
  e <- eigen(J, symmetric = TRUE)
  o <- order(e$values)
  list(nodes = e$values[o], weights = sqrt(pi) * e$vectors[1L, o]^2)
}

# log p(y | beta) with the group intercepts integrated out, by Gauss-Hermite.
sdr_loglik <- function(beta, d, nq = 60L) {
  gh <- sdr_gauss_hermite(nq)
  nodes <- sqrt(2) * d$su * gh$nodes
  lw <- log(gh$weights) - 0.5 * log(pi)
  eta0 <- as.numeric(d$X %*% beta)
  tot <- 0
  for (g in seq_len(d$G)) {
    i <- which(d$grp == g)
    lp <- vapply(seq_len(nq), function(q) {
      e <- eta0[i] + nodes[q]
      sum(d$y[i] * e - log1p(exp(e)))
    }, numeric(1))
    m <- max(lp + lw)
    tot <- tot + m + log(sum(exp(lp + lw - m)))
  }
  tot
}

# Exact marginal of beta[j] on a grid, with the same N(0, prior_sd^2) prior the
# fitter uses. Returns the grid, the normalized marginal density, and its
# mean / SD / quantiles.
sdr_exact_marginal <- function(d, j, prior_sd, ng = 121L, span = 6) {
  f0 <- tulpa_laplace(d$y, d$n_trials, d$X, re_list = d$re, family = "binomial",
                      beta_prior = list(mean = 0, sd = prior_sd),
                      return_hessian = TRUE)
  se <- sqrt(diag(solve(f0$H_beta)))
  g1 <- seq(f0$mode[1] - span * se[1], f0$mode[1] + span * se[1], length.out = ng)
  g2 <- seq(f0$mode[2] - span * se[2], f0$mode[2] + span * se[2], length.out = ng)
  lp <- outer(seq_along(g1), seq_along(g2), Vectorize(function(a, b) {
    be <- c(g1[a], g2[b])
    sdr_loglik(be, d) + sum(dnorm(be, 0, prior_sd, log = TRUE))
  }))
  p <- exp(lp - max(lp))
  dens <- if (j == 1L) rowSums(p) else colSums(p)
  gr <- if (j == 1L) g1 else g2
  dens <- dens / sum(dens)
  m  <- sum(gr * dens)
  sd <- sqrt(sum((gr - m)^2 * dens))
  # Invert the CDF, dropping the flat tails where consecutive values are equal
  # to machine precision (approx() cannot order ties).
  cdf <- cumsum(dens)
  keep <- !duplicated(cdf)
  q <- vapply(c(0.025, 0.5, 0.975),
              function(pp) stats::approx(cdf[keep], gr[keep], xout = pp,
                                         rule = 2)$y,
              numeric(1))
  list(grid = gr, dens = dens, mean = m, sd = sd, q = q, fit = f0)
}

# Summary of a corrected marginal for latent index `j` from a debias solve.
sdr_corrected <- function(f, j, col) {
  x <- f$mode[j] + f$debias_draws[, col]
  list(mean = mean(x), sd = stats::sd(x),
       q = unname(stats::quantile(x, c(0.025, 0.5, 0.975))))
}


test_that("correcting the flagged direction recovers the exact marginal the Gaussian misses", {
  skip_on_cran()
  PSD <- 5
  d <- sdr_fixture()
  ex <- sdr_exact_marginal(d, 1L, PSD)

  # The Laplace Gaussian for the same coordinate.
  V <- solve(ex$fit$H_beta)
  gauss <- list(mean = ex$fit$mode[1], sd = sqrt(V[1, 1]))
  gauss$q <- qnorm(c(0.025, 0.5, 0.975), gauss$mean, gauss$sd)

  # The correction on S = {intercept}, the coordinate the selector flags.
  set.seed(5)
  fc <- tulpa_laplace(d$y, d$n_trials, d$X, re_list = d$re, family = "binomial",
                      beta_prior = list(mean = 0, sd = PSD),
                      return_hessian = TRUE,
                      debias = list(idx = 1L, n_iter = 30000L, warmup = 2000L))
  cor1 <- sdr_corrected(fc, 1L, 1L)

  # The exact marginal is skewed, so the Gaussian's symmetric interval is wrong
  # on at least one side; the correction must reduce the total endpoint error.
  err_g <- sum(abs(gauss$q[c(1, 3)] - ex$q[c(1, 3)]))
  err_c <- sum(abs(cor1$q[c(1, 3)] - ex$q[c(1, 3)]))
  expect_lt(err_c, err_g)
  # ... and it must not make the location worse.
  expect_lte(abs(cor1$mean - ex$mean), abs(gauss$mean - ex$mean) + 1e-8)
})


test_that("the closure question is answered by measurement, not assumption", {
  skip_if_not_slow()
  # Is conditioning x_{-S} on the Gaussian enough, or does S have to be closed
  # under strong posterior coupling? Both are run against the same exact
  # marginal and the errors compared. The random intercepts are the coordinates
  # the intercept is coupled to, so this fixture is where a closure would pay if
  # it ever does.
  PSD <- 5
  d <- sdr_fixture()
  ex <- sdr_exact_marginal(d, 1L, PSD)

  fj <- tulpa_laplace(d$y, d$n_trials, d$X, re_list = d$re, family = "binomial",
                      beta_prior = list(mean = 0, sd = PSD),
                      return_hessian = TRUE, return_joint_hessian = TRUE)
  H <- as.matrix(fj$H_joint)
  closed <- tulpa:::.subspace_closure(H, 1L, threshold = 0.3, max_add = 50L)
  expect_gt(length(closed$added), 0L)   # the coupling is real and detected

  err <- function(idx, seed) {
    set.seed(seed)
    f <- tulpa_laplace(d$y, d$n_trials, d$X, re_list = d$re,
                       family = "binomial",
                       beta_prior = list(mean = 0, sd = PSD),
                       return_hessian = TRUE,
                       debias = list(idx = idx, n_iter = 30000L,
                                     warmup = 3000L))
    cc <- sdr_corrected(f, 1L, which(idx == 1L))
    sum(abs(cc$q[c(1, 3)] - ex$q[c(1, 3)]))
  }
  e_open   <- mean(vapply(1:3, function(s) err(1L, s), numeric(1)))
  e_closed <- mean(vapply(1:3, function(s) err(closed$idx, s), numeric(1)))

  # Whatever the verdict, it is recorded as a number. The gate is only that the
  # closure does not make the corrected interval materially WORSE -- if it were
  # required, the open run would be the one failing here.
  expect_lt(e_closed, e_open + 0.15)
  info <- sprintf("closure endpoint error: open %.4f, closed %.4f (|S| 1 -> %d)",
                  e_open, e_closed, length(closed$idx))
  expect_true(is.finite(e_open) && is.finite(e_closed), info = info)
})


test_that("subspace debias tracks the full Gibbs debias at a fraction of its cost", {
  skip_if_not_slow()
  # The arbiter named in the issue: CI coverage against the FULL Gibbs debias.
  # Aggregated over seeds and both coefficients; the per-arm cost is measured in
  # the same loop so the saving is a number rather than a claim.
  n_seed <- 40L
  cov <- c(plain = 0L, sub = 0L, gibbs = 0L)
  secs <- c(plain = 0, sub = 0, gibbs = 0)
  n_hit <- 0L
  for (s in seq_len(n_seed)) {
    d <- sdr_fixture(seed = 400L + s, G = 20L, per = 3L, b = c(-2.5, 0.7))
    nt <- d$n_trials
    tp <- system.time(fp <- tulpa_re_cov_nested(d$y, nt, d$X, d$rt,
            family = "binomial", control = list(seed = 7L)))[["elapsed"]]
    ts <- system.time(fs <- tulpa_re_cov_nested(d$y, nt, d$X, d$rt,
            family = "binomial",
            control = list(seed = 7L, subspace_debias = TRUE)))[["elapsed"]]
    tg <- system.time(fg <- tulpa_re_cov_gibbs(d$y, nt, d$X, d$rt,
            family = "binomial",
            control = list(n_iter = 2000L, warmup = 1000L,
                           seed = 7L)))[["elapsed"]]
    secs <- secs + c(tp, ts, tg)
    n_hit <- n_hit + length(fs$subspace_debias$idx)
    dr <- list(plain = fp$draws, sub = fs$draws, gibbs = fg$beta_draws)
    for (arm in names(dr)) {
      ci <- apply(dr[[arm]], 2L, stats::quantile, probs = c(0.025, 0.975),
                  names = FALSE)
      cov[[arm]] <- cov[[arm]] +
        sum(c(-2.5, 0.7) >= ci[1L, ] & c(-2.5, 0.7) <= ci[2L, ])
    }
  }
  n_tot <- 2L * n_seed
  rate <- cov / n_tot
  se <- sqrt(rate * (1 - rate) / n_tot)

  # The selector fired: this is not a vacuous pass on an empty S.
  expect_gt(n_hit, 0L)
  # Coverage matches the full debias within two standard errors of the
  # difference (the arms are paired on the same data, so this is conservative).
  expect_lt(abs(rate[["sub"]] - rate[["gibbs"]]),
            2 * sqrt(se[["sub"]]^2 + se[["gibbs"]]^2) + 0.02)
  # ... at a measurably lower cost.
  expect_lt(secs[["sub"]], secs[["gibbs"]])
})
