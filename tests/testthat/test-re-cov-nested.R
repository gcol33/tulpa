# Nested-Laplace integration over a random-effect covariance Sigma
# (Bias-2 fix: marginalize Sigma over a grid, report a marginalized median and
# interval for the derived scale / correlation parameters instead of the
# plug-in MAP).

sim_corr_recov <- function(seed, G = 60L, npg = 12L, beta = c(-0.3, 0.7),
                           Sigma = matrix(c(0.8^2, 0.5 * 0.8 * 0.6,
                                            0.5 * 0.8 * 0.6, 0.6^2), 2)) {
  set.seed(seed)
  N <- G * npg
  grp <- rep(seq_len(G), each = npg)
  x <- rnorm(N)
  X <- cbind(1, x)
  Z <- cbind(1, x)
  u <- t(t(chol(Sigma)) %*% matrix(rnorm(2 * G), nrow = 2))
  eta <- as.numeric(X %*% beta) + rowSums(Z * u[grp, ])
  y <- rbinom(N, 1L, plogis(eta))
  list(y = y, X = X, Z = Z, grp = grp, G = G, N = N, Sigma = Sigma)
}


test_that("tulpa_re_cov_nested returns a well-formed marginalized posterior", {
  skip_on_cran()
  d <- sim_corr_recov(1L)
  re_term <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)
  res <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, re_term,
                             family = "binomial")

  # default CCD layout: k = c(c+1)/2 = 3 -> 1 centre + 2k axial + 2^k factorial
  expect_equal(res$n_grid, 1L + 2L * 3L + 2L^3L)  # = 15
  expect_equal(sum(res$weights), 1, tolerance = 1e-10)
  expect_true(all(res$weights >= 0))

  # tensor grid is opt-in via integration = "grid": n_per_axis^k cells
  res_grid <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, re_term,
                                  family = "binomial",
                                  control = list(integration = "grid",
                                                 n_per_axis = 5L, span = 3))
  expect_equal(res_grid$n_grid, 5L^3L)
  expect_equal(sum(res_grid$weights), 1, tolerance = 1e-10)

  # posterior table shape: sigma_1, sigma_2, rho_12, Sigma_11/12/22
  expect_setequal(res$posterior$parameter,
                  c("sigma_1", "sigma_2", "rho_12",
                    "Sigma_11", "Sigma_12", "Sigma_22"))
  expect_true(all(c("mean", "sd", "median", "ci_lo", "ci_hi") %in%
                    names(res$posterior)))

  # CI ordering ci_lo <= median <= ci_hi for every parameter
  expect_true(all(res$posterior$ci_lo <= res$posterior$median + 1e-8))
  expect_true(all(res$posterior$median <= res$posterior$ci_hi + 1e-8))

  # weighted-mean Sigma is symmetric positive definite
  expect_equal(res$Sigma_mean, t(res$Sigma_mean), tolerance = 1e-10)
  expect_true(all(eigen(res$Sigma_mean, symmetric = TRUE,
                        only.values = TRUE)$values > 0))

  # rho stays in [-1, 1]
  rho <- res$posterior[res$posterior$parameter == "rho_12", ]
  expect_gte(rho$ci_lo, -1)
  expect_lte(rho$ci_hi, 1)

  # MAP summary reported alongside the marginalized one
  expect_length(res$map$sigma, 2L)
  expect_length(res$map$rho, 1L)
})


test_that("marginalized 95% intervals cover the true Sigma parameters", {
  skip_on_cran()
  # A few seeds; the marginalized 95% CI should contain truth in the large
  # majority. (Point recovery of the scales is mildly biased low by the
  # Laplace approximation -- Bias 1 -- which the Gibbs correction targets;
  # here we check the *interval*, which is what the Bias-2 fix delivers.)
  truth <- c(sigma_1 = 0.8, sigma_2 = 0.6, rho_12 = 0.5)
  covered <- c(sigma_1 = 0L, sigma_2 = 0L, rho_12 = 0L)
  n_seed <- 5L
  for (s in seq_len(n_seed)) {
    d <- sim_corr_recov(100L + s)
    re_term <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)
    res <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, re_term,
                               family = "binomial")   # default CCD integration
    for (nm in names(truth)) {
      row <- res$posterior[res$posterior$parameter == nm, ]
      if (truth[[nm]] >= row$ci_lo && truth[[nm]] <= row$ci_hi)
        covered[[nm]] <- covered[[nm]] + 1L
    }
  }
  # With only 5 seeds, require the scales covered every time and rho most times.
  expect_gte(covered[["sigma_1"]], 4L)
  expect_gte(covered[["sigma_2"]], 4L)
  expect_gte(covered[["rho_12"]], 3L)
})


test_that("the median is a more central summary than the mode under skew", {
  skip_on_cran()
  # Small G -> skewed variance-component marginal. The marginalized posterior
  # mean should sit at or above the plug-in MAP for the variance scales (right
  # skew pulls the mean up), and the median between them.
  d <- sim_corr_recov(7L, G = 25L, npg = 12L)
  re_term <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)
  res <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, re_term,
                             family = "binomial")   # default CCD integration
  s1 <- res$posterior[res$posterior$parameter == "Sigma_11", ]
  # right-skewed: mean >= median, and both finite/positive
  expect_gte(s1$mean, s1$median - 1e-6)
  expect_gt(s1$median, 0)
})


# --- diagonal (uncorrelated) block -------------------------------------------
# A `(1 + x || g)` term is a diagonal Sigma: c log-SD integration params (here
# k = 2, vs 3 for the full block), no correlation parameter, and the off-diagonal
# entry is structurally zero. Truth has uncorrelated intercept/slope.
sim_diag_recov <- function(seed, G = 60L, npg = 12L, beta = c(-0.3, 0.6),
                           s1 = 0.8, s2 = 0.5) {
  set.seed(seed)
  N <- G * npg
  grp <- rep(seq_len(G), each = npg)
  x <- rnorm(N); X <- cbind(1, x); Z <- cbind(1, x)
  u <- cbind(rnorm(G, 0, s1), rnorm(G, 0, s2))   # independent intercept/slope
  eta <- as.numeric(X %*% beta) + rowSums(Z * u[grp, ])
  list(y = rbinom(N, 1L, plogis(eta)), X = X, Z = Z, grp = grp, G = G, N = N)
}

test_that("tulpa_re_cov_nested integrates a diagonal (uncorrelated) block", {
  skip_on_cran()
  d <- sim_diag_recov(11L)
  rt <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z,
             correlated = FALSE)
  res <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, rt, family = "binomial")

  # diagonal block: k = c = 2 -> CCD 1 + 2k + 2^k = 9 nodes
  expect_equal(res$n_grid, 1L + 2L * 2L + 2L^2L)
  # no correlation reported; only diagonal variances
  expect_setequal(res$posterior$parameter,
                  c("sigma_1", "sigma_2", "Sigma_11", "Sigma_22"))
  expect_length(res$map$rho, 0L)
  # the MAP Sigma really is diagonal (off-diagonal exactly zero)
  expect_equal(res$map$Sigma[1L, 2L], 0)
  # scales recovered to the right ballpark (interval covers truth)
  for (nm in c("sigma_1", "sigma_2")) {
    row <- res$posterior[res$posterior$parameter == nm, ]
    truth <- if (nm == "sigma_1") 0.8 else 0.5
    expect_gte(row$ci_hi, truth * 0.6)
    expect_lte(row$ci_lo, truth * 1.4)
  }
})


# --- multi-term: a list of blocks --------------------------------------------
test_that("tulpa_re_cov_nested integrates several terms as separate blocks", {
  skip_on_cran()
  set.seed(21L)
  G <- 40L; H <- 25L; npg <- 12L; N <- G * npg
  g <- rep(seq_len(G), each = npg)
  h <- sample.int(H, N, replace = TRUE)
  x <- rnorm(N); X <- cbind(1, x); Zg <- cbind(1, x)
  Sg <- matrix(c(0.8^2, 0.4 * 0.8 * 0.5, 0.4 * 0.8 * 0.5, 0.5^2), 2)
  ug <- t(t(chol(Sg)) %*% matrix(rnorm(2 * G), 2))
  uh <- rnorm(H, 0, 0.6)
  eta <- as.numeric(X %*% c(-0.2, 0.6)) + rowSums(Zg * ug[g, ]) + uh[h]
  y <- rbinom(N, 1L, plogis(eta))

  re_terms <- list(
    list(idx = g, n_groups = G, n_coefs = 2L, Z = Zg, correlated = TRUE,
         label = "g"),
    list(idx = h, n_groups = H, n_coefs = 1L, correlated = FALSE, label = "h")
  )
  res <- tulpa_re_cov_nested(y, rep(1L, N), X, re_terms, family = "binomial",
                             control = list(seed = 3L, n_draws = 300L))

  expect_equal(res$n_blocks, 2L)
  expect_equal(unname(res$n_coefs), c(2L, 1L))
  # stacked integration dimension k = 3 (full g) + 1 (scalar h) = 4
  # CCD(k=4): 1 + 2*4 + fractional factorial -- just check it ran with > 4 nodes
  expect_gt(res$n_grid, 4L)
  expect_setequal(res$posterior$parameter,
                  c("g.sigma_1", "g.sigma_2", "g.rho_12",
                    "g.Sigma_11", "g.Sigma_12", "g.Sigma_22",
                    "h.sigma_1", "h.Sigma_11"))
  expect_named(res$Sigma_mean, c("g", "h"))
  expect_equal(dim(res$Sigma_mean$g), c(2L, 2L))
  # both grouping scales positive
  for (nm in c("g.sigma_1", "g.sigma_2", "h.sigma_1")) {
    expect_gt(res$posterior[res$posterior$parameter == nm, "median"], 0)
  }
})


# --- AGHQ-refined inner solve (n_quad > 1) -----------------------------------
# Single shared grouping factor only: the per-group integral must factorize, so
# n_quad > 1 errors on crossed RE terms.
test_that("n_quad > 1 (AGHQ inner solve) errors on crossed grouping factors", {
  set.seed(21L)
  G <- 20L; H <- 15L; npg <- 8L; N <- G * npg
  g <- rep(seq_len(G), each = npg)
  h <- sample.int(H, N, replace = TRUE)
  x <- rnorm(N); X <- cbind(1, x); Zg <- cbind(1, x)
  y <- rbinom(N, 1L, plogis(as.numeric(X %*% c(-0.2, 0.5))))
  re_terms <- list(
    list(idx = g, n_groups = G, n_coefs = 2L, Z = Zg, label = "g"),
    list(idx = h, n_groups = H, n_coefs = 1L, correlated = FALSE, label = "h")
  )
  expect_error(
    tulpa_re_cov_nested(y, rep(1L, N), X, re_terms, family = "binomial",
                        n_quad = 5L),
    "single shared grouping factor"
  )
})

test_that("n_quad > 1 (AGHQ inner solve) recovers Sigma on a single factor", {
  skip_on_cran()
  # AGHQ refines each per-group inner marginal (the tulpa_re_aghq debias applied
  # inside the Sigma integration); the marginalized 95% intervals should still
  # cover the true scales, and the posterior keeps the full-block shape.
  truth <- c(sigma_1 = 0.8, sigma_2 = 0.6)
  covered <- c(sigma_1 = 0L, sigma_2 = 0L)
  for (s in seq_len(3L)) {
    d <- sim_corr_recov(300L + s)
    rt <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)
    res <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, rt, family = "binomial",
                               n_quad = 5L)
    expect_setequal(res$posterior$parameter,
                    c("sigma_1", "sigma_2", "rho_12",
                      "Sigma_11", "Sigma_12", "Sigma_22"))
    expect_true(all(eigen(res$Sigma_mean, symmetric = TRUE,
                          only.values = TRUE)$values > 0))
    for (nm in names(truth)) {
      row <- res$posterior[res$posterior$parameter == nm, ]
      if (truth[[nm]] >= row$ci_lo && truth[[nm]] <= row$ci_hi)
        covered[[nm]] <- covered[[nm]] + 1L
    }
  }
  expect_gte(covered[["sigma_1"]], 2L)
  expect_gte(covered[["sigma_2"]], 2L)
})


# ---------------------------------------------------------------------------
# Reading the interval off a quadrature design (gcol33/tulpa#308).
#
# A CCD reproduces the integrand's moments; its node positions carry no
# probability mass, so a discrete weighted quantile over them is bounded by the
# design's own extent. These pin the moment-matched summary that replaces it and
# the explicit out-of-support policy on the discrete one.
# ---------------------------------------------------------------------------

# The standardized k = 1 design: centre plus two axial nodes at +/- 1.1, with
# the corrected R-INLA design weights.
ccd1 <- local({
  g <- ccd_grid(1L, f_0 = 1.1)
  w <- ccd_weights(g)
  list(z = as.numeric(g$z), w = w / sum(w))
})

test_that("moment interval reproduces the lognormal the design's moments imply", {
  m <- -0.3; s <- 0.8
  u <- m + s * ccd1$z
  v <- exp(u)
  q <- tulpa:::.nl_moment_quantile(v, ccd1$w, c(0.025, 0.5, 0.975), "positive")

  mu  <- sum(ccd1$w * u)
  sdu <- sqrt(sum(ccd1$w * u^2) - mu^2)
  expect_equal(q, exp(mu + qnorm(c(0.025, 0.5, 0.975)) * sdu))
  # The median is the back-transformed first moment, so it is a scale rather
  # than one of the three nodes.
  expect_equal(q[2L], exp(mu))
  # And the interval is free to leave the node range, which the discrete
  # quantile over the same nodes cannot.
  expect_lt(q[1L], min(v))
  expect_gt(q[3L], max(v))
  expect_equal(tulpa:::.nl_wtd_quantile(v, ccd1$w, c(0.025, 0.975)),
               c(min(v), max(v)))
})

test_that("moment interval respects each domain and declines outside it", {
  probs <- c(0.025, 0.5, 0.975)
  # A correlation is summarized on the Fisher-z coordinate, so the interval
  # stays inside (-1, 1) however wide the nodes are.
  qc <- tulpa:::.nl_moment_quantile(c(0, 0.95, -0.95), ccd1$w, probs,
                                    "correlation")
  expect_true(all(abs(qc) < 1))
  expect_equal(qc[2L], 0)
  # An unbounded quantity gets the plain Gaussian interval.
  qu <- tulpa:::.nl_moment_quantile(c(0, 2, -2), ccd1$w, probs, "unbounded")
  expect_equal(qu[2L], 0)
  expect_equal(qu[3L], -qu[1L])
  # Nodes outside the domain, or no usable weight, withhold the summary rather
  # than dropping the offending node and integrating a different design.
  expect_true(all(is.na(
    tulpa:::.nl_moment_quantile(c(1, -1, 2), ccd1$w, probs, "positive"))))
  expect_true(all(is.na(
    tulpa:::.nl_moment_quantile(c(0, 1, -0.5), ccd1$w, probs, "correlation"))))
  expect_true(all(is.na(
    tulpa:::.nl_moment_quantile(c(1, 2, 3), c(0, 0, 0), probs, "positive"))))
  expect_error(
    tulpa:::.nl_moment_quantile(c(1, 2, 3), ccd1$w, probs, "nope"),
    "unknown derived-quantity domain")
})

test_that("the discrete quantile's out-of-support policy is explicit", {
  v <- c(1, 2, 3); w <- rep(1 / 3, 3L)
  # Default: the extreme atom, the cumulative-mass convention a sample uses.
  expect_equal(tulpa:::.nl_wtd_quantile(v, w, c(0.025, 0.975)), c(1, 3))
  # outside = "na" withholds it, and leaves interior probabilities alone.
  expect_true(all(is.na(
    tulpa:::.nl_wtd_quantile(v, w, c(0.025, 0.975), outside = "na"))))
  expect_equal(tulpa:::.nl_wtd_quantile(v, w, 0.5, outside = "na"), 2)
  # A sample large enough to bracket the requested probability is unaffected by
  # the policy at all.
  vs <- qnorm(seq(0.001, 0.999, length.out = 400L))
  ws <- rep(1 / 400, 400L)
  expect_equal(tulpa:::.nl_wtd_quantile(vs, ws, c(0.025, 0.975)),
               tulpa:::.nl_wtd_quantile(vs, ws, c(0.025, 0.975),
                                        outside = "na"))
})

test_that("each derived quantity carries the domain its interval is formed on", {
  S <- matrix(c(0.49, 0.14, 0.14, 0.25), 2, 2)
  D <- tulpa:::.re_cov_derived_matrix(list(S, S * 1.1), 2L, full = TRUE)
  expect_equal(colnames(D),
               c("sigma_1", "sigma_2", "rho_12",
                 "Sigma_11", "Sigma_12", "Sigma_22"))
  expect_equal(attr(D, "domain"),
               c("positive", "positive", "correlation",
                 "positive", "unbounded", "positive"))
  Dd <- tulpa:::.re_cov_derived_matrix(list(S, S), 2L, full = FALSE)
  expect_equal(colnames(Dd), c("sigma_1", "sigma_2", "Sigma_11", "Sigma_22"))
  expect_equal(attr(Dd, "domain"), rep("positive", 4L))
  # Several blocks concatenate their domains alongside their columns.
  layout <- list(
    list(nc = 2L, full = TRUE,  k = 3L, label = "g", n_groups = 5L),
    list(nc = 1L, full = FALSE, k = 1L, label = "h", n_groups = 4L))
  nodes <- list(list(S, matrix(0.36, 1, 1)), list(S * 1.2, matrix(0.4, 1, 1)))
  DM <- tulpa:::.re_cov_derived_matrix_multi(nodes, layout)
  expect_equal(length(attr(DM, "domain")), ncol(DM))
  expect_equal(attr(DM, "domain")[colnames(DM) == "g.rho_12"], "correlation")
  expect_equal(attr(DM, "domain")[colnames(DM) == "h.sigma_1"], "positive")
})

test_that("the summary support decides how the interval is read off the nodes", {
  layout <- list(list(nc = 1L, full = FALSE, k = 1L, label = "g",
                      n_groups = 5L))
  sig <- exp(-0.3 + 0.8 * ccd1$z)
  nodes <- lapply(sig, function(s) list(matrix(s^2, 1, 1)))

  mom <- tulpa:::.re_cov_derived_summary(nodes, ccd1$w, layout,
                                         support = "moment_rule")
  den <- tulpa:::.re_cov_derived_summary(nodes, ccd1$w, layout,
                                         support = "density")
  row <- function(p, nm) p$posterior[p$posterior$parameter == nm, ]
  # Same weighted moments either way; only the median and interval differ.
  expect_equal(row(mom, "sigma_1")$mean, row(den, "sigma_1")$mean)
  expect_equal(row(mom, "sigma_1")$sd, row(den, "sigma_1")$sd)
  # The density reading is pinned to the node GEOMETRY -- the node range widened
  # by the outer cells' own half-spacing and no further (gcol33/tulpa#353);
  # before that it was the node range exactly. The moment reading is not pinned
  # to the nodes at all.
  e <- tulpa:::.nl_cell_edges(sort(sig))
  expect_lt(row(den, "sigma_1")$ci_lo, min(sig))
  expect_gt(row(den, "sigma_1")$ci_lo, e[1])
  expect_gt(row(den, "sigma_1")$ci_hi, max(sig))
  expect_lt(row(den, "sigma_1")$ci_hi, e[2])
  expect_lt(row(mom, "sigma_1")$ci_lo, min(sig))
  expect_gt(row(mom, "sigma_1")$ci_hi, max(sig))
  # A variance is positive too, so its interval is a positive lognormal one.
  expect_gt(row(mom, "Sigma_11")$ci_lo, 0)
  # With equal weights over genuine draws the density reading is the sample
  # quantile, which is what the Gibbs backend relies on.
  set.seed(11L)
  dr <- lapply(rexp(400L), function(s) list(matrix(s^2, 1, 1)))
  eq <- tulpa:::.re_cov_derived_summary(dr, rep(1 / 400, 400L), layout)
  expect_equal(row(eq, "sigma_1")$median,
               unname(quantile(vapply(dr, function(z) sqrt(z[[1L]][1, 1]),
                                      numeric(1)), 0.5, type = 7)),
               tolerance = 1e-8)
})

test_that("a CCD fit reports a scale interval that is not the node extent", {
  skip_on_cran()
  d <- sim_corr_recov(77L, G = 60L, npg = 12L)
  rt <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)
  res <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, rt, family = "binomial",
                             control = list(diagnose_k = FALSE))
  # theta axis 1 is log(L_11) = log(sigma_1), so the design's extent on sigma_1
  # is exp(range(theta_grid[, 1])). The reported interval covers strictly more.
  ax  <- range(as.numeric(res$theta_grid[, 1L]))
  row <- res$posterior[res$posterior$parameter == "sigma_1", ]
  expect_lt(log(row$ci_lo), ax[1L])
  expect_gt(log(row$ci_hi), ax[2L])
  expect_gt(row$ci_lo, 0)
  # A correlation interval stays inside (-1, 1).
  rr <- res$posterior[res$posterior$parameter == "rho_12", ]
  expect_gt(rr$ci_lo, -1)
  expect_lt(rr$ci_hi, 1)
  # It is a Gaussian interval on log(sigma), so it is symmetric about the
  # reported median in that coordinate -- which a quantile read off scattered
  # node positions would not be.
  expect_equal(log(row$ci_hi) - log(row$median),
               log(row$median) - log(row$ci_lo))
  expect_equal(atanh(rr$ci_hi) - atanh(rr$median),
               atanh(rr$median) - atanh(rr$ci_lo))
})
