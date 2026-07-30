# ranef() on the two RE-covariance backends (gcol33/tulpa#264). Both carry
# per-group information -- re_cov_gibbs samples b in its Metropolis-within-Gibbs
# sweep, re_cov_nested has a Gaussian per-group posterior at every Sigma node --
# so both report group effects. These are exactly the fits whose random-effect
# structure is the point, so an empty table (indistinguishable from a model with
# no random effects) is the one thing ranef() must not return here.
#
# Layout is what the ordering assertions guard: the RE block is stored
# term-major, then group, then coefficient within group, which is the order
# .re_names_from_layout labels and .tulpa_re_map indexes. A transposed layout
# would leave the labels intact and move the numbers to the wrong group.
#
# What is NOT asserted, and why: per-group coverage of the simulated b_g. The RE
# design columns here are a subset of the fixed design's, so the stationarity
# conditions of the penalized mode force sum_g b_g = 0 exactly (summing the b
# score equations leaves sum_g [Sigma^-1 b_g] = 0, the data term having been
# cancelled by the beta equations). One simulated draw of G groups has a non-zero
# group mean, and that direction is absorbed by the intercept -- it is not
# identified from the data. Correlation with the truth is invariant to that
# shift, so it is the recovery statistic used throughout; the group-mean
# direction itself is asserted only as the identity above.

# Independent intercept and slope truths (NOT one a transform of the other), so
# "the intercept rows track the intercept truth" cannot be satisfied by a layout
# that put the slopes there. Poisson counts: a random slope needs within-group
# information, and a binary response leaves the slope correlation at ~0.75 for
# any of these backends -- too coarse for the ordering assertions to bite.
make_ranef_re_data <- function(seed = 7L, G = 20L, npg = 40L) {
  set.seed(seed)
  n <- G * npg
  grp <- rep(seq_len(G), each = npg)
  x <- rnorm(n)
  b1 <- rnorm(G, 0, 0.8)
  b2 <- rnorm(G, 0, 0.5)
  eta <- 0.3 + 0.4 * x + b1[grp] + b2[grp] * x
  list(d = data.frame(y = rpois(n, exp(eta)), x = x, site = factor(grp)),
       b1 = b1, b2 = b2, G = G,
       # group-major, coefficient-within-group: the storage order under test
       truth = as.numeric(rbind(b1, b2)))
}

# Shared assertions for a per-group table that carries spread.
expect_ranef_recovers <- function(r, sim) {
  expect_equal(nrow(r), 2L * sim$G)
  expect_true(all(is.finite(r$estimate)))
  expect_true(all(is.finite(r$sd)) && all(r$sd > 0))
  expect_true(all(r$conf.low < r$estimate & r$estimate < r$conf.high))
  int <- grepl("^site\\[", r$term)
  slo <- grepl("^site\\.x\\[", r$term)
  expect_equal(sum(int), sim$G)
  expect_equal(sum(slo), sim$G)
  # Measured across seeds and both backends: intercepts 0.978-0.989, slopes
  # 0.909-0.967. A coefficient-major layout would put half the slopes on the
  # intercept rows and collapse both.
  expect_gt(cor(r$estimate[int], sim$b1), 0.90)
  expect_gt(cor(r$estimate[slo], sim$b2), 0.85)
}


# --- tier 1: the Gaussian-mixture summary the nested path reports through -----

test_that("a one-node mixture is exactly that Gaussian", {
  s <- tulpa:::.nl_gauss_mixture_summary(
    matrix(c(1.5, -2), 1L, 2L), matrix(c(0.25, 4), 1L, 2L), 1,
    probs = c(0.025, 0.5, 0.975))
  expect_equal(s$mean, c(1.5, -2))
  expect_equal(s$sd, c(0.5, 2))
  expect_equal(s$quantiles[1L, ], qnorm(c(0.025, 0.5, 0.975), 1.5, 0.5))
  expect_equal(s$quantiles[2L, ], qnorm(c(0.025, 0.5, 0.975), -2, 2))
})


test_that("mixture quantiles invert the mixture CDF and moments obey total variance", {
  set.seed(11)
  ng <- 7L; np <- 3L
  mu <- matrix(rnorm(ng * np, 0, 1.5), ng, np)
  va <- matrix(runif(ng * np, 0.05, 2), ng, np)
  w  <- runif(ng); w <- w / sum(w)
  probs <- c(0.025, 0.5, 0.975)
  s <- tulpa:::.nl_gauss_mixture_summary(mu, va, w, probs = probs)

  # F(q_p) = p at the returned quantiles, to bisection precision.
  for (j in seq_len(np)) {
    Fq <- vapply(s$quantiles[j, ], function(q)
      sum(w * pnorm(q, mu[, j], sqrt(va[, j]))), numeric(1))
    expect_equal(Fq, probs, tolerance = 1e-8)
  }
  # Exact mixture moments: E[x] and E[x^2] - E[x]^2 over the joint (node, x).
  expect_equal(s$mean, as.numeric(crossprod(w, mu)))
  expect_equal(s$sd, sqrt(as.numeric(crossprod(w, mu^2 + va)) - s$mean^2))
  # The within-node curvature is included, so the SD exceeds the between-node
  # spread a mode-only summary would report.
  between <- sqrt(as.numeric(crossprod(w, mu^2)) - s$mean^2)
  expect_true(all(s$sd > between))
})


test_that("mixture summary withholds spread when the variances are unusable", {
  set.seed(12)
  mu <- matrix(rnorm(12), 4L, 3L)
  va <- matrix(runif(12, 0.1, 1), 4L, 3L)
  w  <- rep(0.25, 4L)
  ref <- tulpa:::.nl_gauss_mixture_summary(mu, va, w)

  for (bad in list(NULL, replace(va, 5L, NA_real_), replace(va, 5L, -1))) {
    s <- tulpa:::.nl_gauss_mixture_summary(mu, bad, w)
    expect_equal(s$mean, ref$mean)                 # the mean still stands
    expect_true(all(is.na(s$sd)))                  # the spread does not
    expect_true(all(is.na(s$quantiles)))
  }
})


test_that("mixture summary drops zero-weight and non-finite nodes", {
  set.seed(13)
  mu <- matrix(rnorm(15), 5L, 3L)
  va <- matrix(runif(15, 0.1, 1), 5L, 3L)
  w  <- c(0.3, 0, 0.3, 0.2, 0.2)
  mu[4L, 1L] <- NA_real_                            # kills the whole node row
  s <- tulpa:::.nl_gauss_mixture_summary(mu, va, w)
  keep <- c(1L, 3L, 5L)
  wk <- w[keep] / sum(w[keep])
  expect_equal(s$mean, as.numeric(crossprod(wk, mu[keep, , drop = FALSE])))
  expect_null(tulpa:::.nl_gauss_mixture_summary(mu, va, rep(0, 5L)))
})


test_that("a stated ranef_unavailable reason is reported instead of an empty table", {
  fit <- structure(list(
    re_layout = list(list(group_var = "g", levels = c("1", "2"),
                          coef_labels = "(Intercept)")),
    n_fixed = 1L,
    ranef_unavailable = "this backend integrates b out by quadrature."),
    class = "tulpa_fit")
  expect_error(ranef(fit), "integrates b out by quadrature")
  expect_error(ranef(fit), "no per-group random effects")
})


# --- tier 2: the backends themselves -----------------------------------------

test_that("ranef() on a re_cov_nested fit reports the grid-marginalized group posterior", {
  skip_on_cran()
  sim <- make_ranef_re_data()
  fit <- tulpa(y ~ x + (1 + x | site), data = sim$d, family = "poisson",
               mode = "re_cov_nested")

  expect_equal(fit$backend, "re_cov_nested")
  r <- ranef(fit)
  expect_ranef_recovers(r, sim)

  # The estimate is the node-weighted conditional mean, and the reported spread is
  # the mixture's -- not one node's, and not the between-node scatter, which it
  # exceeds because the within-node curvature is in it too.
  w  <- fit$weights / sum(fit$weights)
  mu <- fit$re_nodes
  between <- sqrt(pmax(as.numeric(crossprod(w, mu^2)) -
                         as.numeric(crossprod(w, mu))^2, 0))
  expect_equal(r$estimate, as.numeric(crossprod(w, mu)))
  expect_true(all(r$sd > between))

  # Every node contributed a per-group mean and variance.
  expect_equal(dim(fit$re_nodes), c(fit$n_grid, 2L * sim$G))
  expect_equal(dim(fit$re_var_nodes), c(fit$n_grid, 2L * sim$G))
  expect_true(all(is.finite(fit$re_var_nodes) & fit$re_var_nodes > 0))

  # The identity the Details of this file open with: the mode pins the group mean
  # of each coefficient at zero, to the Newton tolerance.
  expect_equal(mean(r$estimate[grepl("^site\\[", r$term)]), 0,
               tolerance = 1e-4)
  expect_equal(mean(r$estimate[grepl("^site\\.x\\[", r$term)]), 0,
               tolerance = 1e-4)
})


test_that("ranef() on a re_cov_gibbs fit summarizes the sampled random effects", {
  skip_on_cran()
  sim <- make_ranef_re_data()
  fit <- tulpa(y ~ x + (1 + x | site), data = sim$d, family = "poisson",
               mode = "re_cov_gibbs",
               control = list(seed = 4L, n_iter = 500L, warmup = 250L))

  expect_equal(fit$backend, "re_cov_gibbs")
  r <- ranef(fit)
  expect_ranef_recovers(r, sim)

  # ranef() is the empirical summary of the recorded b draws, which are
  # row-aligned with the beta draws (one joint state per recorded sweep).
  expect_equal(dim(fit$re), c(fit$n_kept, 2L * sim$G))
  expect_equal(nrow(fit$re), nrow(fit$draws))
  expect_equal(r$estimate, as.numeric(colMeans(fit$re)))
  expect_equal(r$sd, as.numeric(apply(fit$re, 2L, sd)))
  expect_equal(r$conf.low,
               as.numeric(apply(fit$re, 2L, quantile, 0.025)))
})


test_that("the RE-covariance backends agree with the EB mode on the group effects", {
  skip_on_cran()
  sim <- make_ranef_re_data()
  common <- list(formula = y ~ x + (1 + x | site), data = sim$d,
                 family = "poisson")
  nested <- ranef(do.call(tulpa, c(common, list(mode = "re_cov_nested"))))
  eb     <- ranef(do.call(tulpa, c(common, list(mode = "eb"))))
  gibbs  <- ranef(do.call(tulpa, c(common, list(
    mode = "re_cov_gibbs",
    control = list(seed = 4L, n_iter = 1000L, warmup = 500L)))))

  # Three independently written paths (grid mixture, plug-in mode, MCMC). A
  # per-group ordering error in any one breaks this, whatever the labels say.
  # nested and eb both condition at a mode, so they agree outright.
  expect_gt(cor(nested$estimate, eb$estimate), 0.999)
  expect_lt(max(abs(nested$estimate - eb$estimate)), 0.02)

  # The sampler does not pin the group-mean direction the mode-based paths fix at
  # zero -- it samples it, and that direction is confounded with the intercept.
  # So it agrees up to a SHARED shift per coefficient: removing the shift leaves
  # per-group residuals an order of magnitude smaller than the RE scale.
  int <- grepl("^site\\[", nested$term)
  slo <- grepl("^site\\.x\\[", nested$term)
  expect_gt(cor(gibbs$estimate, eb$estimate), 0.99)
  for (blk in list(int, slo)) {
    resid <- (gibbs$estimate - nested$estimate)[blk]
    expect_lt(sd(resid), 0.06)
  }
  # Independent estimators of the same per-group posterior width.
  expect_lt(abs(mean(gibbs$sd) / mean(nested$sd) - 1), 0.25)
})


test_that("ranef() keeps the block order with several random-effect terms", {
  skip_on_cran()
  sim <- make_ranef_re_data()
  set.seed(21)
  n <- nrow(sim$d); H <- 8L
  rp <- rep_len(seq_len(H), n)
  c1 <- rnorm(H, 0, 0.6)
  grp <- as.integer(sim$d$site)
  eta <- 0.3 + 0.4 * sim$d$x + sim$b1[grp] + sim$b2[grp] * sim$d$x + c1[rp]
  d2 <- data.frame(y = rpois(n, exp(eta)), x = sim$d$x,
                   site = sim$d$site, rep = factor(rp))

  for (md in c("re_cov_nested", "re_cov_gibbs")) {
    fit <- tulpa(y ~ x + (1 + x | site) + (1 | rep), data = d2,
                 family = "poisson", mode = md,
                 control = if (md == "re_cov_gibbs")
                   list(seed = 5L, n_iter = 500L, warmup = 250L) else list())
    r <- ranef(fit)
    expect_equal(nrow(r), 2L * sim$G + H, info = md)
    # The site block comes first (both its coefficients interleaved per group),
    # then the rep block -- term-major, group, coefficient.
    expect_equal(head(r$term, 4L),
                 c("site[1]", "site.x[1]", "site[2]", "site.x[2]"), info = md)
    expect_equal(tail(r$term, 1L), sprintf("rep[%d]", H), info = md)
    expect_gt(cor(r$estimate[grepl("^site\\[", r$term)], sim$b1), 0.85)
    expect_gt(cor(r$estimate[grepl("^site\\.x\\[", r$term)], sim$b2), 0.85)
    expect_gt(cor(r$estimate[grepl("^rep\\[", r$term)], c1), 0.85)
  }
})


test_that("the AGHQ inner marginal says why it has no group effects", {
  skip_on_cran()
  sim <- make_ranef_re_data(G = 10L, npg = 20L)
  fit <- tulpa(y ~ x + (1 + x | site), data = sim$d, family = "poisson",
               mode = "re_cov_nested", control = list(re_cov = "aghq"))

  expect_null(fit$re_nodes)
  expect_type(fit$ranef_unavailable, "character")
  expect_error(ranef(fit), "adaptive Gauss-Hermite")
  expect_error(ranef(fit), "n_quad = 1")
})


test_that("a direct tulpa_re_cov_gibbs() call returns labelled random-effect draws", {
  skip_on_cran()
  set.seed(31)
  G <- 10L; per <- 20L; n <- G * per
  grp <- rep(seq_len(G), each = per)
  x <- rnorm(n)
  b <- cbind(rnorm(G, 0, 0.7), rnorm(G, 0, 0.5))
  eta <- -0.2 + 0.5 * x + b[grp, 1] + b[grp, 2] * x
  y <- rpois(n, exp(eta))
  re_term <- list(idx = grp, n_groups = G, n_coefs = 2L, Z = cbind(1, x),
                  correlated = TRUE, label = "g")
  fit <- tulpa_re_cov_gibbs(y, rep(1L, n), cbind(1, x), re_term,
                            family = "poisson",
                            control = list(seed = 2L, n_iter = 400L,
                                           warmup = 200L))
  expect_equal(dim(fit$re), c(fit$n_kept, G * 2L))
  expect_equal(colnames(fit$re)[1:3], c("g[1,1]", "g[1,2]", "g[2,1]"))
  expect_gt(cor(colMeans(fit$re), as.numeric(t(b))), 0.7)
})
