# ranef() on a random effect the subspace debias sampled (gcol33/tulpa#314).
#
# When the selector pulls a random-effect coordinate into S, that coordinate is
# moved by the Metropolis sampler at every integration node and its draws are
# what the fit reports through the fixed effects. Reporting the Gaussian
# mixture for it describes a distribution the fit did not use, so those
# coordinates are reported from their draws and the rest from the mixture, with
# `source` saying per row which.
#
# WHETHER THE SAMPLED REPORT IS THE BETTER ONE is measured, not assumed. Two
# arbiters were run while this landed, neither of them engine code:
#
#   1. Exact quadrature of the same target the nested path integrates (beta and
#      sigma both integrated out, PC hyperprior, no MCMC anywhere). On a
#      large-sigma Bernoulli fixture (G = 12, 3 per group, true SD 2.0) the
#      sampled report is closer on every summary: mean absolute error of the
#      posterior mean 0.0844 against the mixture's 0.1572, of the posterior SD
#      0.0517 against 0.1320, and of the interval endpoints 0.1547 against
#      0.3924. On two milder fixtures the mixture is the closer of the two on
#      the mean (0.0036 / 0.0360 against 0.0521 / 0.0590), the inner conditional
#      there being near-Gaussian at every node.
#   2. A tulpa_re_cov_gibbs() sweep, below.
#
# What holds across every fixture measured is the SHAPE: the per-group skewness
# of the sampled report correlates 0.92-0.99 with the exact (and 0.95-0.99 with
# the Gibbs sweep's), and the Gaussian mixture carries no per-group skewness of
# its own at all. That is the gate this file gets.


# --- tier 1: the recombination and the overlay -------------------------------

test_that(".re_cov_debias_coord_draws() mixes on the picks it is handed", {
  # The node mixture is the fixed-effect draws' own: given those picks, every
  # draw must land in ITS node's sampled set (mode + one of that node's
  # offsets), never in another node's. A re-drawn mixture would break this.
  nodes <- list(matrix(c(0.10, 0.20, 0.30,
                         5.10, 5.20, 5.30), 3, 2),
                matrix(c(-0.10, -0.20,
                         -5.10, -5.20), 2, 2))
  centers <- rbind(c(1, 10), c(100, 1000))
  picks <- c(1L, 1L, 2L, 2L, 1L, 2L)

  set.seed(1)
  out <- .re_cov_debias_coord_draws(picks, nodes, centers, cols = c(1L, 2L))
  expect_equal(dim(out), c(6L, 2L))
  for (d in seq_along(picks)) {
    k <- picks[d]
    allowed <- nodes[[k]] + rep(centers[k, ], each = nrow(nodes[[k]]))
    expect_true(any(abs(allowed[, 1] - out[d, 1]) < 1e-12))
    # both coordinates come from the SAME sampled row of that node
    r <- which.min(abs(allowed[, 1] - out[d, 1]))
    expect_equal(out[d, 2], allowed[r, 2])
  }
})


test_that(".re_cov_debias_coord_draws() falls back to the mode for a node that declined", {
  nodes <- list(matrix(c(0.1, 0.2), 2, 1), NULL)
  centers <- cbind(c(3, 7))
  picks <- c(1L, 2L, 2L, 1L)
  out <- .re_cov_debias_coord_draws(picks, nodes, centers, cols = 1L)
  expect_equal(out[picks == 2L, 1], c(7, 7))
  expect_true(all(out[picks == 1L, 1] %in% c(3.1, 3.2)))

  # No node sampled at all: nothing to report, so the caller keeps the mixture.
  expect_null(.re_cov_debias_coord_draws(picks, list(NULL, NULL), centers, 1L))
  expect_null(.re_cov_debias_coord_draws(picks, nodes, centers, cols = integer(0)))
})


test_that(".ranef_overlay_sampled() replaces only the named rows and stamps the source", {
  tab <- data.frame(term = paste0("g[", 1:4, "]"),
                    estimate = c(1, 2, 3, 4), sd = c(1, 1, 1, 1),
                    conf.low = c(0, 1, 2, 3), conf.high = c(2, 3, 4, 5),
                    source = "mixture", stringsAsFactors = FALSE)
  set.seed(2)
  D <- cbind(rnorm(500, 10, 2), rnorm(500, -10, 3))
  obj <- structure(list(re_debias_draws = D, re_debias_idx = c(2L, 4L)),
                   class = "tulpa_fit")
  out <- .ranef_overlay_sampled(tab, obj)

  expect_equal(out$source, c("mixture", "sampled", "mixture", "sampled"))
  expect_equal(out[c(1, 3), c("estimate", "sd", "conf.low", "conf.high")],
               tab[c(1, 3), c("estimate", "sd", "conf.low", "conf.high")])
  emp <- .ranef_empirical(D)
  expect_equal(out$estimate[c(2, 4)], emp$estimate)
  expect_equal(out$sd[c(2, 4)], emp$sd)
  expect_equal(out$conf.low[c(2, 4)], emp$conf.low)
  expect_equal(out$conf.high[c(2, 4)], emp$conf.high)

  # A fit carrying nothing sampled leaves the table alone.
  expect_identical(.ranef_overlay_sampled(tab, structure(list(),
                                                         class = "tulpa_fit")),
                   tab)
})


# --- tier 2: the fit itself --------------------------------------------------

rsd_data <- function(seed = 21L, G = 14L, per = 4L) {
  set.seed(seed)
  n <- G * per
  grp <- rep(seq_len(G), each = per)
  x <- rnorm(n)
  data.frame(y = rbinom(n, 1L, plogis(-2.5 + 0.8 * x + rnorm(G, 0, 0.7)[grp])),
             x = x, g = factor(grp))
}


test_that("ranef() reports a sampled random effect empirically and says so per row", {
  skip_on_cran()
  G <- 14L
  d <- rsd_data(G = G)
  fit <- tulpa(y ~ x + (1 | g), data = d, family = "binomial",
               mode = "re_cov_nested",
               control = list(seed = 3L,
                              subspace_debias = list(probe = seq_len(2L + G))))

  # The selector has to fire on a random effect, or the rest of this is vacuous.
  expect_gt(length(fit$re_debias_idx), 0L)
  # ... and it must NOT take every one, so both provenances are exercised.
  expect_lt(length(fit$re_debias_idx), G)
  expect_equal(ncol(fit$re_debias_draws), length(fit$re_debias_idx))

  r <- ranef(fit)
  expect_equal(nrow(r), G)
  expect_true("source" %in% names(r))
  sampled <- which(r$source == "sampled")
  expect_equal(sampled, sort(fit$re_debias_idx))
  expect_equal(setdiff(unique(r$source), "sampled"), "mixture")

  # Sampled rows ARE the empirical summary of the recorded draws -- the same
  # summary the Gibbs backend reports its own `fit$re` with.
  emp <- .ranef_empirical(fit$re_debias_draws)
  ord <- order(fit$re_debias_idx)
  expect_equal(r$estimate[sampled], emp$estimate[ord])
  expect_equal(r$sd[sampled], emp$sd[ord])
  expect_equal(r$conf.low[sampled], emp$conf.low[ord])
  expect_equal(r$conf.high[sampled], emp$conf.high[ord])

  # Untouched rows are still the Gaussian mixture, unchanged by the overlay.
  mx <- .nl_gauss_mixture_summary(fit$re_nodes, fit$re_var_nodes, fit$weights,
                                  probs = c(0.025, 0.975))
  rest <- setdiff(seq_len(G), sampled)
  expect_equal(r$estimate[rest], mx$mean[rest])
  expect_equal(r$sd[rest], mx$sd[rest])
  expect_equal(r$conf.low[rest], mx$quantiles[rest, 1L])
})


test_that("a fit with no random effect in S reports exactly what the plain fit reports", {
  skip_on_cran()
  d <- rsd_data()
  common <- list(formula = y ~ x + (1 | g), data = d, family = "binomial",
                 mode = "re_cov_nested")
  plain <- do.call(tulpa, c(common, list(control = list(seed = 3L))))
  # The default probe is the fixed effects, so S can contain no random effect.
  fixed_only <- do.call(tulpa, c(common, list(
    control = list(seed = 3L, subspace_debias = TRUE))))

  expect_gt(length(fixed_only$subspace_debias$idx), 0L)   # it did correct
  expect_true(all(fixed_only$subspace_debias$idx <= 2L))  # ... fixed effects only
  expect_null(fixed_only$re_debias_idx)
  expect_null(fixed_only$re_debias_draws)

  expect_identical(ranef(fixed_only), ranef(plain))
  expect_identical(ranef(plain)$source, rep("mixture", nrow(ranef(plain))))
})


# --- tier 3: recovery against the exact debias -------------------------------

test_that("the sampled per-group posterior tracks a full Gibbs debias", {
  skip_if_not_slow()
  # A well-identified fixture (G = 30, 8 per group, true RE SD 1.5). The two
  # backends put DIFFERENT hyperpriors on Sigma -- the nested path is flat in
  # log sigma by default, the Gibbs sweep conjugate inverse-Wishart -- so the
  # comparison is only clean where the data, not the prior, fixes the scale.
  # On the small fixtures above that mismatch dominates and the two disagree by
  # more than either differs from the exact answer.
  set.seed(51L)
  G <- 30L; per <- 8L; n <- G * per
  grp <- rep(seq_len(G), each = per)
  x <- rnorm(n)
  y <- rbinom(n, 1L, plogis(-1.0 + 0.5 * x + rnorm(G, 0, 1.5)[grp]))
  X <- cbind(1, x); nt <- rep(1L, n)
  rt <- list(idx = grp, n_groups = G, n_coefs = 1L)

  fs <- tulpa_re_cov_nested(y, nt, X, rt, family = "binomial",
        control = list(seed = 3L, n_draws = 20000L,
                       subspace_debias = list(idx = seq_len(2L + G),
                                              n_iter = 20000L)))
  fp <- tulpa_re_cov_nested(y, nt, X, rt, family = "binomial",
                            control = list(seed = 3L, n_draws = 20000L))
  fg <- tulpa_re_cov_gibbs(y, nt, X, rt, family = "binomial",
                           control = list(seed = 4L, n_iter = 20000L,
                                          warmup = 4000L))

  expect_equal(length(fs$re_debias_idx), G)      # every group was sampled
  S  <- fs$re_debias_draws
  mx <- .nl_gauss_mixture_summary(fp$re_nodes, fp$re_var_nodes, fp$weights,
                                  probs = c(0.025, 0.975))
  gm <- colMeans(fg$re); gs <- apply(fg$re, 2L, stats::sd)
  skew <- function(v) mean((v - mean(v))^3) / stats::sd(v)^3
  gk <- apply(fg$re, 2L, skew)

  # The mode-based grid pins sum_g b_g = 0; the sampler samples that direction,
  # where it is confounded with the intercept. So the comparison is up to one
  # shared shift, as it is throughout test-ranef-re-cov.R.
  shift <- function(a, b) a - mean(a) - (b - mean(b))

  # Location. Measured over three data seeds: sampled 0.048 / 0.060 / 0.028
  # against the mixture's 0.080 / 0.179 / 0.027, on an RE scale of 1.5.
  expect_gt(cor(colMeans(S), gm), 0.99)
  expect_lt(stats::sd(shift(colMeans(S), gm)), 0.15)

  # Spread. Sampled 1.017 / 1.010 / 1.008 of the Gibbs width across those seeds,
  # the mixture 0.971 / 0.940 / 0.982 -- the sampled report is the closer of the
  # two to the exact debias on every one.
  r_samp <- mean(apply(S, 2L, stats::sd)) / mean(gs)
  r_mix  <- mean(mx$sd) / mean(gs)
  expect_lt(abs(r_samp - 1), 0.06)
  expect_lt(abs(r_samp - 1), abs(r_mix - 1) + 0.01)

  # Shape, which the Gaussian mixture has no way to report at all: 0.952 /
  # 0.967 / 0.960 across those seeds, and 0.92-0.99 against exact quadrature on
  # the fixtures in the header.
  expect_gt(cor(apply(S, 2L, skew), gk), 0.85)
})
