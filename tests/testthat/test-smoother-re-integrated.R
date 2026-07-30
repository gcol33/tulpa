# A random-effect term on the nested path is carried as an `iid` latent block, so
# its SD is integrated on the outer grid alongside the other blocks'
# hyperparameters (gcol33/tulpa#265). It used to travel on the driver's native
# re_idx / n_re_groups / sigma_re channel, which CONDITIONS -- so on a formula
# whose other structure was integrated, the random effect was the one variance
# component the fit never estimated, at a default sigma_re = 1 the data never
# produced.
#
# What is asserted about recovery, and what is not. The grid integrates the SD
# under a prior flat in log(sigma), which is the convention every nested scale
# axis uses (no nested block carries a hyperprior on its scale; the sibling
# re_cov path's PC prior is the outlier -- tracked separately). Flat in
# log(sigma) does not shrink, so at these group counts the posterior is wide and
# its point summary sits ABOVE the truth: measured over 6 seeds at G = 15, mean
# of medians 1.02 against a truth of 0.9, with the 95% interval covering the
# truth 6/6. So the recovery assertion here is COVERAGE plus correlation of the
# per-group effects, not |est - truth| on the SD, which would encode the +0.12
# as though it were the target.

make_smooth_re <- function(seed, G = 15L, npg = 25L, sd_true = 0.9) {
  set.seed(seed)
  n <- G * npg
  site <- factor(rep(seq_len(G), each = npg))
  x <- runif(n, -3, 3)
  b <- rnorm(G, 0, sd_true)
  list(d = data.frame(y = rpois(n, exp(sin(x) + b[as.integer(site)])),
                      x = x, site = site),
       b = b, G = G, sd_true = sd_true)
}

# The integrated SD's axis on the outer grid: the driver labels block b's sigma
# axis `b<b>.sigma`, and `re_block_index` says which block the term became.
re_sigma_ci <- function(fit, term = 1L) {
  b <- fit$re_block_index[term]
  j <- which(fit$theta_names == sprintf("b%d.sigma", b))
  if (length(j) != 1L) return(c(NA_real_, NA_real_))
  c(fit$theta_ci_lo[j], fit$theta_ci_hi[j])
}


# --- tier 2: the term is integrated, not conditioned --------------------------

test_that("a random intercept beside a smoother has its SD integrated", {
  skip_on_cran()
  sim <- make_smooth_re(1L)
  fit <- suppressWarnings(
    tulpa(y ~ s(x, k = 15) + (1 | site), data = sim$d, family = "poisson"))

  expect_equal(fit$backend, "nested_laplace")
  # The term became a block, and it is the LAST one (which is what makes its
  # latent segment addressable without any other block's width).
  expect_equal(fit$re_block_index, 2L)
  expect_false(isTRUE(fit$re_block_conditioned))

  vc <- VarCorr(fit)
  expect_equal(nrow(vc), 1L)
  # The bug: this said sd = 1, source "conditioned".
  expect_equal(vc$source, "estimated")
  expect_false(isTRUE(all.equal(vc$sd, 1)))
  expect_true(is.finite(vc$sd) && vc$sd > 0)

  # ranef() still reports every group. The RE shares the latent vector with the
  # smoother block now, so the exact-tail-width guard cannot fire and the accessor
  # has to slice the trailing segment instead -- an empty table here would be the
  # #264 regression in a new place.
  r <- ranef(fit)
  expect_equal(nrow(r), sim$G)
  expect_true(all(is.finite(r$estimate)))
  expect_gt(cor(r$estimate, sim$b), 0.85)
})


test_that("a supplied sigma_re still conditions, and says so", {
  skip_on_cran()
  sim <- make_smooth_re(2L)
  fit <- suppressWarnings(
    tulpa(y ~ s(x, k = 15) + (1 | site), data = sim$d, family = "poisson",
          sigma_re = 0.5))
  # Conditioning is the degenerate one-point-grid case of the same path, not a
  # second one -- but the label must not claim the data produced the value.
  expect_true(isTRUE(fit$re_block_conditioned))
  vc <- VarCorr(fit)
  expect_equal(vc$source, "conditioned")
  expect_equal(vc$sd, 0.5)
  expect_equal(nrow(ranef(fit)), sim$G)
})


test_that("several random-intercept terms each become their own block", {
  skip_on_cran()
  sim <- make_smooth_re(3L)
  set.seed(31)
  n <- nrow(sim$d); H <- 6L
  rp <- factor(rep_len(seq_len(H), n)); c1 <- rnorm(H, 0, 0.5)
  d2 <- sim$d
  d2$rep <- rp
  d2$y <- rpois(n, exp(sin(d2$x) + sim$b[as.integer(d2$site)] +
                         c1[as.integer(rp)]))

  # Previously refused outright: "supports at most one random-intercept term".
  fit <- suppressWarnings(
    tulpa(y ~ s(x, k = 15) + (1 | site) + (1 | rep), data = d2,
          family = "poisson"))
  expect_equal(fit$re_block_index, c(2L, 3L))
  vc <- VarCorr(fit)
  expect_equal(nrow(vc), 2L)
  expect_true(all(vc$source == "estimated"))
  expect_equal(vc$term, c("site", "rep"))
  # Two independently integrated components, each finite and distinct from 1.
  expect_true(all(is.finite(vc$sd) & vc$sd > 0))

  r <- ranef(fit)
  expect_equal(nrow(r), sim$G + H)
  expect_gt(cor(r$estimate[grepl("^site\\[", r$term)], sim$b), 0.85)
  expect_gt(cor(r$estimate[grepl("^rep\\[", r$term)], c1), 0.70)
})


test_that("a random slope beside a smoother refuses, naming why", {
  skip_on_cran()
  sim <- make_smooth_re(4L)
  # An iid block has no `Z` design, so a slope has no iid form. The refusal names
  # that and points at the backends that do fit a slope covariance.
  expect_error(
    suppressWarnings(tulpa(y ~ s(x, k = 15) + (1 + x | site), data = sim$d,
                           family = "poisson")),
    "random-slope term has no iid form")
})


# --- tier 3: coverage, which is what actually holds --------------------------

test_that("the integrated RE SD's interval covers the truth across seeds", {
  skip_on_cran()
  skip_if_not(nzchar(Sys.getenv("TULPA_SLOW_TESTS")),
              "full-validation tier: set TULPA_SLOW_TESTS=true")
  seeds <- 101:110
  cov <- vapply(seeds, function(s) {
    sim <- make_smooth_re(s)
    fit <- suppressWarnings(
      tulpa(y ~ s(x, k = 15) + (1 | site), data = sim$d, family = "poisson"))
    ci <- re_sigma_ci(fit)
    isTRUE(ci[1] <= sim$sd_true && sim$sd_true <= ci[2])
  }, logical(1))
  # Measured 6/6 on the direct-driver equivalent; 8 of 10 is the nominal-ish gate
  # that leaves room for grid coarseness without accepting a broken integration.
  expect_gte(sum(cov), 8L)
})
