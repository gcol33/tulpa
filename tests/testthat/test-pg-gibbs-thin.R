# Thinning arithmetic in the Polya-Gamma Gibbs kernels (gcol33/tulpa#426).
#
# Every kernel saves on `(iter - n_warmup) %% thin == 0` and used to size its
# draw matrices with `(n_iter - n_warmup) / thin` truncated toward zero. When
# thin does not divide the post-warmup run the loop saves one row more than the
# matrices hold: Rcpp::NumericMatrix::operator() does no bounds check, so the
# extra write lands on row 0 of the next column for every column but the last,
# and one element past the allocation for the last. The saved value is a
# plausible finite number, so a corrupted fit returns silently.
#
# 100 iterations with 50 warmup and thin 3 saves at 50, 53, ..., 99: 17 rows,
# not the 16 the truncating division allocated.

.thin_expected <- function(n_iter, warmup, thin) {
  length(seq(warmup, n_iter - 1L)[(seq(warmup, n_iter - 1L) - warmup) %% thin == 0L])
}

.thin_fixture <- function(seed = 21L, n = 120L, n_groups = 6L) {
  set.seed(seed)
  x <- rnorm(n)
  X <- cbind(1, x)
  grp <- rep_len(seq_len(n_groups), n)
  b <- rnorm(n_groups, 0, 0.4)
  eta <- as.numeric(X %*% c(-0.2, 0.6)) + b[grp]
  list(X = X, grp = as.integer(grp), n_groups = as.integer(n_groups),
       y_binom = rbinom(n, 1L, plogis(eta)),
       y_count = rpois(n, exp(eta)),
       n_trials = rep(1L, n), n = n)
}

# A 4-unit chain, so an areal route has a real adjacency without costing much.
.thin_spatial <- function(n) {
  list(type = "icar",
       adjacency = matrix(c(0, 1, 0, 0,
                            1, 0, 1, 0,
                            0, 1, 0, 1,
                            0, 0, 1, 0), 4, 4, byrow = TRUE),
       spatial_idx = as.integer(rep_len(1:4, n)))
}

test_that("thin = 3 over 50 post-warmup iterations saves 17 rows, not 16", {
  expect_equal(.thin_expected(100L, 50L, 3L), 17L)

  fx <- .thin_fixture()
  ctl <- list(n_iter = 100L, warmup = 50L, thin = 3L, seed = 5L)

  routes <- list(
    binomial = function() tulpa_gibbs(
      fx$y_binom, fx$n_trials, fx$X, fx$grp, fx$n_groups,
      family = "binomial", control = ctl),
    neg_binomial_2 = function() tulpa_gibbs(
      fx$y_count, fx$n_trials, fx$X, fx$grp, fx$n_groups,
      family = "neg_binomial_2", control = ctl),
    binomial_icar = function() tulpa_gibbs(
      fx$y_binom, fx$n_trials, fx$X, fx$grp, fx$n_groups,
      family = "binomial", spatial = .thin_spatial(fx$n), control = ctl),
    negbin_icar = function() tulpa_gibbs(
      fx$y_count, fx$n_trials, fx$X, fx$grp, fx$n_groups,
      family = "neg_binomial_2", spatial = .thin_spatial(fx$n), control = ctl)
  )

  for (nm in names(routes)) {
    fit <- suppressWarnings(routes[[nm]]())
    expect_equal(nrow(fit$beta), 17L, info = nm)
    # The overflowing write used to land on row 0 of the next column, so the
    # first saved draw of every column but the first was replaced by the last
    # saved draw of its predecessor.
    expect_true(all(is.finite(fit$beta)), info = nm)
    expect_equal(anyDuplicated(round(fit$beta, 12)), 0L, info = nm)
  }
})

test_that("a thin that does divide the run is unaffected", {
  fx <- .thin_fixture(seed = 22L)
  fit <- tulpa_gibbs(fx$y_binom, fx$n_trials, fx$X, fx$grp, fx$n_groups,
                     family = "binomial",
                     control = list(n_iter = 100L, warmup = 50L, thin = 5L,
                                    seed = 5L))
  expect_equal(nrow(fit$beta), 10L)
  expect_equal(.thin_expected(100L, 50L, 5L), 10L)
})

test_that("thin reaches the spatial route rather than being dropped", {
  fx <- .thin_fixture(seed = 23L)
  base <- list(n_iter = 100L, warmup = 50L, seed = 5L)
  f1 <- suppressWarnings(tulpa_gibbs(
    fx$y_binom, fx$n_trials, fx$X, fx$grp, fx$n_groups, family = "binomial",
    spatial = .thin_spatial(fx$n), control = c(base, list(thin = 1L))))
  f5 <- suppressWarnings(tulpa_gibbs(
    fx$y_binom, fx$n_trials, fx$X, fx$grp, fx$n_groups, family = "binomial",
    spatial = .thin_spatial(fx$n), control = c(base, list(thin = 5L))))
  expect_equal(nrow(f1$beta), 50L)
  expect_equal(nrow(f5$beta), 10L)
  # Same chain, same seed: the thinned run is the every-fifth subsequence.
  expect_equal(f5$beta, f1$beta[seq(1L, 50L, by = 5L), , drop = FALSE])
})

test_that("a thin below 1 is refused rather than reaching the modulus", {
  fx <- .thin_fixture(seed = 24L)
  expect_error(
    tulpa_gibbs(fx$y_binom, fx$n_trials, fx$X, fx$grp, fx$n_groups,
                family = "binomial",
                control = list(n_iter = 100L, warmup = 50L, thin = 0L)),
    "thin"
  )
})
