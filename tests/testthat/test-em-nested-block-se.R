# A nested-prior EM block must expose a fixed-effect H_beta so the MI / Gibbs
# correction can report a real pooled SE instead of NA/NaN (#204).
# .fit_block_via_nested_laplace now retains the per-grid Hessians and attaches
# the grid-marginalized H_beta (law of total variance), the same summary()/vcov()
# use on a nested fit.

test_that("a nested-prior EM block attaches a finite grid-marginalized H_beta (#204)", {
  skip_on_cran()
  set.seed(3)
  S <- 10L; reps <- 8L
  unit <- rep(seq_len(S), each = reps); N <- length(unit)
  # Ring adjacency for the icar field.
  W <- matrix(0, S, S)
  for (i in seq_len(S)) { j <- i %% S + 1L; W[i, j] <- 1; W[j, i] <- 1 }
  grp <- rep(seq_len(4L), length.out = N)           # separate iid RE grouping
  x <- rnorm(N)
  y <- rbinom(N, 1, plogis(-0.2 + 0.7 * x))
  prior <- tulpa:::.spatial_spec_to_nl_prior(
    list(type = "icar", adjacency = W, spatial_idx = unit))
  block <- list(
    y = y, n_trials = rep(1L, N), X = cbind(1, x), family = "binomial",
    re_list = list(list(idx = grp, n_groups = 4L, sigma = 1, n_coefs = 1L)),
    prior = prior)

  fit <- tulpa:::.fit_block_via_nested_laplace(block, n_threads = 1L)
  expect_false(is.null(fit$H_beta))
  expect_equal(dim(fit$H_beta), c(2L, 2L))
  expect_true(all(is.finite(fit$H_beta)))

  # The correction path derives a finite SE from it (was NA/NaN before).
  withse <- tulpa:::.attach_beta_se(fit, n_fixed = 2L)
  expect_true(all(is.finite(withse$se)))
  expect_true(all(withse$se > 0))
})

# A block carrying a prior but no RE term must still reach the engine: re_idx
# carries one 0 ("no random effect") per observation, not a length-1 sentinel.
test_that("a nested-prior EM block with no RE term fits (gcol33/tulpaObs#148)", {
  skip_on_cran()
  set.seed(3)
  S <- 10L; reps <- 8L
  unit <- rep(seq_len(S), each = reps); N <- length(unit)
  W <- matrix(0, S, S)
  for (i in seq_len(S)) { j <- i %% S + 1L; W[i, j] <- 1; W[j, i] <- 1 }
  x <- rnorm(N)
  y <- rbinom(N, 1, plogis(-0.2 + 0.7 * x))
  prior <- tulpa:::.spatial_spec_to_nl_prior(
    list(type = "icar", adjacency = W, spatial_idx = unit))
  block <- list(
    y = y, n_trials = rep(1L, N), X = cbind(1, x), family = "binomial",
    prior = prior)

  fit <- tulpa:::.fit_block_via_nested_laplace(block, n_threads = 1L)
  expect_true(all(is.finite(fit$mode)))
  expect_equal(dim(fit$H_beta), c(2L, 2L))
})
