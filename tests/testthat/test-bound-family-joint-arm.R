# The two families whose density reads a per-observation BOUND rather than the
# point response -- truncated_gaussian (a ceiling) and interval_gaussian (a
# class interval) -- reach their closed forms through the response payload's
# bound arrays, not through the mu-space ladder. Two consequences the joint arm
# path has to honour, one on each side of the same branch:
#
#   - the spec constructor's "is this family compiled" gate asks the mu-space
#     ladder, which does not carry either family, so a gate reading it alone
#     rejects a family that is in fact compiled;
#   - a payload declaring one of these families but carrying no bound falls PAST
#     the callbacks' bound branches into that same ladder, which raises from
#     inside an OpenMP reduction -- std::terminate, not an R error.
#
# Both are checked on the calling thread, so the fit either runs or names what
# is missing.

bound_arm_fit <- function(arm_extra, family, y, n_group = 6L, per_group = 4L,
                          seed = 3L) {
  set.seed(seed)
  region <- rep(seq_len(n_group), each = per_group)
  X <- cbind(1, stats::rnorm(n_group)[region])
  ctl <- list(max_iter = 200L, tol = 1e-9, n_threads = 1L,
              diagnose_k = FALSE, diagnose_skew = FALSE,
              auto_recenter = FALSE, progress = FALSE)
  arm <- c(list(y = y, X = X, family = family, phi = 0.6), arm_extra)
  suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(arm),
    prior = list(list(type = "iid", obs_idx = as.integer(region),
                      n_units = n_group,
                      sigma_grid = c(0.3, 0.6, 1.2))),
    control = ctl))
}

trunc_y <- function(n) -abs(stats::rnorm(n, 0.4, 0.5))   # log-cover, ceiling 0


test_that("a truncated_gaussian arm carrying its ceiling fits", {
  set.seed(3L)
  y <- trunc_y(24L)
  fit <- bound_arm_fit(list(trunc_upper = rep(0, 24L)), "truncated_gaussian", y)
  expect_true(all(is.finite(fit$log_marginal)))
  expect_true(all(is.finite(fit$beta_mean)))
})


test_that("an interval_gaussian arm carrying its bounds fits", {
  set.seed(3L)
  lo <- rep(c(-Inf, -1, 0), length.out = 24L)
  hi <- rep(c(-1, 0, Inf), length.out = 24L)
  fit <- bound_arm_fit(list(lower = lo, upper = hi), "interval_gaussian",
                       rep(0, 24L))
  expect_true(all(is.finite(fit$log_marginal)))
  expect_true(all(is.finite(fit$beta_mean)))
})


test_that("a bound-reading family with no bound names the bound it needs", {
  set.seed(3L)
  expect_error(
    bound_arm_fit(list(), "truncated_gaussian", trunc_y(24L)),
    "trunc_upper")
  expect_error(
    bound_arm_fit(list(), "interval_gaussian", rep(0, 24L)),
    "lower")
})
