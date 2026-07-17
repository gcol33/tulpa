# Front-door input validation (#156f) and hard-errors on silently-ignored
# inputs (#156g). These are cheap structural checks (no fitting), so they run
# on every profile.

test_that("flagship drivers reject a mismatched design (#156f)", {
  y <- rbinom(20, 1, 0.5)
  region <- rep(1:5, length.out = 20)
  prior <- list(list(type = "iid", obs_idx = region, n_units = 5L,
                     sigma_grid = c(0.5, 1, 2)))
  X <- cbind(1, rnorm(19))                       # nrow(X) != length(y)
  expect_error(
    tulpa_nested_laplace(y = y, n_trials = rep(1L, 20), X = X, prior = prior),
    "nrow\\(X\\)"
  )
  # n_trials length mismatch.
  X2 <- cbind(1, rnorm(20))
  expect_error(
    tulpa_gibbs(y = y, n_trials = rep(1L, 5), X = X2,
                group = rep(1L, 20), n_groups = 1L, family = "binomial"),
    "length\\(n_trials\\)"
  )
  # group out of range.
  expect_error(
    tulpa_gibbs(y = y, n_trials = rep(1L, 20), X = X2,
                group = c(3L, rep(1L, 19)), n_groups = 1L, family = "binomial"),
    "group"
  )
})

test_that("tulpa_em_laplace hard-errors on the ignored spatial/re_list (#156g)", {
  e_step <- function(fits, ...) list(weights = 1)
  m_step <- function(weights, ...) list()
  expect_error(
    tulpa_em_laplace(e_step, m_step, spatial = list(type = "icar")),
    "not consumed by tulpa_em_laplace"
  )
  expect_error(
    tulpa_em_laplace(e_step, m_step, re_list = list(list(idx = 1))),
    "not consumed by tulpa_em_laplace"
  )
})

test_that("plot.tulpa_st_summary hard-errors on an unknown type (#156g)", {
  st <- structure(
    data.frame(s = rep(1:3, each = 2), t = rep(1:2, 3), mean = rnorm(6)),
    n_spatial = 3L, n_times = 2L,
    class = c("tulpa_st_summary", "data.frame"))
  expect_error(plot(st, type = "not_a_type"), "Unknown plot")
})
