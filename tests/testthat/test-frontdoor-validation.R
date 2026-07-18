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

test_that("statistical hyperpriors ride re_prior, not control (#156b)", {
  set.seed(1)
  d <- data.frame(y = rbinom(120, 1, 0.5), x = rnorm(120),
                  g = factor(rep(1:12, 10)))
  # A statistical hyperprior in control is now rejected...
  expect_error(
    tulpa(y ~ x + (1 + x | g), data = d, family = "binomial",
          mode = "laplace", control = list(prior_sigma = c(2, 0.05))),
    "Unknown control knob"
  )
  # ...and a misspelled re_prior key is caught rather than silently ignored.
  expect_error(
    tulpa(y ~ x + (1 | g), data = d, family = "binomial",
          re_prior = list(pri_sigma = 1)),
    "Unknown control knob"
  )
  # A valid re_prior is accepted on the re-covariance path.
  fit <- tulpa(y ~ x + (1 + x | g), data = d, family = "binomial",
               mode = "laplace", re_prior = list(prior_sigma = c(2, 0.05),
                                                  eta = 3))
  expect_identical(fit$backend, "re_cov_nested")
})

test_that("a no-intercept single slope (0 + x | g) is rejected on intercept-only paths (#195)", {
  set.seed(1)
  d <- data.frame(y = rbinom(120, 1, 0.5), x = rnorm(120),
                  g = factor(rep(1:12, 10)))
  # (0 + x | g) has n_coefs == 1 but has_intercept == FALSE: it rides its slope
  # column as a design. The group-index-only paths (AGQ, Gibbs, SPDE, the
  # nested-Laplace native RE) cannot represent that and must reject it clearly
  # rather than silently fit it as a random intercept.
  expect_error(
    tulpa(y ~ x + (0 + x | g), data = d, family = "binomial", mode = "agq"),
    "random-intercept"
  )
  expect_error(
    tulpa(y ~ x + (0 + x | g), data = d, family = "binomial", mode = "gibbs"),
    "random-intercept"
  )
})

test_that("plain Laplace still fits a single random slope (0 + x | g) (#195)", {
  skip_on_cran()
  set.seed(1)
  d <- data.frame(y = rbinom(120, 1, 0.5), x = rnorm(120),
                  g = factor(rep(1:12, 10)))
  # The scalar-sigma_re Laplace path carries the slope column as Z, so it is a
  # valid 1-variance-component fit and must NOT be rejected or redirected to the
  # RE-covariance integrator.
  fit <- tulpa(y ~ x + (0 + x | g), data = d, family = "binomial",
               mode = "laplace")
  expect_identical(fit$backend, "laplace")
})

test_that("spatiotemporal summary aligns s/t labels with the array when S != T (#197)", {
  S <- 3L; T <- 2L; nd <- 4L
  # Draws are stored s-fastest: the column for (s, t) is (t - 1) * S + s. Give
  # each column a distinct constant so a mispaired label is visible.
  draws <- matrix(rep(seq_len(S * T), each = nd), nrow = nd)
  obj <- structure(
    list(spatiotemporal = list(n_spatial = S, n_times = T),
         .internal = list(spatiotemporal_draws = draws)),
    class = "tulpa_fit")
  arr  <- spatiotemporal_effects(obj, format = "array")
  summ <- spatiotemporal_effects(obj, format = "summary")
  for (i in seq_len(S)) for (j in seq_len(T)) {
    row <- summ[summ$s == i & summ$t == j, ]
    expect_equal(row$mean, mean(arr[i, j, ]))
  }
})

test_that("plot.tulpa_st_summary hard-errors on an unknown type (#156g)", {
  st <- structure(
    data.frame(s = rep(1:3, each = 2), t = rep(1:2, 3), mean = rnorm(6)),
    n_spatial = 3L, n_times = 2L,
    class = c("tulpa_st_summary", "data.frame"))
  expect_error(plot(st, type = "not_a_type"), "Unknown plot")
})
