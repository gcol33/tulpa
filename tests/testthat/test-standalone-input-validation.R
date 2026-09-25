# Input validation on the standalone fitters (gcol33/tulpa#886): invalid inputs
# are refused by name at the door instead of surfacing as an optimizer / C++
# failure, and incomplete rows are refused rather than dropped in silence.

.sv_glmm <- function() {
  set.seed(1)
  J <- 20L
  g <- rep(seq_len(J), each = 6)
  x <- rnorm(length(g))
  y <- rbinom(length(g), 1, plogis(0.3 + 0.5 * x + rnorm(J)[g]))
  list(y = y, X = cbind(1, x), g = g, J = J,
       rt = list(idx = g, n_groups = J, n_coefs = 1L))
}

test_that("a family typo is named at every standalone door", {
  s <- .sv_glmm()
  msg <- "Unknown family 'binomal'"
  expect_error(tulpa_eb(s$y, NULL, s$X, s$rt, family = "binomal"), msg)
  expect_error(tulpa_laplace(s$y, rep(1L, length(s$y)), s$X,
                             family = "binomal"), msg)
  expect_error(tulpa_nested_laplace(s$y, rep(1L, length(s$y)), s$X,
                                    prior = list(type = "iid", obs_idx = s$g,
                                                 n_units = s$J),
                                    family = "binomal"), msg)
})

test_that("n_quad must be a whole number (tulpa_eb, re_cov_nested, agq_fit)", {
  s <- .sv_glmm()
  msg <- "`n_quad` must be a single whole number >= 1; got 2.5"
  expect_error(tulpa_eb(s$y, NULL, s$X, s$rt, n_quad = 2.5), msg)
  expect_error(tulpa_re_cov_nested(s$y, NULL, s$X, s$rt, n_quad = 2.5), msg)
  expect_error(agq_fit(s$y, s$X, s$g, n_quad = 2.5), msg)
  expect_error(agq_fit(s$y, s$X, s$g, n_quad = 0),
               "`n_quad` must be a single whole number >= 1; got 0")
})

test_that("agq_fit refuses NA, y > n_trials and non-positive scales", {
  s <- .sv_glmm()
  expect_error(agq_fit(replace(s$y, 1, NA), s$X, s$g),
               "agq_fit: Non-finite value\\(s\\) in the response")
  expect_error(agq_fit(s$y * 3L, s$X, s$g, family = "binomial"),
               "requires a 0/1 response")
  expect_error(agq_fit(s$y * 3L, s$X, s$g, family = "binomial",
                       n_trials = rep(2L, length(s$y))),
               "requires `y <= n_trials`")
  expect_error(agq_fit(s$y, s$X, replace(s$g, 2, NA)),
               "`group` has missing value\\(s\\) \\(first at row 2\\)")
  for (v in c(-1, 0)) {
    expect_error(agq_fit(s$y, s$X, s$g, sigma_init = v),
                 "`sigma_init` must be a single positive number")
  }
  expect_error(agq_fit(s$y, s$X, s$g, family = "gaussian", sigma_eps = -1),
               "`sigma_eps` must be a single positive number")
  expect_error(agq_fit(s$y, s$X, s$g, beta_init = 0),
               "length\\(beta_init\\) \\(1\\) must equal ncol\\(X\\) \\(2\\)")
})

.sv_ordinal <- function(n = 300L) {
  set.seed(1)
  x <- rnorm(n)
  Fm <- plogis(outer(-0.8 * x, c(-1, 0.5, 2), "+"))
  P <- cbind(Fm, 1) - cbind(0, Fm)
  y <- ordered(apply(P, 1, function(pr) sample.int(4L, 1L, prob = pr)))
  data.frame(y = y, x = x)
}

test_that("ordinal / multinomial refuse NA rows and empty response levels", {
  d <- .sv_ordinal()
  dn <- d
  dn$x[4] <- NA
  expect_error(tulpa_ordinal(y ~ x, dn),
               "tulpa_ordinal: Non-finite value\\(s\\) in the model matrix")
  expect_error(tulpa_multinomial(y ~ x, dn),
               "tulpa_multinomial: Non-finite value\\(s\\) in the model matrix")
  dn <- d
  dn$y[2] <- NA
  expect_error(tulpa_ordinal(y ~ x, dn), "Non-finite value\\(s\\) in the response")

  de <- d
  de$y <- factor(d$y, levels = 1:5, ordered = TRUE)       # empty top level
  expect_error(tulpa_ordinal(y ~ x, de),
               "response level\\(s\\) '5' have no observations")
  de$y <- factor(as.character(d$y), levels = c(1, 2, 9, 3, 4))  # empty middle
  expect_error(tulpa_multinomial(y ~ x, de),
               "response level\\(s\\) '9' have no observations")
})

test_that("tulpa_ordinal fits the cutpoints-only model y ~ 1", {
  skip_on_cran()
  d <- .sv_ordinal()
  f <- tulpa_ordinal(y ~ 1, d)
  emp <- stats::qlogis(cumsum(table(d$y)) / nrow(d))[1:3]
  expect_equal(unname(f$cutpoints), unname(emp), tolerance = 0.01)
  expect_length(coef(f), 0L)
})

test_that("mala() refuses a gradient of the wrong length", {
  lp <- function(t) -0.5 * sum(t^2)
  expect_error(mala(lp, function(t) -t[1], init = c(0, 0), n_iter = 10,
                    warmup = 5),
               "must return a numeric vector of length 2 .* of length 1")
  expect_error(mala(lp, function(t) c(NA, 0), init = c(0, 0), n_iter = 10,
                    warmup = 5),
               "`grad_log_posterior\\(init\\)` is not finite")
})
