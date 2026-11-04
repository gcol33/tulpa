test_that("compact box curves equal the trapezoid CDF at every grid point", {
  skip_on_cran_budget()
  x <- seq(-4, 4, length.out = 4097L)
  cases <- list(list(lo = -.123, hi = .127, lg = 0),
                list(lo = -5, hi = 5, lg = 0),
                list(lo = c(-5, -.1, 1), hi = c(-1, .4, 5), lg = c(-3, 0, -1)),
                list(lo = c(-2, -1, .2), hi = c(.5, 1, 3), lg = c(0, -2, -5)))
  for (p in cases) {
    f <- Reduce(`+`, lapply(seq_along(p$lo), function(k)
      exp(p$lg[k] - max(p$lg)) * (x >= p$lo[k] & x < p$hi[k])))
    expected <- c(0, cumsum((f[-1L] + f[-length(f)]) / 2))
    expected <- expected / expected[length(expected)]
    row <- list(kind = "box", u = seq_along(p$lo), lg = p$lg,
                lo_b = p$lo, hi_b = p$hi, lo = min(p$lo), hi = max(p$hi))
    g <- tulpa:::cpp_nl_lq_group_cdf(row, x)
    got <- approx(x[g$index], g$F, xout = x, ties = "ordered", rule = 2)$y
    expect_equal(got, expected, tolerance = 1e-12)
  }
})

test_that("a conditional box draw remains inside its row's support", {
  x <- seq(-10, 10, length.out = 65545L)
  row <- list(kind = "box", u = 0, lg = 0, lo_b = -.1, hi_b = .1, lo = -.1, hi = .1)
  g <- tulpa:::cpp_nl_lq_group_cdf(row, x)
  p <- seq(.01, .99, length.out = 99L)
  got <- tulpa:::.nl_lq_inverse(x[g$index], g$F, p)
  expect_true(all(got >= -.101 & got <= .101))
  expect_lt(max(abs(got - qunif(p, -.1, .1))), 2 * diff(x)[1L])
})

test_that("a CDF mixture retains its measure past the dense matrix size", {
  skip_on_cran()
  m <- 65545L
  n <- 59000L
  expect_gt(as.double(m) * n, .Machine$integer.max)
  x <- seq(0, 1, length.out = m)
  row <- list(kind = "box", u = 0, lg = 0, lo_b = -1, hi_b = 2, lo = -1, hi = 2)
  mix <- tulpa:::cpp_nl_lq_mixture_cdf(rep(list(row), n), rep(1 / n, n), x)
  got <- mix$F
  expect_true(all(mix$usable))
  expect_equal(got, seq(0, 1, length.out = m), tolerance = 1e-12)
  expect_true(all(diff(got) >= 0))
})

test_that("broad conditional rows integrate the Gaussian CDF in bounded storage", {
  skip_on_cran()
  x <- seq(-8, 8, length.out = 4097L)
  row <- list(kind = "lq", u = c(-1, 0, 1), lg = c(-.5, 0, -.5),
              d1 = .5, d2 = -.5, lt = c(0, 0, 0),
              lo_b = c(-8, -.5, .5), hi_b = c(-.5, .5, 8), lo = -8, hi = 8)
  n <- 20000L
  mix <- tulpa:::cpp_nl_lq_mixture_cdf(rep(list(row), n), rep(1 / n, n), x)
  exact <- (pnorm(x) - pnorm(-8)) / (pnorm(8) - pnorm(-8))
  expect_true(all(mix$usable))
  expect_equal(mix$F, exact, tolerance = 1e-6)
})
