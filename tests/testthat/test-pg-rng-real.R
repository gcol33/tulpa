# PG(b, z) at real shape b: rpg_real composes an exact integer-part draw
# with the truncated sum-of-gammas fractional part (tail-mean corrected).
# Closed-form moments of PG(b, c):
#   E    = b / (2c) * tanh(c / 2)
#   Var  = b * (sinh(c) - c) / (4 c^3 * cosh(c/2)^2)
# (limits b/4 and b/24 at c = 0).

test_that("rpg_real matches PG(b, c) moments at fractional shapes", {
  skip_on_cran()
  set.seed(707)
  n <- 200000L

  for (case in list(list(b = 0.7, c = 1.3), list(b = 2.6, c = 0.0),
                    list(b = 5.4, c = 2.5))) {
    b <- case$b; cc <- case$c
    x <- cpp_rpg_real(rep(b, n), rep(cc, n))
    if (cc == 0) {
      m_true <- b / 4
      v_true <- b / 24
    } else {
      m_true <- b / (2 * cc) * tanh(cc / 2)
      v_true <- b * (sinh(cc) - cc) / (4 * cc^3 * cosh(cc / 2)^2)
    }
    expect_lt(abs(mean(x) / m_true - 1), 0.01)
    expect_lt(abs(var(x) / v_true - 1), 0.03)
    expect_true(all(x > 0))
  }

  # Integer shapes must agree with the integer sampler's distribution.
  y_int  <- cpp_rpg(rep(3L, n), rep(1.0, n))
  y_real <- cpp_rpg_real(rep(3.0, n), rep(1.0, n))
  expect_lt(abs(mean(y_real) / mean(y_int) - 1), 0.01)
  expect_lt(abs(var(y_real) / var(y_int) - 1), 0.03)
})
