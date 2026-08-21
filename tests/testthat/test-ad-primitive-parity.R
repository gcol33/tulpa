# Value and derivative parity of the shared scalar primitives across the four
# scalar types the templated log posterior is instantiated for (gcol33/tulpa#446).
#
# The runtime gradient check differences the double instantiation against a
# reverse-mode one as though they were one function, so a primitive that
# computes a different value -- or a different partial -- on one type is a
# divergence that check cannot see: it is comparing the two sides of it.

test_that("expm1 keeps its precision on every type, not just the double one", {
  # exp(x) - 1 loses about half its significant digits as x -> 0; expm1 does
  # not. Comparing against the R primitive is the arbiter, not against the
  # cancelling form.
  for (x in c(-1e-10, -1e-8, -1e-4, -0.5, 0.5, 2)) {
    g <- cpp_test_scalar_guard("expm1", x)
    ref <- expm1(x)
    expect_equal(g$value_double, ref, tolerance = 1e-15, info = paste("x =", x))
    expect_equal(g$value_dual,   ref, tolerance = 1e-15, info = paste("x =", x))
    expect_equal(g$value_tape,   ref, tolerance = 1e-15, info = paste("x =", x))
    expect_equal(g$value_arena,  ref, tolerance = 1e-15, info = paste("x =", x))
    # d/dx expm1(x) = exp(x)
    expect_equal(g$grad_dual,  exp(x), tolerance = 1e-12)
    expect_equal(g$grad_tape,  exp(x), tolerance = 1e-12)
    expect_equal(g$grad_arena, exp(x), tolerance = 1e-12)
  }
})

test_that("log1m_exp agrees across types in the small-argument branch", {
  # log(1 - exp(-a)). The branch at a <= log(2) goes through expm1 precisely to
  # avoid the cancellation in log1p(-exp(-a)); a cancelling expm1 puts it back.
  for (a in c(1e-10, 1e-8, 1e-4, 0.5, 0.6931471805599453, 1, 40)) {
    g <- cpp_test_scalar_guard("log1m_exp", a)
    ref <- log(-expm1(-a))
    expect_equal(g$value_double, ref, tolerance = 1e-12, info = paste("a =", a))
    expect_equal(g$value_dual,   ref, tolerance = 1e-12, info = paste("a =", a))
    expect_equal(g$value_tape,   ref, tolerance = 1e-12, info = paste("a =", a))
    expect_equal(g$value_arena,  ref, tolerance = 1e-12, info = paste("a =", a))
    # d/da log(1 - exp(-a)) = exp(-a) / (1 - exp(-a))
    d <- exp(-a) / (-expm1(-a))
    expect_equal(g$grad_dual,  d, tolerance = 1e-6, info = paste("a =", a))
    expect_equal(g$grad_tape,  d, tolerance = 1e-6, info = paste("a =", a))
    expect_equal(g$grad_arena, d, tolerance = 1e-6, info = paste("a =", a))
  }
})

test_that("safe_log reports one derivative in the clamped region", {
  # At or below zero every overload returns the same constant value, so the
  # implemented function is locally flat and its derivative is 0. A reverse
  # partial of 1/1e-15 there is a 1e15 multiplier on a region the double
  # reference finite-differences as flat.
  for (x in c(-1, -1e-12, 0)) {
    g <- cpp_test_scalar_guard("log", x)
    expect_equal(g$value_double, g$value_tape,  tolerance = 0)
    expect_equal(g$value_double, g$value_arena, tolerance = 0)
    expect_equal(g$value_double, g$value_dual,  tolerance = 0)
    expect_equal(g$grad_dual,  0, tolerance = 0, info = paste("x =", x))
    expect_equal(g$grad_tape,  0, tolerance = 0, info = paste("x =", x))
    expect_equal(g$grad_arena, 0, tolerance = 0, info = paste("x =", x))
  }
  # Above the clamp the three still agree with each other and with 1/x.
  for (x in c(1e-13, 0.5, 3)) {
    g <- cpp_test_scalar_guard("log", x)
    expect_equal(g$grad_dual,  1 / x, tolerance = 1e-12)
    expect_equal(g$grad_tape,  1 / x, tolerance = 1e-12)
    expect_equal(g$grad_arena, 1 / x, tolerance = 1e-12)
  }
})

test_that("log_sum_exp returns -Inf, not NaN, when both components underflow", {
  g <- cpp_test_lse_guard(-Inf, -Inf)
  expect_equal(g$value_double, -Inf)
  expect_equal(g$value_dual,   -Inf)
  expect_equal(g$value_tape,   -Inf)
  expect_equal(g$value_arena,  -Inf)
  # and no NaN reaches the adjoint buffer
  expect_false(is.nan(g$grad_a_dual))
  expect_false(is.nan(g$grad_a_tape))
  expect_false(is.nan(g$grad_b_tape))
  expect_false(is.nan(g$grad_a_arena))
  expect_false(is.nan(g$grad_b_arena))
})

test_that("log_sum_exp agrees across types on ordinary arguments", {
  for (ab in list(c(0, 0), c(-3, 2), c(700, 699), c(-800, -801))) {
    a <- ab[1]; b <- ab[2]
    ref <- max(a, b) + log(exp(a - max(a, b)) + exp(b - max(a, b)))
    g <- cpp_test_lse_guard(a, b)
    expect_equal(g$value_double, ref, tolerance = 1e-12)
    expect_equal(g$value_dual,   ref, tolerance = 1e-12)
    expect_equal(g$value_tape,   ref, tolerance = 1e-12)
    expect_equal(g$value_arena,  ref, tolerance = 1e-12)
    # softmax weights
    wa <- exp(a - ref)
    expect_equal(g$grad_a_dual,  wa, tolerance = 1e-10)
    expect_equal(g$grad_a_tape,  wa, tolerance = 1e-10)
    expect_equal(g$grad_a_arena, wa, tolerance = 1e-10)
  }
})
