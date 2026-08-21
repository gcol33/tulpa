# The scalar primitives the templated log posterior is built from, evaluated on
# every scalar type it is instantiated for.
#
# compute_log_post runs as `double` (the value path the numerical gradient
# finite-differences) and as a reverse-mode Var (the live gradient), so the
# runtime cross-check compares two instantiations as though they were one
# function. That only holds if each primitive agrees in value AND in derivative
# across the types, at the guard boundaries as much as in the interior.

# f(x) and f'(x) as the value path computes them, for the primitives whose
# derivative is elementary.
.prim_ref <- function(fn, x, p = 2) {
  switch(fn,
    exp   = list(v = exp(min(max(x, -700), 700)),
                 g = exp(min(max(x, -700), 700))),
    log   = list(v = if (x > 0) log(x) else -1e10,
                 g = if (x > 0) 1 / x else 0),
    sqrt  = list(v = if (x > 0) sqrt(x) else 0,
                 g = if (x > 0) 0.5 / sqrt(x) else 0),
    pow   = list(v = x^p, g = p * x^(p - 1)),
    logit = list(v = log(x / (1 - x)), g = 1 / (x * (1 - x))),
    stop("unknown primitive")
  )
}

test_that("every scalar type evaluates the same primitive in the interior", {
  cases <- list(
    list(fn = "exp",   x = 1.3),
    list(fn = "exp",   x = -4.2),
    list(fn = "log",   x = 2.7),
    list(fn = "sqrt",  x = 9.0),
    list(fn = "pow",   x = 1.7, p = 2.5),
    list(fn = "logit", x = 0.32)
  )
  for (cs in cases) {
    p <- cs$p %||% 2
    r <- cpp_test_scalar_guard(cs$fn, cs$x, p)
    ref <- .prim_ref(cs$fn, cs$x, p)
    lbl <- sprintf("%s(%g)", cs$fn, cs$x)
    expect_equal(r$value_double, ref$v, tolerance = 1e-12, info = lbl)
    expect_equal(r$value_tape,   ref$v, tolerance = 1e-12, info = lbl)
    expect_equal(r$value_arena,  ref$v, tolerance = 1e-12, info = lbl)
    expect_equal(r$grad_tape,    ref$g, tolerance = 1e-10, info = lbl)
    expect_equal(r$grad_arena,   ref$g, tolerance = 1e-10, info = lbl)
    if (!is.na(r$grad_dual)) {
      expect_equal(r$value_dual, ref$v, tolerance = 1e-12, info = lbl)
      expect_equal(r$grad_dual,  ref$g, tolerance = 1e-10, info = lbl)
    }
  }
})

test_that("exp is clamped on every scalar type, not only the value path", {
  # exp(800) overflows to +Inf unclamped. The value path has always clamped, so
  # an unclamped reverse mode is a different function from the one the runtime
  # gradient check finite-differences.
  for (x in c(800, 1e4, -800, -1e4)) {
    r <- cpp_test_scalar_guard("exp", x)
    expect_true(is.finite(r$value_double))
    expect_true(is.finite(r$value_tape),  info = paste("tape at", x))
    expect_true(is.finite(r$value_arena), info = paste("arena at", x))
    expect_true(is.finite(r$grad_tape))
    expect_true(is.finite(r$grad_arena))
    expect_equal(r$value_arena, r$value_double, tolerance = 1e-12)
    expect_equal(r$value_tape,  r$value_double, tolerance = 1e-12)
  }
})

test_that("sqrt at and below zero is finite in value and derivative everywhere", {
  # An HSGP spectral density underflows to exactly 0 at a long lengthscale and a
  # high basis index, so sqrt(0) is an ordinary argument here. Unguarded, the
  # reverse-mode partial is 0.5 / 0 = +Inf and any nonzero upstream adjoint
  # makes the reported gradient Inf or NaN while the value path stays finite.
  for (x in c(0, -1e-12, -3.5)) {
    r <- cpp_test_scalar_guard("sqrt", x)
    expect_equal(r$value_double, 0)
    expect_equal(r$value_tape,   0)
    expect_equal(r$value_arena,  0)
    expect_equal(r$grad_tape,    0)
    expect_equal(r$grad_arena,   0)
    expect_equal(r$grad_dual,    0)
    expect_equal(r$value_dual,   0)
  }
})

test_that("log at and below zero returns the finite sentinel on every type", {
  for (x in c(0, -2)) {
    r <- cpp_test_scalar_guard("log", x)
    expect_equal(r$value_double, -1e10)
    expect_equal(r$value_tape,   -1e10)
    expect_equal(r$value_arena,  -1e10)
    expect_true(is.finite(r$grad_tape))
    expect_true(is.finite(r$grad_arena))
  }
})

test_that("pow keeps a finite partial where p * x^(p-1) diverges", {
  r <- cpp_test_scalar_guard("pow", 0, 0.5)
  expect_equal(r$value_tape, 0)
  expect_equal(r$value_arena, 0)
  expect_true(is.finite(r$grad_tape))
  expect_true(is.finite(r$grad_arena))
  # p > 1 is finite unguarded and must be untouched.
  r2 <- cpp_test_scalar_guard("pow", 0, 3)
  expect_equal(r2$grad_tape, 0)
  expect_equal(r2$grad_arena, 0)
})

test_that("logit is finite at the endpoints of the unit interval", {
  for (x in c(0, 1)) {
    r <- cpp_test_scalar_guard("logit", x)
    expect_true(is.finite(r$value_tape),  info = paste("tape at", x))
    expect_true(is.finite(r$value_arena), info = paste("arena at", x))
    expect_true(is.finite(r$grad_tape))
    expect_true(is.finite(r$grad_arena))
    expect_equal(r$value_tape, r$value_arena, tolerance = 1e-12)
    expect_equal(r$grad_tape,  r$grad_arena,  tolerance = 1e-12)
  }
})

# --- the tape primitives, each against its own closed-form derivative ---------

test_that("tape reverse mode reproduces every elementary derivative", {
  x <- 0.7
  chk <- function(res, info) {
    expect_equal(res$gradient, res$expected_gradient, tolerance = 1e-9,
                 info = info)
  }
  chk(cpp_test_autodiff_exp_chain(x), "exp(x^2)")
  chk(cpp_test_autodiff_log(x),       "log")
  chk(cpp_test_autodiff_sqrt(x),      "sqrt")
  chk(cpp_test_autodiff_pow(x, 3),    "pow")
  chk(cpp_test_autodiff_log1p(x),     "log1p")
  chk(cpp_test_autodiff_logit(x),     "logit")
  chk(cpp_test_autodiff_lgamma(2.3),  "lgamma")
  chk(cpp_test_autodiff_softplus(x),  "softplus")
  chk(cpp_test_autodiff_inv_logit(x), "inv_logit")
  chk(cpp_test_autodiff_division(3.0, 1.5), "division")
  chk(cpp_test_autodiff_log_sum_exp(1.2, -0.4), "log_sum_exp")
  chk(cpp_test_autodiff_negbin_loglik(3L, 2.5, 4.0), "negbin loglik")
})

test_that("tape values match the R closed forms", {
  expect_equal(cpp_test_autodiff_log(0.7)$value, log(0.7))
  expect_equal(cpp_test_autodiff_sqrt(0.7)$value, sqrt(0.7))
  expect_equal(cpp_test_autodiff_log1p(0.7)$value, log1p(0.7))
  expect_equal(cpp_test_autodiff_lgamma(2.3)$value, lgamma(2.3))
  expect_equal(cpp_test_autodiff_softplus(0.7)$value, log1p(exp(0.7)))
  expect_equal(cpp_test_autodiff_inv_logit(0.7)$value, plogis(0.7))
  expect_equal(cpp_test_autodiff_log_sum_exp(1.2, -0.4)$value,
               log(exp(1.2) + exp(-0.4)))
  # The helper omits the y-only normalizer -lgamma(y + 1), which carries no
  # gradient in mu or phi.
  expect_equal(cpp_test_autodiff_negbin_loglik(3L, 2.5, 4.0)$value,
               dnbinom(3, size = 4, mu = 2.5, log = TRUE) + lgamma(3 + 1))
})

test_that("tape gradients of a multivariate objective are exact", {
  x <- c(-1.5, 0.25, 3)
  g <- cpp_test_autodiff_gradient(x)
  expect_equal(g$value, sum(x^2))
  expect_equal(g$gradient, 2 * x)

  y <- c(0L, 2L, 5L)
  eta <- c(-0.3, 0.8, 1.4)
  ll <- cpp_test_autodiff_log_likelihood(y, eta)
  expect_equal(ll$value, sum(y * eta - exp(eta)))
  expect_equal(ll$gradient, ll$expected_gradient, tolerance = 1e-10)
  expect_equal(ll$gradient, y - exp(eta), tolerance = 1e-10)
})
