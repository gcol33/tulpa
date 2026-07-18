# Minor robustness / doc gaps from the 0.0.88 audit (#205, items 2-5).

test_that("binary `x - 1` in a bar LHS drops the intercept (#205)", {
  d <- decompose_bar_lhs(quote(x - 1))
  expect_false(d$has_intercept)
  expect_length(d$slope_terms, 1L)
  expect_identical(deparse(d$slope_terms[[1]]), "x")
  # The idiomatic `0 + x` is unchanged, and `1 + x` keeps the intercept.
  expect_false(decompose_bar_lhs(quote(0 + x))$has_intercept)
  expect_true(decompose_bar_lhs(quote(1 + x))$has_intercept)
})

test_that("spatial_multiscale sampler rejects undocumented modes (#205)", {
  # match.arg rejects the undocumented modes before `coords` is even touched.
  expect_error(spatial_multiscale(sampler = "riemannian"), "should be one of")
  expect_error(spatial_multiscale(sampler = "lbfgs"), "should be one of")
  # The accepted set is exactly the documented four.
  accepted <- eval(formals(spatial_multiscale)$sampler)
  expect_setequal(accepted,
                  c("auto", "noncentered", "centered", "interweaved"))
})

test_that("ACF / Geweke handle a 3-D [iter, chain, param] chain fit (#205)", {
  set.seed(1)
  arr <- array(rnorm(100 * 2 * 3), dim = c(100, 2, 3),
               dimnames = list(NULL, NULL, c("a", "b", "c")))
  fit <- structure(list(draws = arr, draws_kind = "chain", n_chains = 2L),
                   class = "tulpa_fit")
  pd <- tulpa:::.tulpa_pooled_draws(fit)
  expect_equal(dim(pd), c(200L, 3L))
  expect_equal(colnames(pd), c("a", "b", "c"))
  expect_no_error(geweke_test(fit))
})

test_that("ranef does not emit a field/hyperparameter tail as random effects (#205)", {
  fit <- structure(list(
    re_layout = list(list(group_var = "g", levels = c("1", "2"),
                          coef_labels = "(Intercept)")),
    n_fixed = 1L,
    draws = matrix(rnorm(50 * 5), nrow = 50,
                   dimnames = list(NULL, c("beta[1]", "field[1]", "field[2]",
                                           "field[3]", "log_tau")))),
    class = "tulpa_fit")
  # No `re[`-named columns and the tail width (4) != the RE layout (2 levels):
  # this is the field / hyperparameter tail, not identifiable random effects.
  expect_equal(nrow(ranef(fit)), 0L)
})
