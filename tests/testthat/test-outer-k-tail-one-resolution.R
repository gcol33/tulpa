# The GPD tail size is resolved ONCE per fit, and every backend records it.
#
# gcol33/tulpa#631 made the outer k-hat's estimand independent of its own draw
# budget by holding the fitted tail to a fixed FRACTION, resolved inside
# `.k_dispatch()` so all four backends inherit it. Two things then reopened it:
#
# gcol33/tulpa#690 -- the joint path chose its proposal through `.k_dispatch()`
# (held fraction) and then RE-FITTED the reported shape through
# `.joint_pareto_uncertainty()` at the raw request, which is the published rule
# whenever the caller named nothing. The reported k was a different quantile of
# the weight distribution from the one the choice was made on, and it moved with
# `control$k_samples` again.
# gcol33/tulpa#691 -- the per-arm pass had the same shape, so `by_arm_k` was not
# comparable with the `pareto_k` printed beside it.
# gcol33/tulpa#692 -- three of the four backends recorded no tail size and
# raised no cap warning, so a raised or capped tail was invisible there.

test_that("the budget-stable resolution is idempotent and floors at the rule", {
  ref <- tulpa:::.nl_diag("k_samples")

  # At the reference budget the helper returns NULL, so a default fit is fitted
  # under the published rule exactly as before -- that is what makes the whole
  # change a no-op on a default fit.
  expect_null(tulpa:::.k_outer_tail_points(ref, NULL))

  # An explicit request is honoured, which is what makes resolving twice safe:
  # `.k_dispatch()` re-resolves the value the caller already resolved.
  expect_identical(tulpa:::.k_outer_tail_points(ref, 42L), 42L)
  r1 <- tulpa:::.k_outer_tail_points(4L * ref, NULL)
  expect_identical(tulpa:::.k_outer_tail_points(4L * ref, r1), r1)

  # Above the reference budget the held fraction is more tail than the
  # published rule, so it is used; below it the published rule is the more
  # generous of the two and is kept.
  expect_true(r1 > tulpa:::.psis_tail_len(4L * ref))
  expect_null(tulpa:::.k_outer_tail_points(ref %/% 4L, NULL))
})

test_that("the 20% cap warns once, from a shared helper", {
  # .tulpa_psis_k_uncertainty() applies the cap silently by design so the
  # bootstrap re-fits do not each warn; something has to say it once per fit,
  # and only the joint driver did.
  expect_warning(tulpa:::.k_tail_cap_warn(300L, 500L), "20% PSIS tail cap")
  expect_silent(tulpa:::.k_tail_cap_warn(80L, 500L))
  expect_silent(tulpa:::.k_tail_cap_warn(NULL, 500L))
})

test_that("every outer-k backend reports the tail it fitted on", {
  skip_on_cran()
  # A one-axis synthetic target, driven through the shared report path the
  # single-block, SPDE and RE-covariance backends all use.
  lt <- function(U) -0.5 * rowSums(U^2) / 4
  spec <- tulpa:::.k_cand_spec(lt = lt, u_hat = 0, Su = matrix(1, 1, 1),
                               proposal_source = "mode_hessian")
  out <- tulpa:::.k_dispatch_report(spec, tulpa:::.nl_diag("k_samples"))
  expect_true(is.finite(out$pareto_k))
  expect_true(is.finite(out$tail_points))
  expect_identical(out$tail_points,
                   as.integer(tulpa:::.psis_tail_len(out$n_eval)))

  # An explicit request is what it reports, not the published rule.
  out2 <- tulpa:::.k_dispatch_report(spec, tulpa:::.nl_diag("k_samples"),
                                     tail_points = 50L)
  expect_identical(out2$tail_points, 50L)
})

test_that("the single-block grid path passes its tail_points on", {
  skip_on_cran()
  # It accepted the argument and dropped it, so an explicit request never
  # reached the scorer at all on that path.
  fml <- names(formals(tulpa:::.nested_grid_pareto_k))
  expect_true("tail_points" %in% fml)
  body_txt <- paste(deparse(body(tulpa:::.nested_grid_pareto_k)),
                    collapse = " ")
  expect_match(body_txt, "tail_points = tail_points")
})

test_that("the joint pass reports the k it chose its proposal on", {
  skip_on_cran()
  # The joint driver resolves once at the top and hands the resolved value to
  # `.k_dispatch()` and to every re-fit, so the reported k and the chosen
  # proposal are read at one quantile.
  src <- paste(deparse(tulpa:::.joint_pareto_k), collapse = "\n")
  expect_match(src, "k_tail_points <- .k_outer_tail_points", fixed = TRUE)
  # and nothing between the resolution and the scorers re-reads the raw request
  after <- sub(".*k_tail_points <- .k_outer_tail_points\\(n_samples, k_tail_points\\)",
               "", src)
  expect_false(grepl("tail_points = NULL", after, fixed = TRUE))
})
