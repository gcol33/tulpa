# What the reporting doors say when the outer grid collapses onto one cell
# (gcol33/tulpa#863).
#
# Every layer of this already existed and was unit-tested in isolation:
# `.tulpa_grid_reliability()` computes `ess_grid` / `max_weight` from the stored
# weights unconditionally, `.joint_attach_pareto_k_regime()` attaches the regime
# before the `diagnose_k` branch, and `.tulpa_outer_regime_note()` has its own
# tests. Nothing asserted that a fit whose grid actually collapsed carries the
# verdict out through `diagnostics()`, its `print()` and `diagnostic_summary()`.
#
# That gap is what an occu_cover batch surfaced: a 25 km run read
# `ess_grid = 1.0000067` of 27 cells at `max_weight = 0.99999` and its consumer
# reported nothing, having re-derived an `ess_grid >= 2 && max_weight <= 0.9`
# rule of its own instead of reading the engine's verdict. The engine's answer
# has to be reachable from the doors a caller uses, or every consumer writes
# that rule again and they drift.
#
# The fits run at `diagnose_k = FALSE`, which is how the batch above was fitted
# (its reliability rows carry an empty `pareto_k`). The regime is read off
# stored weights by `.joint_attach_pareto_k_regime()` BEFORE the `diagnose_k`
# branch, so a collapse has to be reported whether or not the k-hat ran -- that
# independence is the property these tests pin.

test_that("a collapsed outer grid reports its regime through diagnostics()", {
    skip_on_cran()
    sim <- .sparse_icar_arm()
    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm),
                                      prior = .sparse_icar_pinned_prior(sim),
                                      control = list(diagnose_k = FALSE))

    # The fit collapsed, and the k-hat diagnostic never ran.
    expect_identical(fit$pareto_k_regime, "collapsed_edge")
    expect_true(is.na(fit$pareto_k))
    expect_identical(fit$pareto_k_declined, "not_requested")

    d <- diagnostics(fit)

    # The quadrature numbers are computed whether or not k-hat ran.
    expect_true(is.finite(attr(d, "ess_grid")))
    expect_true(is.finite(attr(d, "max_weight")))
    expect_gt(attr(d, "max_weight"), 0.5)
    expect_lt(attr(d, "ess_grid"), attr(d, "n_grid"))

    # The verdict travels, and it names the axis to widen.
    expect_identical(attr(d, "outer_regime"), "collapsed_edge")
    note <- attr(d, "outer_regime_note")
    expect_true(is.character(note) && nzchar(note))
    expect_match(note, "collapsed against a boundary node")
    expect_match(note, "sigma")
})

test_that("printing the diagnostics of a collapsed fit says so", {
    skip_on_cran()
    sim <- .sparse_icar_arm()
    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm),
                                      prior = .sparse_icar_pinned_prior(sim),
                                      control = list(diagnose_k = FALSE))

    out <- paste(utils::capture.output(print(diagnostics(fit))), collapse = "\n")

    expect_match(out, "outer grid quadrature ESS")
    expect_match(out, "collapsed against a boundary node")
})

test_that("diagnostic_summary() warns on a collapsed grid and says which axis", {
    skip_on_cran()
    sim <- .sparse_icar_arm()
    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm),
                                      prior = .sparse_icar_pinned_prior(sim),
                                      control = list(diagnose_k = FALSE))

    s <- diagnostic_summary(fit)

    expect_identical(s$outer_regime, "collapsed_edge")
    expect_identical(s$status, "WARN")
    expect_true(any(grepl("collapsed against a boundary node", s$recommendations,
                          fixed = TRUE)))
})

test_that("the approximation summary row carries the regime's reading, not only its code", {
    skip_on_cran()
    sim <- .sparse_icar_arm()
    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm),
                                      prior = .sparse_icar_pinned_prior(sim),
                                      control = list(diagnose_k = FALSE))

    # `attr(diagnostics(fit), "summary")` is the one-row frame a consumer
    # writes to a reliability CSV. It carried `outer_regime` without its
    # reading, so the axis to widen was reachable only from the printed form.
    srow <- attr(diagnostics(fit), "summary")

    expect_identical(srow$outer_regime, "collapsed_edge")
    expect_true(is.character(srow$outer_regime_note))
    expect_match(srow$outer_regime_note, "collapsed against a boundary node")
    expect_match(srow$outer_regime_note, "sigma")
    expect_true(is.finite(srow$ess_grid))
    expect_true(is.finite(srow$max_weight))
})

test_that("a spread grid reports no collapse", {
    skip_on_cran()
    sim <- .sparse_icar_arm()
    # The same data with the axis left to the engine: the recenter resolves the
    # rail, so the grid spreads and there is no collapse to report.
    prior <- .sparse_icar_pinned_prior(sim)
    prior$sigma_grid <- NULL
    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm),
                                      prior = prior,
                                      control = list(diagnose_k = FALSE))

    expect_identical(fit$pareto_k_regime, "spread")

    d <- diagnostics(fit)
    expect_identical(attr(d, "outer_regime"), "spread")
    out <- paste(utils::capture.output(print(d)), collapse = "\n")
    expect_false(grepl("collapsed against a boundary node", out, fixed = TRUE))
    expect_false(grepl("collapsed onto an interior mode", out, fixed = TRUE))
})
