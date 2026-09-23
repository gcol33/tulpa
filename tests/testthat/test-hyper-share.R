# How much of the fixed-effect marginal the hyperparameter integration
# contributes (gcol33/tulpa#862).
#
# The nested marginal is `within + between` under the law of total variance
# over the outer grid, and `between` is the part that exists BECAUSE the
# hyperparameter was integrated rather than held at a point. A fit whose
# marginal barely moves when its hyperparameter posterior does looked
# identical to one that integrated properly, because nothing reported the
# split. Measured on an occu_cover fixture the share was 3.4% on the affected
# coefficient while varying the hyperprior moved the field scale by 80% and the
# reported marginal by under 6%.
#
# Reported as a number with no threshold: what counts as too little depends on
# the model.

test_that("the hyperparameter share is reported and matches the grid pieces", {
    skip_on_cran()
    sim <- .sparse_icar_arm()
    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm),
                                      prior = .sparse_icar_pinned_prior(sim),
                                      control = list(diagnose_k = FALSE))

    d  <- diagnostics(fit)
    sh <- attr(d, "hyper_share")
    expect_false(is.null(sh))
    expect_true(all(is.finite(sh)))
    expect_true(all(sh >= 0 & sh <= 1))

    # Recomputed from the fit's own retained pieces, which is what the reported
    # number has to equal: between / (within + between), per coefficient.
    mom <- .nested_fixed_moments(fit)
    expect_false(is.null(mom))
    w       <- mom$w
    within  <- colSums(mom$var * w)
    mu_bar  <- colSums(mom$mu * w)
    between <- colSums(sweep(mom$mu, 2L, mu_bar, "-")^2 * w)
    expect_equal(as.numeric(sh), as.numeric(between / (within + between)),
                 tolerance = 1e-10)

    expect_equal(attr(d, "hyper_share_min"), min(sh), tolerance = 1e-12)
    expect_equal(attr(d, "hyper_share_max"), max(sh), tolerance = 1e-12)
})

test_that("the share reaches the printed diagnostics and the summary row", {
    skip_on_cran()
    sim <- .sparse_icar_arm()
    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm),
                                      prior = .sparse_icar_pinned_prior(sim),
                                      control = list(diagnose_k = FALSE))
    d <- diagnostics(fit)

    out <- paste(utils::capture.output(print(d)), collapse = "\n")
    expect_match(out, "hyperparameter integration contributes")

    srow <- attr(d, "summary")
    expect_true(is.finite(srow$hyper_share_min))
    expect_true(is.finite(srow$hyper_share_max))
    expect_equal(srow$hyper_share_min, attr(d, "hyper_share_min"),
                 tolerance = 1e-12)
})

test_that("a fit with no retained grid pieces reports no share rather than a wrong one", {
    skip_on_cran()
    sim <- .sparse_icar_arm()
    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm),
                                      prior = .sparse_icar_pinned_prior(sim),
                                      control = list(diagnose_k = FALSE,
                                                     keep_grid_hessians = FALSE))
    # Without the per-cell pieces the decomposition is unavailable. It has to be
    # absent, not zero -- zero would read as "the hyperparameter contributed
    # nothing", which is a claim about the fit rather than about the record.
    expect_null(.tulpa_hyper_share(fit))
    expect_null(attr(diagnostics(fit), "hyper_share"))
})

test_that("diagnostic_summary() carries the share too, and agrees with diagnostics()", {
    skip_on_cran()
    sim <- .sparse_icar_arm()
    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm),
                                      prior = .sparse_icar_pinned_prior(sim),
                                      control = list(diagnose_k = FALSE))

    # The share reached `attr(diagnostics(fit), "summary")` from 0.6.0 and not
    # `diagnostic_summary()`, so the one number that quantifies this issue's
    # symptom was reachable from one reporting door and not the other. Both
    # read `.tulpa_hyper_share()`, so the assertion is that they cannot drift.
    srow <- attr(diagnostics(fit), "summary")
    ds   <- diagnostic_summary(fit, quiet = TRUE)

    expect_true(is.finite(ds$hyper_share_min))
    expect_true(is.finite(ds$hyper_share_max))
    expect_equal(ds$hyper_share_min, srow$hyper_share_min)
    expect_equal(ds$hyper_share_max, srow$hyper_share_max)
    expect_lte(ds$hyper_share_min, ds$hyper_share_max)
    expect_gte(ds$hyper_share_min, 0)
    expect_lte(ds$hyper_share_max, 1)

    # Stated, never scored: a small share is not by itself a bad interval, so
    # it must not move the verdict.
    expect_true(any(grepl("Hyperparameter integration contributes",
                          ds$recommendations, fixed = TRUE)))

    out <- paste(utils::capture.output(print(ds)), collapse = "\n")
    expect_match(out, "Hyperparameter share of the fixed-effect marginal")
})
