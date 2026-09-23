# A flagged inner layer names its own remedy (gcol33/tulpa#862).
#
# The engine ships two corrections for a misfit inner Gaussian --
# `control$subspace_debias` (exact Metropolis on the flagged coordinates) and
# `control$cila` (reweighting the whole inner Gaussian by the exact joint) --
# and both are off by default. A fit whose inner layer banded `unreliable`
# therefore reported the uncorrected marginal while saying only that it was
# unreliable, never that a correction existed. The band is a diagnostic; the
# band plus the remedy is an instruction.
#
# The sparse ICAR fixture bands BOTH inner scores `unreliable` (gamma_3 ~ 2.5,
# inner importance k-hat ~ 0.73), so it exercises the note without needing the
# occu_cover route.

test_that("a flagged inner layer names the corrections it did not run", {
    skip_on_cran()
    sim <- .sparse_icar_arm()
    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm),
                                      prior = .sparse_icar_pinned_prior(sim),
                                      control = list(diagnose_k = FALSE))

    d <- diagnostics(fit)
    expect_identical(attr(d, "inner_skew_band"), "unreliable")

    note <- attr(d, "inner_debias_note")
    expect_true(is.character(note) && nzchar(note))
    expect_match(note, "subspace_debias", fixed = TRUE)
    expect_match(note, "cila", fixed = TRUE)
    # It has to say the corrections did not run, or a reader takes the reported
    # marginal for a corrected one.
    expect_match(note, "off by default", fixed = TRUE)

    out <- paste(utils::capture.output(print(d)), collapse = "\n")
    expect_match(out, "subspace_debias", fixed = TRUE)
})

test_that("diagnostic_summary() carries the remedy after the bands, not before", {
    skip_on_cran()
    sim <- .sparse_icar_arm()
    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm),
                                      prior = .sparse_icar_pinned_prior(sim),
                                      control = list(diagnose_k = FALSE))

    rec <- diagnostic_summary(fit)$recommendations
    i_band <- grep("gamma_3", rec, fixed = TRUE)
    i_note <- grep("subspace_debias", rec, fixed = TRUE)

    expect_length(i_note, 1L)
    expect_true(length(i_band) >= 1L)
    # The remedy reads as a remedy only after the band that motivates it.
    expect_gt(i_note, max(i_band))
})

test_that("a fit that ran a correction is not told to run one", {
    skip_on_cran()
    sim <- .sparse_icar_arm()
    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm),
                                      prior = .sparse_icar_pinned_prior(sim),
                                      control = list(diagnose_k = FALSE,
                                                     subspace_debias = TRUE))

    expect_false(is.null(fit$subspace_debias))
    expect_null(attr(diagnostics(fit), "inner_debias_note"))
    expect_false(any(grepl("subspace_debias",
                           diagnostic_summary(fit)$recommendations,
                           fixed = TRUE)))
})
