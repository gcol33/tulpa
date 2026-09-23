# Is the nested-Laplace FIXED-EFFECT interval calibrated when a spatial field
# is in the model? (gcol33/tulpa#862)
#
# The gap. tulpa gates the fixed-effect interval's coverage only on an IID
# region-grouped RE block (`test-nested-laplace-recovery.R`). The spatial
# recovery tests gate the HYPERPARAMETERS -- tau, rho, sigma2, range -- and
# never beta; the joint-multi sweep gates (sigma, alpha) and never beta. So the
# one combination a spatial consumer actually reports from -- a fixed-effect
# interval with a field in the model -- had no gate, and gcol33/tulpa#862 ran
# eight rounds of nested_laplace-versus-NUTS comparison with no reference to
# consult. Two routes disagreeing says nothing about which is wrong; coverage
# against a simulated truth does, and it is what settles the question.
#
# Measured at tulpa 0.6.0 by `dev_notes/sbc_cover/spatial_beta_coverage.R`,
# 40 seeds per regime, on the same fixture this gate runs:
#
#   strong  10x10 lattice, 10 reps, amp 0.8   beta1  97.5%   beta2 100.0%
#   weak     8x8  lattice,  3 reps, amp 1.6   beta1 100.0%   beta2  95.0%
#   pooled                                    0.981 [0.946, 0.996], n = 160
#
# The `weak` regime is gcol33/tulpa#862's own geometry -- 64 sites, three
# replicates, a field larger than the fixed effects -- and it is calibrated
# there, marginally conservative rather than narrow.
#
# Gate shape follows `test-nested-laplace-recovery.R`: a POOLED rate plus a
# loose per-cell floor. A tight per-cell gate would be mis-designed, because a
# correctly calibrated method fails one of several per-cell floors a good
# fraction of the time; pooling the trials is what makes the threshold stable.
# There is deliberately no upper gate -- an interval that is somewhat
# conservative in a weakly identified regime is not a defect, and inventing a
# ceiling here would fail a fit that is behaving.

skip_if_not_slow()

N_SEED_COV <- 40L

test_that("the fixed-effect interval is calibrated with a spatial field present", {
    skip_on_cran()

    seeds  <- seq_len(N_SEED_COV)
    pooled <- integer(0)
    per_cell <- c()

    for (rn in names(.SPATIAL_BETA_REGIMES)) {
        sw <- .spatial_beta_sweep(rn, seeds)

        # Every cell has to have produced a fit; a regime that silently failed
        # to converge would otherwise pool as an empty vector and pass.
        expect_true(all(sw$n >= 0.9 * N_SEED_COV),
                    info = paste("too many failed fits in regime", rn))

        pooled   <- c(pooled, as.vector(sw$cov[!is.na(sw$cov)]))
        per_cell <- c(per_cell, stats::setNames(
            sw$rate, paste0(rn, "_beta", seq_along(sw$rate))))
    }

    # Per-cell floor: loose, for the reason in the header.
    for (nm in names(per_cell)) {
        expect_gte(per_cell[[nm]], 0.80)
    }

    # The pooled rate is the gate. Measured 0.981 over 160 trials.
    expect_gte(mean(pooled), 0.90)
    expect_gte(length(pooled), 2L * 2L * 0.9 * N_SEED_COV)
})

test_that("the weakly identified regime is where the interval is judged", {
    skip_on_cran()

    # gcol33/tulpa#862 is about a 64-site, 3-replicate fixture specifically, so
    # that regime carries its own assertion rather than only contributing to a
    # pool a well-identified regime could carry.
    sw <- .spatial_beta_sweep("weak", seq_len(N_SEED_COV))
    cov <- as.vector(sw$cov[!is.na(sw$cov)])

    expect_gte(mean(cov), 0.85)

    # And the interval is not calibrated by being unbounded: a 95% interval on
    # these coefficients runs to roughly 1.1 / 0.7 at this fixture, so a gate
    # that only checked coverage would pass an infinitely wide one.
    expect_lt(mean(sw$width[, 1L], na.rm = TRUE), 3)
    expect_lt(mean(sw$width[, 2L], na.rm = TRUE), 3)
})
