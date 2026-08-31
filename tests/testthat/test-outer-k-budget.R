# The outer k-hat's dependence on the draw budget (gcol33/tulpa#631).
#
# `control$k_samples` is documented as the estimate's PRECISION knob. It is not,
# under the automatic PSIS tail rule: `.psis_tail_len(S) = min(S/5, 3 sqrt(S))`,
# so the fitted tail FRACTION is `3 / sqrt(S)` and SHRINKS as the budget grows.
# A larger budget therefore characterises a DEEPER quantile of the weight
# distribution, and on a ratio whose log grows super-linearly in the whitened
# radius the deeper quantile reads heavier.
#
# The estimator is not what moves: `tulpa_psis` reproduces `loo::psis` to 1e-13
# at every budget in the table below (checked in dev_notes/issue631). What moves
# is which part of the tail is being described.
#
# These pin the two halves the `k_quality` escalation's precision rung depends
# on: the automatic rule moves the number, a held fraction does not.

# A Student-t target against a Gaussian proposal matched to its exact moments:
# the importance ratio is unbounded, and its log grows faster than linearly in
# the whitened radius.
.kb_lr <- function(n, seed, s = 0.4, df = 8) {
    lg <- function(u) stats::dt(u / s, df, log = TRUE)
    u  <- seq(-80 * s, 80 * s, length.out = 200001L)
    l  <- lg(u); l <- l - max(l); w <- exp(l); w <- w / sum(w)
    mu <- sum(w * u); sd <- sqrt(sum(w * (u - mu)^2))
    set.seed(seed)
    z <- stats::rnorm(n)
    lg(mu + sd * z) + 0.5 * z^2
}

.kb_median <- function(n, tail_points = NULL, seeds = 1:6) {
    stats::median(vapply(seeds, function(sd_i)
        tulpa:::tulpa_psis(.kb_lr(n, sd_i), tail_points = tail_points)$pareto_k,
        numeric(1)))
}

test_that("the automatic PSIS tail rule fits a shrinking fraction of the draws", {
    fr <- vapply(c(500L, 2000L, 10000L, 50000L),
                 function(S) tulpa:::.psis_tail_len(S) / S, numeric(1))
    expect_true(all(diff(fr) < 0))                       # strictly shrinking
    expect_lt(fr[4L], fr[1L] / 5)                        # and by a lot
    expect_true(all(fr <= 0.2 + 1e-12))                  # never past the cap
})

test_that("under that rule the outer k-hat moves with the draw budget", {
    skip_on_cran()
    k_500   <- .kb_median(500L)
    k_10000 <- .kb_median(10000L)
    # Same target, same proposal, same family of draws -- only the budget moves,
    # and the reported shape crosses from the ok band to far past unreliable.
    expect_lt(k_500, 1)
    expect_gt(k_10000, 2)
    expect_gt(k_10000, 3 * k_500)
})

test_that("holding the tail fraction makes the budget a precision knob", {
    skip_on_cran()
    # The fraction `k_samples = 500` already implies, held as the budget grows.
    frac <- tulpa:::.psis_tail_len(500L) / 500
    tp   <- function(n) max(tulpa:::.PSIS_MIN_EVAL, as.integer(floor(frac * n)))
    ks <- vapply(c(500L, 2000L, 10000L, 50000L),
                 function(n) .kb_median(n, tail_points = tp(n)), numeric(1))
    # The number stops moving: every budget from 2000 up agrees to within 0.05,
    # against a range of more than 7 under the automatic rule.
    expect_lt(max(ks[-1L]) - min(ks[-1L]), 0.05)
    expect_lt(abs(ks[1L] - stats::median(ks)), 0.2)
    # And the spread across seeds NARROWS, which is what a precision knob does.
    spread <- function(n, t) {
        v <- vapply(1:6, function(sd_i)
            tulpa:::tulpa_psis(.kb_lr(n, sd_i), tail_points = t)$pareto_k,
            numeric(1))
        diff(range(v))
    }
    expect_lt(spread(50000L, tp(50000L)), spread(500L, tp(500L)))
})

# --- the resolved rule the outer paths actually use --------------------------
#
# The two blocks above establish the PROPERTY (the automatic rule moves the
# number, a held fraction does not). These pin the RULE the four outer backends
# now resolve their tail size by (gcol33/tulpa#631, unblocked by #632's single
# default): `.k_outer_tail_points()`.

test_that("the default budget is the published rule, exactly", {
    # The whole safety of the change: at the shipped budget the helper returns
    # NULL, so `tulpa_psis()` takes its own default and a default fit is
    # unchanged to the bit. The explicit-request path -- and its 20% cap -- is
    # not entered at all.
    ref <- tulpa:::.nl_diag("k_samples")
    expect_null(tulpa:::.k_outer_tail_points(ref))
    expect_null(tulpa:::.k_outer_tail_points(as.numeric(ref)))

    # An explicit request is honoured and the helper is idempotent, which is
    # what lets `.k_dispatch_report()` and `.k_dispatch()` both resolve.
    expect_identical(tulpa:::.k_outer_tail_points(ref, 42L), 42L)
    expect_identical(tulpa:::.k_outer_tail_points(4000L, 42L), 42L)
    expect_identical(
        tulpa:::.k_outer_tail_points(4000L, tulpa:::.k_outer_tail_points(4000L)),
        tulpa:::.k_outer_tail_points(4000L))
})

test_that("away from the default the tail FRACTION is held, not the rule", {
    ref  <- tulpa:::.nl_diag("k_samples")
    frac <- tulpa:::.psis_tail_len(ref) / ref

    # Held: the realized fraction tracks the reference wherever the budget is
    # large enough that neither the floor nor the 20% cap binds.
    for (n in c(1000L, 2000L, 10000L, 50000L)) {
        tp <- tulpa:::.k_outer_tail_points(n)
        expect_equal(tp / n, frac, tolerance = 1e-3, info = n)
        # and it stays under the defensive cap, so the cap warning never fires
        expect_lte(tp, floor(0.2 * n))
    }

    # Not the published rule: that fraction shrinks as 3/sqrt(S).
    expect_lt(tulpa:::.psis_tail_len(50000L) / 50000, frac / 2)
})

test_that("the held fraction is what makes the budget a precision knob", {
    # The measurement of gcol33/tulpa#631, re-read through the shipped helper
    # rather than through a hand-passed tail size: the same heavy-tailed target,
    # the same draws, scored under the automatic rule and under the resolved
    # one. The automatic rule crosses bands; the resolved rule does not.
    ns <- c(500L, 2000L, 10000L, 50000L)
    auto <- vapply(ns, function(n) .kb_median(n), numeric(1))
    held <- vapply(ns, function(n)
        .kb_median(n, tail_points = tulpa:::.k_outer_tail_points(n)), numeric(1))

    # The automatic rule climbs monotonically and leaves every band.
    expect_true(all(diff(auto) > 0))
    expect_lt(auto[1], tulpa:::.nl_diag("k_usable"))
    expect_gt(auto[length(auto)], 4)

    # The resolved rule holds the estimand: everything within a band's width of
    # the value read at the default budget, and no band boundary crossed.
    expect_true(all(abs(held - held[1]) < 0.2),
                info = paste(round(held, 3), collapse = " "))
    expect_identical(length(unique(held > tulpa:::.nl_diag("k_usable"))), 1L)
})

test_that("every outer backend resolves the same tail size", {
    # gcol33/tulpa#630 made `.k_dispatch()` the one candidate loop behind all
    # four backends, so resolving there is what gives all four the same rule.
    # Scored on one spec at two budgets: the k-hat moves less than a band.
    set.seed(11)
    lt <- function(U) stats::dt(U[, 1L] / 0.4, df = 8, log = TRUE)
    spec <- tulpa:::.k_cand_spec(lt = lt, u_hat = 0, Su = matrix(0.4^2, 1, 1),
                                 proposal_source = "mode_hessian")
    k <- vapply(c(500L, 5000L), function(n) {
        set.seed(4); tulpa:::.k_dispatch(spec, n)$best$pareto_k
    }, numeric(1))
    expect_true(all(is.finite(k)))
    expect_lt(abs(k[2] - k[1]), 0.35)
})

test_that("the held fraction is a floor, so no budget gets a noisier fit", {
    # Taking the held fraction as a REPLACEMENT for the published rule buys a
    # stable estimand by making cheap diagnostics noisier: below the reference
    # budget the published rule is in its `S/5` regime and is the more generous
    # of the two (40 tail points at 200 draws against the fraction's 27). It is
    # therefore a floor. Measured cost of getting this wrong: a per-arm k-hat
    # crossing the reported band on the 200-draw fixture in
    # test-joint-pareto-k-proposal.R.
    for (n in c(50L, 100L, 200L, 400L, 500L, 600L, 2000L, 20000L)) {
        tp <- tulpa:::.k_outer_tail_points(n)
        used <- if (is.null(tp)) tulpa:::.psis_tail_len(n) else tp
        expect_true(used >= tulpa:::.psis_tail_len(n), info = n)
    }

    # And the fraction is confined rather than shrinking without bound: the
    # published rule runs 20% -> 1.3% over this range, the resolved one does not
    # leave [13%, 20%].
    fr <- vapply(c(100L, 500L, 2000L, 10000L, 50000L), function(n) {
        tp <- tulpa:::.k_outer_tail_points(n)
        (if (is.null(tp)) tulpa:::.psis_tail_len(n) else tp) / n
    }, numeric(1))
    expect_true(all(fr >= 0.13 & fr <= 0.20), info = paste(round(fr, 4), collapse = " "))
    expect_lt(tulpa:::.psis_tail_len(50000L) / 50000, 0.02)   # what it replaces
})
