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
