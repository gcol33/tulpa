# Finite-difference check of the interval-censored Gaussian kernel
# (ordered-probit with KNOWN thresholds; gcol33/tulpaObs ordinal cover). The
# latent value is Normal(eta, sigma^2) and the observation records only that it
# fell in (lower, upper]; the log-density is the class probability MASS
#   log P = log( Phi((upper - eta)/sigma) - Phi((lower - eta)/sigma) ).
# cpp_interval_gaussian_terms() returns the analytic (ll, d logP/d eta,
# -d2 logP/d eta2); each is checked against central differences of an
# independent R evaluation, with the open outer classes (+/-Inf bounds).

# Reference log-mass from R's pnorm, tail-accurate, matching the kernel.
.ig_ll_R <- function(lower, upper, eta, sigma) {
    zl <- (lower - eta) / sigma
    zu <- (upper - eta) / sigma
    P <- if (zu <= 0) {
        stats::pnorm(zu) - stats::pnorm(zl)
    } else if (zl >= 0) {
        stats::pnorm(zl, lower.tail = FALSE) - stats::pnorm(zu, lower.tail = FALSE)
    } else {
        stats::pnorm(zu) - stats::pnorm(zl)
    }
    log(max(P, 1e-300))
}

test_that("analytic grad / neg_hess match central differences (finite interval)", {
    h <- 1e-5
    grid <- expand.grid(
        eta   = c(-2.0, -0.5, 0.3, 1.4),
        sigma = c(0.4, 0.85, 1.6),
        band  = c("low", "mid", "high")
    )
    bounds <- list(low = c(-1.5, -0.7), mid = c(-0.2, 0.6), high = c(0.9, 2.1))
    for (r in seq_len(nrow(grid))) {
        b   <- bounds[[as.character(grid$band[r])]]
        eta <- grid$eta[r]; sg <- grid$sigma[r]
        out <- tulpa:::cpp_interval_gaussian_terms(b[1], b[2], eta, sg)

        expect_equal(out[["ll"]], .ig_ll_R(b[1], b[2], eta, sg), tolerance = 1e-10)

        g_fd <- (.ig_ll_R(b[1], b[2], eta + h, sg) -
                 .ig_ll_R(b[1], b[2], eta - h, sg)) / (2 * h)
        expect_equal(out[["grad"]], g_fd, tolerance = 1e-5)

        nh_fd <- -(.ig_ll_R(b[1], b[2], eta + h, sg) -
                   2 * .ig_ll_R(b[1], b[2], eta, sg) +
                   .ig_ll_R(b[1], b[2], eta - h, sg)) / h^2
        expect_equal(out[["neg_hess"]], nh_fd, tolerance = 1e-3)
        expect_gte(out[["neg_hess"]], 0)   # log-concave in eta
    }
})

test_that("open outer classes (+/-Inf bounds) match a one-sided tail", {
    h <- 1e-5
    for (eta in c(-1.0, 0.0, 1.2)) for (sg in c(0.5, 1.3)) {
        # Lowest class: (-Inf, b] -> P = Phi((b - eta)/sigma).
        b   <- 0.4
        lo  <- tulpa:::cpp_interval_gaussian_terms(-Inf, b, eta, sg)
        expect_equal(lo[["ll"]], stats::pnorm((b - eta) / sg, log.p = TRUE),
                     tolerance = 1e-10)
        g_fd <- (stats::pnorm((b - eta - h) / sg, log.p = TRUE) -
                 stats::pnorm((b - eta + h) / sg, log.p = TRUE)) / (2 * h)
        expect_equal(lo[["grad"]], g_fd, tolerance = 1e-5)
        expect_gte(lo[["neg_hess"]], 0)

        # Highest class: (a, +Inf) -> P = 1 - Phi((a - eta)/sigma).
        a  <- -0.3
        hi <- tulpa:::cpp_interval_gaussian_terms(a, Inf, eta, sg)
        expect_equal(hi[["ll"]],
                     stats::pnorm((a - eta) / sg, lower.tail = FALSE, log.p = TRUE),
                     tolerance = 1e-10)
        expect_gte(hi[["neg_hess"]], 0)
    }

    # Fully open (-Inf, +Inf) is the whole line: P = 1, flat in eta.
    full <- tulpa:::cpp_interval_gaussian_terms(-Inf, Inf, 0.7, 0.9)
    expect_equal(full[["ll"]], 0)
    expect_equal(full[["grad"]], 0)
})

test_that("a finite bound far from eta stays finite (no Inf * 0 = NaN)", {
    # zl/zu overflow while the density underflows: the z * phi(z) product must
    # take its analytic 0 limit rather than NaN.
    out <- tulpa:::cpp_interval_gaussian_terms(8.0, 9.0, 0.0, 0.3)
    expect_true(is.finite(out[["ll"]]))
    expect_true(is.finite(out[["grad"]]))
    expect_true(is.finite(out[["neg_hess"]]) && out[["neg_hess"]] >= 0)
})

test_that("an interval far below eta keeps a gradient pointing back to it", {
    # gcol33/tulpa#462: the probability was differenced on the natural scale and
    # floored at 1e-300, so once both pnorm tails underflowed the kernel
    # returned a finite ll, an exactly zero gradient and a floored positive
    # curvature -- a plateau of the floor. The line search accepts a finite
    # objective and newton_converged reads max|grad| == 0 as a mode, so a solve
    # that wandered out here reported convergence at a point that is not one.
    sg  <- 0.4
    eta <- 0.0
    for (mult in c(20, 38, 50, 120, 400)) {
        lo <- eta + mult * sg
        hi <- lo + sg
        out <- tulpa:::cpp_interval_gaussian_terms(lo, hi, eta, sg)

        expect_true(is.finite(out[["ll"]]), info = paste("mult", mult))
        expect_lt(out[["ll"]], -100)
        # The interval sits ABOVE eta, so increasing eta increases the mass:
        # the gradient is strictly positive and of order (lo - eta) / sigma^2.
        expect_gt(out[["grad"]], 0)
        expect_gt(abs(out[["grad"]]), 1)
        expect_gt(out[["neg_hess"]], 0)
    }

    # Mirror image below eta: the gradient turns around.
    below <- tulpa:::cpp_interval_gaussian_terms(-50 * sg - sg, -50 * sg, 0, sg)
    expect_lt(below[["grad"]], 0)
    expect_true(is.finite(below[["ll"]]))
})

test_that("the deep-tail gradient matches the one-sided Mills ratio", {
    # Far outside a narrow interval the mass is dominated by the near edge, so
    # d logP / d eta -> (edge - eta) / sigma^2 to leading order. That is an
    # independent reference for a regime where a finite difference of the R
    # log-mass cannot be formed (both pnorm values underflow to 0).
    sg <- 0.5; eta <- 0.0
    for (mult in c(45, 60, 100)) {
        lo <- mult * sg
        out <- tulpa:::cpp_interval_gaussian_terms(lo, lo + 1e-3 * sg, eta, sg)
        expect_equal(out[["grad"]], (lo - eta) / sg^2, tolerance = 1e-2,
                     info = paste("mult", mult))
        # log P ~ log(width) + log phi(z) for a narrow interval.
        z <- (lo - eta) / sg
        expect_equal(out[["ll"]],
                     log(1e-3 * sg) - 0.5 * z^2 - 0.5 * log(2 * pi) - log(sg),
                     tolerance = 1e-2)
    }
})

test_that("an empty interval is refused rather than floored", {
    # upper <= lower carries no mass: -Inf is what the line search backtracks
    # off, where a floored finite value would be accepted as a trial point.
    out <- tulpa:::cpp_interval_gaussian_terms(1.0, 1.0, 0.0, 1.0)
    expect_equal(out[["ll"]], -Inf)
    expect_equal(out[["grad"]], 0)
})
