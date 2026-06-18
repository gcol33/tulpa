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
