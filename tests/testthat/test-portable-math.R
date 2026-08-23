# The family log-likelihood and curvature ladders run inside OpenMP reductions,
# and Rmath's lgammafn / digamma / trigamma / psigamma / pnorm route their
# domain and range paths through R's warning() / error(), which touch R's
# global error state and can longjmp -- out of a structured block, which is
# std::terminate rather than an R error. tulpa/portable_math.h holds
# replacements that carry no R state (gcol33/tulpa#461); this file is the
# arbiter that they compute the same numbers as the routines they replaced,
# over the argument ranges the families reach.

.pm <- function(fn, x, y = numeric(0)) {
    tulpa:::cpp_portable_math(fn, as.numeric(x), as.numeric(y))
}

# Relative agreement, with an absolute floor for values near zero.
expect_rel <- function(got, want, tol, info = NULL) {
    d <- abs(got - want) / pmax(abs(want), 1e-8)
    expect_lt(max(d), tol, label = info %||% "max relative difference")
}
`%||%` <- function(a, b) if (is.null(a)) b else a

test_that("lgamma and lchoose reproduce R over the count-family range", {
    # Counts and dispersions: y + r and r in a negative binomial, n and y in a
    # binomial or beta-binomial, all positive and mostly small.
    x <- c(1e-3, 0.01, 0.5, 1, 1.5, 2, 3, 6.5, 7, 7.5, 12, 50, 500, 1e4, 1e7)
    expect_rel(.pm("lgamma", x), lgamma(x), 1e-13)

    n <- c(1, 2, 5, 10, 40, 250, 5000)
    for (nn in n) {
        k <- 0:nn
        expect_rel(.pm("lchoose", rep(nn, length(k)), k),
                   lchoose(nn, k), 1e-11, info = paste("lchoose n =", nn))
    }
    # Out of range is -Inf, matching R.
    expect_equal(.pm("lchoose", c(5, 5), c(-1, 6)), c(-Inf, -Inf))
})

test_that("the polygamma ladder reproduces R across the recurrence switch", {
    # The recurrence shifts x up to >= 7 and then takes the asymptotic series,
    # so the switch itself has to be covered, not only its two sides.
    #
    # The tolerances widen up the ladder and are the series' own truncation
    # error, worst just past the switch: measured maxima over this grid are
    # 2.1e-14 / 1.6e-11 / 1.9e-10 / 1.2e-09, each attained at x = 7 where the
    # recurrence stops shifting and the asymptotic expansion is evaluated at
    # its smallest argument. Away from the switch every one of them is at
    # machine precision.
    x <- c(0.05, 0.2, 0.9, 1, 1.7, 3.3, 6.5, 6.999, 7, 7.001, 9, 25, 400, 1e5)
    expect_rel(.pm("digamma", x),    digamma(x),         1e-12)
    expect_rel(.pm("trigamma", x),   trigamma(x),        1e-10)
    expect_rel(.pm("tetragamma", x), psigamma(x, 2L),    1e-9)
    expect_rel(.pm("pentagamma", x), psigamma(x, 3L),    1e-8)

    # Well away from the switch the agreement is machine precision, which is
    # what says the widened tolerances above are the switch and not drift.
    far <- c(20, 60, 250, 1e4)
    expect_rel(.pm("trigamma", far),   trigamma(far),     1e-14)
    expect_rel(.pm("tetragamma", far), psigamma(far, 2L), 1e-14)
    expect_rel(.pm("pentagamma", far), psigamma(far, 3L), 1e-14)
})

test_that("the normal CDF and density reproduce R, including the deep tails", {
    z <- c(-60, -45, -37.6, -37.5, -37.4, -20, -8, -3, -1, 0, 1, 3, 8, 20, 60)

    # The natural-scale CDF underflows below about -37.5 in R as well, so
    # compare where R itself still carries digits.
    zf <- z[z > -37]
    expect_rel(.pm("pnorm", zf), pnorm(zf), 1e-13)

    # log Phi stays finite everywhere; past the erfc underflow the header
    # switches to the asymptotic Mills ratio, whose stated truncation error at
    # the switch is ~2e-13 relative and falls from there.
    expect_rel(.pm("pnorm_log", z), pnorm(z, log.p = TRUE), 1e-12)

    expect_rel(.pm("dnorm", z),     dnorm(z), 1e-13)
    expect_rel(.pm("dnorm_log", z), dnorm(z, log = TRUE), 1e-13)
})

test_that("log1m_exp is accurate on both sides of its split", {
    # log(1 - exp(-a)); the split at log 2 keeps the accurate form on each side.
    a <- c(1e-12, 1e-6, 0.1, 0.6931471, 0.6931472, 1, 5, 30, 100, 700)
    ref <- log(-expm1(-a))                    # exact in double for a <= log 2
    ref[a > log(2)] <- log1p(-exp(-a[a > log(2)]))
    expect_rel(.pm("log1m_exp", a), ref, 1e-13)
    expect_equal(.pm("log1m_exp", 0), -Inf)
})

test_that("no Rmath call survives on the family parallel path", {
    # The swap is only as good as its coverage: a reintroduced R::lgammafn in
    # either family header puts an R error path back inside the reduction.
    hdr <- testthat::test_path("..", "..", "src",
                               c("laplace_family_link.h",
                                 "laplace_family_curvature.h"))
    skip_if_not(all(file.exists(hdr)), "package sources not available")
    src <- unlist(lapply(hdr, readLines, warn = FALSE))
    expect_equal(grep("R::", src, fixed = TRUE, value = TRUE), character(0))
})
