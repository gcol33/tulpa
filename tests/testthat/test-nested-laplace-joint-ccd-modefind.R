# CCD outer mode-find robustness (gcol33/tulpa#62).
#
# These exercise the mode-find / conditioning machinery directly against
# synthetic log-posterior objectives -- no inner Newton solves -- so they are
# fast, deterministic, and run on CRAN. They lock in the three guards that keep
# an ill-conditioned outer posterior (a sigma-alpha ridge) from burning hours of
# full-field inner solves in the line search:
#   * fast decline on a flat / ridged centre Hessian (before any line search),
#   * a trust-region clamp on the Newton step,
#   * warm-start advancement to each accepted point.

# Concave quadratic log-posterior with mode `m` and per-axis precision `prec`
# (H = -diag(prec)). Optionally records every multi-row (stencil) centre and the
# total eval count into `rec`, and attaches a "modes" attribute (pretending the
# inner latent mode equals the outer u) so the warm-start path has something to
# carry.
.mk_quad_eval <- function(m, prec, rec = NULL, modes_attr = FALSE) {
    function(U) {
        U <- matrix(U, ncol = length(m))
        if (!is.null(rec)) {
            rec$n <- rec$n + 1L
            if (nrow(U) > 1L)
                rec$centres[[length(rec$centres) + 1L]] <- U[1L, ]
        }
        lp <- apply(U, 1L, function(u) -0.5 * sum(prec * (u - m)^2))
        if (modes_attr) attr(lp, "modes") <- U
        lp
    }
}

test_that(".joint_ccd_engage applies the auto (>=4) / ccd (>=3) / grid thresholds", {
    eng <- tulpa:::.joint_ccd_engage
    # "grid" never engages, at any axis count.
    expect_false(eng("grid", 3L)); expect_false(eng("grid", 6L))
    # "auto" (default) engages only at >= 4 axes; tensor grid at <= 3.
    expect_false(eng("auto", 2L)); expect_false(eng("auto", 3L))
    expect_true(eng("auto", 4L));  expect_true(eng("auto", 6L))
    # explicit "ccd" lowers the threshold to >= 3 axes.
    expect_false(eng("ccd", 2L))
    expect_true(eng("ccd", 3L));   expect_true(eng("ccd", 4L))
})

test_that(".joint_ccd_outer_hess_ok accepts neg-def, rejects ridge / indefinite", {
    ok   <- tulpa:::.joint_ccd_outer_hess_ok
    expect_true(ok(diag(c(-1, -2, -3))))
    # near-zero curvature in one direction (a ridge) -> reject
    expect_false(ok(diag(c(-1, -1, -1e-12))))
    # indefinite (a saddle) -> reject
    expect_false(ok(diag(c(-1, 1, -1))))
    # non-finite -> reject
    expect_false(ok(matrix(c(-1, NA, NA, -1), 2L)))
    expect_false(ok(NULL))
})

test_that("mode-find converges to a well-conditioned concave mode", {
    m    <- c(0.5, -0.3, 0.2)
    eval <- .mk_quad_eval(m, prec = c(2, 3, 1.5))
    mf <- tulpa:::.joint_ccd_modefind(
        u0 = c(0, 0, 0), eval1 = eval,
        lower = rep(-5, 3), upper = rep(5, 3), h = rep(0.1, 3),
        trust = rep(2, 3))
    expect_identical(mf$status, "ok")
    expect_true(mf$converged)
    expect_equal(mf$par, m, tolerance = 1e-2)
    expect_true(tulpa:::.joint_ccd_outer_hess_ok(mf$hess))
})

test_that("mode-find declines FAST (status ridge) on a flat-curvature centre", {
    # Zero curvature along axis 3: a ridge. The centre stencil alone reveals it,
    # so the mode-find must bail before any backtracking line search.
    rec  <- new.env(); rec$n <- 0L; rec$centres <- list()
    eval <- .mk_quad_eval(c(0, 0, 0), prec = c(2, 3, 0), rec = rec)
    mf <- tulpa:::.joint_ccd_modefind(
        u0 = c(0.1, 0.1, 0.1), eval1 = eval,
        lower = rep(-5, 3), upper = rep(5, 3), h = rep(0.1, 3),
        trust = rep(2, 3))
    expect_identical(mf$status, "ridge")
    expect_false(mf$converged)
    # One centre eval + one stencil = 2 calls; crucially, NO line-search probes.
    expect_lte(rec$n, 3L)
})

test_that("the Newton step is trust-clamped so no round leaps past `trust`", {
    # Mode far from the start with a small trust radius: each accepted step is
    # bounded per coordinate, so the iterate crawls rather than jumping to the
    # edge (which on a real fit would be an extreme, slow-to-solve hyperparameter).
    rec   <- new.env(); rec$n <- 0L; rec$centres <- list()
    m     <- c(4, -4, 3)
    trust <- rep(0.25, 3)
    eval  <- .mk_quad_eval(m, prec = c(1, 1, 1), rec = rec)
    mf <- tulpa:::.joint_ccd_modefind(
        u0 = c(0, 0, 0), eval1 = eval,
        lower = rep(-10, 3), upper = rep(10, 3), h = rep(0.1, 3),
        max_rounds = 3L, trust = trust)
    centres <- do.call(rbind, rec$centres)
    expect_gte(nrow(centres), 2L)
    steps <- abs(diff(centres))
    # Every per-coordinate step stays within the trust radius (+ FD slack).
    expect_true(all(steps <= trust + 1e-6))
})

test_that("the line search is capped at `max_halve` evaluations per round", {
    # A descent-only objective (start sits past the mode along a steep axis) with
    # a bad search direction forces the line search to backtrack; the cap bounds
    # how many full-field solves a single round can cost.
    rec  <- new.env(); rec$n <- 0L; rec$centres <- list()
    eval <- .mk_quad_eval(c(0, 0, 0), prec = c(50, 1, 1), rec = rec)
    n_before <- 0L
    mf <- tulpa:::.joint_ccd_modefind(
        u0 = c(2, 0, 0), eval1 = eval,
        lower = rep(-5, 3), upper = rep(5, 3), h = rep(0.1, 3),
        max_rounds = 1L, max_halve = 2L, trust = rep(10, 3))
    # Round budget: 1 centre + 1 stencil + at most (max_halve + 1) line probes.
    expect_lte(rec$n, 1L + 1L + (2L + 1L))
})

test_that("on_accept receives the inner mode at each accepted point", {
    got  <- new.env(); got$modes <- list()
    eval <- .mk_quad_eval(c(0.4, -0.2, 0.1), prec = c(2, 2, 2),
                          modes_attr = TRUE)
    mf <- tulpa:::.joint_ccd_modefind(
        u0 = c(0, 0, 0), eval1 = eval,
        lower = rep(-5, 3), upper = rep(5, 3), h = rep(0.1, 3),
        trust = rep(2, 3),
        on_accept = function(mode) got$modes[[length(got$modes) + 1L]] <- mode)
    expect_identical(mf$status, "ok")
    expect_gte(length(got$modes), 1L)
    # Every advanced warm start is a finite vector of the right length.
    expect_true(all(vapply(got$modes,
                           function(v) length(v) == 3L && all(is.finite(v)),
                           logical(1))))
    # The last accepted warm start tracks the converged outer mode (modes == u).
    expect_equal(got$modes[[length(got$modes)]], mf$par, tolerance = 1e-6)
})
