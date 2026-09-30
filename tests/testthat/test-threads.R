# Inner-loop thread resolution: performance-core detection and the
# control$n_threads_scatter cap that keeps the per-observation inner OpenMP
# from oversubscribing a hybrid CPU's efficiency cores.

test_that("performance-core count is a sane value or NA", {
    pc <- tulpa:::.tulpa_perf_cores()
    if (is.na(pc)) {
        succeed()  # topology unresolved (off Windows) -- caller falls back
    } else {
        # A hardware-topology count read directly from the OS, so it is bounded
        # by physical cores, not the OpenMP/R runtime cap: under `--as-cran`
        # cpp_get_max_threads() is throttled to 2 while the true P-core count is
        # not, so comparing the two is a category error. Sane absolute bound only.
        expect_true(pc >= 1L && pc <= 1024L)
    }
})

test_that("inner threads default-cap at the performance-core count", {
    pc <- tulpa:::.tulpa_perf_cores()
    # A request far above any plausible core count.
    got <- tulpa:::.tulpa_inner_threads(1024L)
    if (is.na(pc)) {
        expect_equal(got, 1024L)        # no topology -> unchanged
    } else {
        expect_equal(got, pc)           # capped to performance cores
    }
})

test_that("n_threads_scatter overrides the default cap", {
    # Explicit cap below the request wins regardless of topology.
    expect_equal(tulpa:::.tulpa_inner_threads(16L, 4L), 4L)
    # Explicit cap above the request is a no-op (it is a cap, not a floor).
    expect_equal(tulpa:::.tulpa_inner_threads(4L, 16L), 4L)
})

test_that("requests at or below the cap pass through unchanged", {
    expect_equal(tulpa:::.tulpa_inner_threads(1L), 1L)
    expect_equal(tulpa:::.tulpa_inner_threads(1L, 8L), 1L)
})

test_that("the thread grant is every thread the fit was handed", {
    width <- function(n) tulpa:::.nl_outer_width(n)
    cap <- width(0L)
    expect_gte(cap, 1L)
    expect_identical(tulpa:::.tulpa_thread_grant(1L, 1L), 1L)
    expect_identical(tulpa:::.tulpa_thread_grant(1L, 3L), 3L)
    expect_identical(tulpa:::.tulpa_thread_grant(2L, 1L), width(2L))
    # A request of 0 is the whole team, and the grant follows it.
    expect_identical(tulpa:::.tulpa_thread_grant(0L, 1L), cap)
    # The outer request is clamped to the team before it enters the grant.
    expect_identical(tulpa:::.tulpa_thread_grant(cap + 64L, 1L), cap)
})

test_that("every joint kernel call runs its lone solves on the fit's grant", {
    skip_on_cran()
    grant <- tulpa:::.tulpa_inner_threads(tulpa:::.tulpa_thread_grant(2L, 1L))
    skip_if(grant < 2L, "the environment hands out a single thread")
    # The pinned dispersion axis sends the fit through the var-of-means
    # consistency pass, whose rounds call the kernel one cell after another
    # (gcol33/tulpa#924).
    fx <- .pgp_fixture()
    orig <- tulpa:::.cpp_joint_multi
    seen <- list()
    local_mocked_bindings(.cpp_joint_multi = function(...) {
        a <- list(...)
        seen[[length(seen) + 1L]] <<- c(inner = as.integer(a$n_threads),
                                        outer = as.integer(a$n_threads_outer))
        orig(...)
    })
    fit <- tulpa_nested_laplace_joint(
        responses = fx$responses, prior = fx$prior,
        phi_grid = list(pos = .PGP_COARSE),
        control = list(n_threads = 1L, n_threads_outer = 2L))
    expect_false(is.null(fit$var_of_means_consistency_info))
    calls <- do.call(rbind, seen)
    expect_true(any(calls[, "outer"] <= 1L))
    expect_true(all(calls[, "inner"] == grant))
})
