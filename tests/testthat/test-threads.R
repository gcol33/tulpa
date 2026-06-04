# Inner-loop thread resolution: performance-core detection and the
# control$n_threads_scatter cap that keeps the per-observation inner OpenMP
# from oversubscribing a hybrid CPU's efficiency cores.

test_that("performance-core count is a sane value or NA", {
    pc <- tulpa:::.tulpa_perf_cores()
    if (is.na(pc)) {
        succeed()  # topology unresolved (off Windows) -- caller falls back
    } else {
        expect_true(pc >= 1L)
        expect_true(pc <= cpp_get_max_threads())
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
