# The coupled joint Newton from an INDEFINITE start Hessian (gcol33/tulpa#344).
#
# The occupancy mixture's dark-cell term log(psi (1-p)^J + 1 - psi) is not
# concave in (eta_occ, eta_det), so a data set with few detections has a
# negative curvature direction at the Newton start x = 0. A plain Cholesky has
# no factor there, and the dense joint loop -- which every small coupled fit
# runs on -- had no conditioner: the step came back non-finite, the line search
# accepted nothing, and the solve reported its START VECTOR as the mode after
# any number of iterations.
#
# Every coupled fixture in the suite sat at b_occ = 0.2, b_det = -0.5, where the
# start Hessian is positive definite, which is why nothing caught it. The
# fixture below is the sparse-detection counterpart, and it ESTABLISHES that it
# reaches the regime (lambda_min < 0 from the R log posterior, not from the
# engine) rather than reaching it by luck.

PREC_344 <- 0.25

# b_occ = -2.5, b_det = -1.0 leaves 2-9 of 100 cells with any detection and puts
# the start Hessian's smallest eigenvalue near -13; the b_occ = 0.2 control is
# the setting the rest of the coupled suite uses, near +20.
data_344 <- function(sparse_detection, seed = 500001L) {
    if (sparse_detection) {
        coupled_occ_data(seed = seed, n_cells = 100L, n_visits = 4L,
                         b_occ = -2.5, b_det = -1.0)
    } else {
        coupled_occ_data(seed = seed, n_cells = 100L, n_visits = 4L,
                         b_occ = 0.2, b_det = -0.5)
    }
}

fit_344 <- function(d, ...) {
    ctl <- utils::modifyList(
        list(max_iter = 300L, tol = 1e-10, n_threads = 1L,
             keep_grid_hessians = TRUE, diagnose_k = FALSE,
             diagnose_skew = TRUE, auto_recenter = FALSE, progress = FALSE),
        list(...))
    suppressWarnings(tulpa_nested_laplace_joint(
        responses = coupled_occ_arms(d, beta_prec = PREC_344),
        prior = coupled_occ_flat_prior(d),
        cell_coupling = "test_occupancy_mixture",
        control = ctl))
}


test_that("the sparse-detection fixture reaches the indefinite-start regime", {
    d  <- data_344(TRUE)
    lp <- coupled_occ_log_post(d, PREC_344)
    cur <- coupled_occ_curvature(lp)

    # Fewer than a tenth of the cells carry any detection, and the negative
    # curvature direction that produces is what the conditioner has to handle.
    expect_lt(sum(d$n_seen > 0), 10L)
    expect_lt(cur$lambda_min, -5)

    # The control: the setting every other coupled fixture uses is PD at the
    # same start, so it cannot exercise this path.
    ctrl <- coupled_occ_curvature(coupled_occ_log_post(data_344(FALSE),
                                                       PREC_344))
    expect_gt(ctrl$lambda_min, 0)
})


test_that("the coupled joint Newton steps from an indefinite start Hessian", {
    skip_on_cran()
    coupled_occ_register()

    d   <- data_344(TRUE)
    ref <- coupled_occ_ref_mode(coupled_occ_log_post(d, PREC_344))
    f   <- fit_344(d)

    expect_true(as.logical(f$converged)[1L])
    # A handful of iterations, not the budget: the failure this closes reported
    # `n_iter = max_iter` with the step vector untouched at any budget.
    expect_lt(as.integer(f$n_iter)[1L], 40L)
    expect_lt(as.numeric(f$score_max)[1L], 1e-8)
    expect_equal(as.numeric(f$modes[1L, 1:2]), ref, tolerance = 1e-5)
    # The start is 2.3 away from the mode, so a solve that reported the start
    # would pass none of the above.
    expect_gt(max(abs(ref)), 2)
    expect_true(is.finite(as.numeric(f$log_marginal)[1L]))
})


test_that("the dense and sparse joint loops agree from an indefinite start", {
    skip_on_cran()
    coupled_occ_register()

    # The sparse loop already conditioned the Hessian (joint_pd_step_solve), so
    # it is the arbiter for what the dense loop now does: same policy, different
    # factorization backend.
    for (sparse_detection in c(TRUE, FALSE)) {
        d <- data_344(sparse_detection)
        a <- fit_344(d)
        b <- fit_344(d, force_sparse = TRUE)
        expect_true(as.logical(a$converged)[1L])
        expect_true(as.logical(b$converged)[1L])
        expect_equal(as.numeric(a$modes[1L, ]), as.numeric(b$modes[1L, ]),
                     tolerance = 1e-10)
        expect_equal(as.numeric(a$log_marginal)[1L],
                     as.numeric(b$log_marginal)[1L], tolerance = 1e-10)
        expect_identical(as.integer(a$n_iter)[1L], as.integer(b$n_iter)[1L])
    }
})


test_that("the positive-definite start takes the plain Newton step it always did", {
    skip_on_cran()
    coupled_occ_register()

    # With H already PD the first factorization attempt succeeds, no diagonal
    # load is added, and the step IS the Newton step -- so this fit's numbers
    # are pinned against a change to the conditioner leaking into the PD path.
    d <- data_344(FALSE)
    f <- fit_344(d)
    expect_true(as.logical(f$converged)[1L])
    expect_identical(as.integer(f$n_iter)[1L], 6L)
    expect_equal(as.numeric(f$modes[1L, 1:2]),
                 c(0.4709966040, -0.7866276485), tolerance = 1e-8)
    expect_equal(as.numeric(f$log_marginal)[1L], -194.1857868, tolerance = 1e-6)
})


test_that("a fit that did not converge declines with the cause that stopped it", {
    skip_on_cran()
    coupled_occ_register()

    # One Newton iteration cannot reach the mode, so this is a solve that
    # stopped short of one -- the state every downstream report has to name
    # correctly.
    d <- data_344(TRUE)
    f <- fit_344(d, max_iter = 1L)
    expect_false(as.logical(f$converged)[1L])

    # `backend_unsupported` is FALSE here: the same backend on the same fixture
    # computes gamma_3 on every converged solve.
    conv <- fit_344(d)
    expect_true(is.na(conv$inner_skew_declined))
    expect_true(any(is.finite(conv$inner_skew)))

    expect_identical(f$inner_skew_declined, "not_converged")
    expect_identical(sub(":.*$", "", f$inner_pareto_k_declined), "not_converged")
    # `block_not_extracted` names a retention step downstream of the real cause.
    expect_identical(f$grid_fixed_declined, "not_converged")
})


test_that("a non-converged fit reports no coefficient read off its start vector", {
    skip_on_cran()
    coupled_occ_register()

    d <- data_344(TRUE)
    f <- fit_344(d, max_iter = 1L)

    # `$modes` stays: it is the solver's record of where it stopped, and the
    # warm-start chain and the refinement passes read it. What must not happen
    # is that vector surfacing as an estimate.
    expect_true(is.matrix(f$modes))
    expect_true(all(is.finite(f$modes[1L, ])))

    est <- coef(f)
    expect_true(all(is.na(est)))
    ci <- confint(f)
    expect_true(all(is.na(ci)))
    expect_identical(attr(ci, "interval_declined"), "not_converged")
})


test_that("the convergence predicates read the cells the weights put mass on", {
    ok <- tulpa:::.nested_converged_cells(
        list(converged = c(TRUE, FALSE, NA)), 3L)
    expect_identical(ok, c(TRUE, FALSE, FALSE))

    # No flag at all is a backend that does not report one, not a stalled solve.
    expect_identical(tulpa:::.nested_converged_cells(list(), 2L),
                     c(TRUE, TRUE))
    expect_true(tulpa:::.nested_any_weighted_converged(list()))

    # A converged cell carrying no weight does not rescue the fit.
    expect_false(tulpa:::.nested_any_weighted_converged(
        list(converged = c(TRUE, FALSE), weights = c(0, 1))))
    expect_true(tulpa:::.nested_any_weighted_converged(
        list(converged = c(TRUE, FALSE), weights = c(1, 0))))

    # Weights that name no cell leave every cell readable rather than none.
    expect_true(tulpa:::.nested_any_weighted_converged(
        list(converged = c(TRUE, FALSE), weights = c(NA, NA))))
    expect_false(tulpa:::.nested_any_weighted_converged(
        list(converged = c(FALSE, FALSE), weights = c(NA, NA))))
})
