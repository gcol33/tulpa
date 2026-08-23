# A family entry with no registered closed form returns NaN, never a
# plausible-looking number computed from the wrong input.
#
# gcol33/tulpa#464: obs_curvature_deta2_for_family() summed the working
# curvature's second eta-derivative with the observed-minus-working delta's.
# Where the delta's second derivative is unregistered that term contributes 0
# and the sum is the WORKING answer presented as the OBSERVED one -- finite,
# smooth, and wrong by exactly the delta. The closed-form outer Hessian is
# built from it, so the error lands in the hyperparameter standard errors and
# the grid weights while the mode and the outer gradient stay right.
#
# gcol33/tulpa#459: an unregistered family name is a programming error the
# package raises deliberately, and the raise has to reach R. Every dispatch
# that sits inside an OpenMP reduction resolves the family on the calling
# thread first, because an exception leaving a structured block is
# std::terminate -- the session, not an error message.

.families_with_2nd <- c("poisson", "binomial", "gaussian", "neg_binomial_2",
                        "neg_binomial_1", "gamma", "beta", "beta_binomial",
                        "t", "lognormal", "truncated_poisson",
                        "truncated_neg_binomial_2")

test_that("the observed-curvature 2nd derivative declines with NaN, not a number", {
    declined <- character(0)
    for (f in .families_with_2nd) {
        ok <- tryCatch(tulpa:::cpp_family_has_obs_curvature_2nd_derivative(f),
                       error = function(e) NA)
        if (is.na(ok)) next
        v <- tryCatch(
            tulpa:::cpp_family_obs_curvature_deta2(
                y = 2, n_trials = 10L, eta = 0.3, family = f, phi = 1.5,
                phi2 = 4.0),
            error = function(e) NULL)
        if (is.null(v)) next
        # The flag and the value say the same thing: a declining family gets
        # NaN, an admitted one gets a finite number.
        expect_equal(unname(v[["exact"]]), if (ok) 1 else 0, info = f)
        if (ok) {
            expect_true(is.finite(v[["d2w_obs_deta2"]]), info = f)
        } else {
            expect_true(is.nan(v[["d2w_obs_deta2"]]), info = f)
        }
        if (!ok) declined <- c(declined, f)
    }
    # The gate has to actually decline something, else the test above passes on
    # an empty set.
    expect_gt(length(declined), 0)
})

test_that("truncated_neg_binomial_2 is the declining case, and used to be finite", {
    # Its working weight IS the observed curvature at first order but the
    # second-order delta is unregistered, so this is precisely the family whose
    # sum was the working answer wearing the observed one's name.
    expect_false(tulpa:::cpp_family_has_obs_curvature_2nd_derivative(
        "truncated_neg_binomial_2"))
    v <- tulpa:::cpp_family_obs_curvature_deta2(
        y = 3, n_trials = 1L, eta = 0.2, family = "truncated_neg_binomial_2",
        phi = 2.0)
    expect_true(is.nan(v[["d2w_obs_deta2"]]))
    # The WORKING second derivative is available and finite: the decline is a
    # statement about the observed one, not about the family being unsupported.
    w <- tulpa:::cpp_family_curvature_deta2(
        y = 3, n_trials = 1L, eta = 0.2, family = "truncated_neg_binomial_2",
        phi = 2.0)
    expect_true(is.finite(w[["d2w_deta2"]]))
})

test_that("the ZI mixture 4th-derivative entry declines in every field at once", {
    # A partial decline would let a caller read one finite field off a
    # five-field answer that has no exact form.
    f <- "beta_binomial"
    expect_false(tulpa:::cpp_family_has_zi_curvature_2nd_derivative(f))
    d <- tulpa:::cpp_zi_mixture_curvature_deriv2(
        y = c(0, 2), n_trials = c(10L, 10L), eta = c(-0.4, 0.6),
        logit_zi = c(-1.0, 0.2), family = f, phi = 3.0)
    expect_true(all(is.nan(d)))

    # An admitted family returns finite numbers through the same entry.
    d_ok <- tulpa:::cpp_zi_mixture_curvature_deriv2(
        y = c(0, 2), n_trials = c(1L, 1L), eta = c(-0.4, 0.6),
        logit_zi = c(-1.0, 0.2), family = "poisson", phi = 1.0)
    expect_true(tulpa:::cpp_family_has_zi_curvature_2nd_derivative("poisson"))
    expect_true(all(is.finite(d_ok)))
})

test_that("an unregistered family is an R error at more than one thread", {
    # The point is not that it errors -- it is WHERE. At n_threads > 1 the
    # per-observation dispatch runs on OpenMP workers, so an unhoisted
    # Rcpp::stop would abort the session instead of returning here.
    set.seed(4591L)
    N <- 40L
    X <- cbind(1, rnorm(N))
    y <- as.numeric(rpois(N, 2))
    for (nt in c(1L, 4L)) {
        expect_error(
            tulpa:::cpp_laplace_fit(
                y = y, n = rep(1L, N), X = X, re_idx = rep(0, N),
                n_re_groups = 0L, sigma_re = 1.0,
                family = "not_a_family", phi = 1.0,
                max_iter = 5L, tol = 1e-6, n_threads = nt),
            "no compiled implementation|not_a_family",
            info = paste("n_threads =", nt)
        )
    }
})

test_that("tweedie without its variance power is refused before any solve", {
    # tweedie_power() has no default and stops per observation; the stop is
    # hoisted to the calling thread, so it reaches R at any thread count.
    set.seed(4592L)
    N <- 30L
    X <- cbind(1, rnorm(N))
    y <- abs(rnorm(N)) + 0.1
    for (nt in c(1L, 4L)) {
        expect_error(
            tulpa:::cpp_laplace_fit(
                y = y, n = rep(1L, N), X = X, re_idx = rep(0, N),
                n_re_groups = 0L, sigma_re = 1.0,
                family = "tweedie", phi = 1.0,
                max_iter = 5L, tol = 1e-6, n_threads = nt),
            "phi2|variance power",
            info = paste("n_threads =", nt)
        )
    }
})
