# Sparse path numerical equivalence against the dense path
# (gcol33/tulpa: Stage 1.5b of the joint sparse rollout).
#
# The joint multi-block kernel auto-routes to a sparse Newton path when
# n_x crosses SPARSE_THRESHOLD or when any block contributes through
# non-INDEXED_SINGLE semantics. At small n_x with classic blocks
# (icar/bym2/car_proper/rw1/rw2/ar1/iid) both paths produce mathematically
# identical Hessians; force_sparse = TRUE lets us exercise the sparse code
# at small scale against the validated dense reference.
#
# Tolerance: 1e-6 on log_marginal is well within Newton-step quantization;
# the modes typically agree to <1e-7 component-wise. We assert log_marginal
# equality first because mode comparison can have spurious differences
# when convergence stops early on either path.

skip_on_cran()

.chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    n_neighbors <- vapply(nbr, length, integer(1))
    list(
        adj_row_ptr = as.integer(c(0L, cumsum(n_neighbors))),
        adj_col_idx = as.integer(unlist(nbr)) - 1L,
        n_neighbors = as.integer(n_neighbors),
        n_spatial_units = n_s
    )
}

.sim_joint_small <- function(seed = 42L, N = 200L, n_s = 25L,
                              sigma = 0.6, rho = 0.7, alpha_true = 1.0,
                              tau_b = 0.5) {
    set.seed(seed)
    adj <- .chain_adj(n_s)
    # Sample a smooth field via independent draws + smoothing (good enough
    # for a numerical-equivalence test; we don't care about recovery).
    w <- as.numeric(arima.sim(n = n_s, list(ar = 0.6))) * sigma
    w <- w - mean(w)

    s_idx_1 <- sample(seq_len(n_s), N, replace = TRUE)
    s_idx_2 <- sample(seq_len(n_s), N, replace = TRUE)
    x1 <- rnorm(N); x2 <- rnorm(N)

    eta1 <- 0.3 + 0.5 * x1 + w[s_idx_1]
    eta2 <- -0.2 + 0.4 * x2 + alpha_true * w[s_idx_2]

    y1 <- rbinom(N, 1L, plogis(eta1))
    y2 <- rnorm(N, mean = eta2, sd = 0.4)

    list(
        n_s = n_s, N = N, adj = adj,
        responses = list(
            occ = list(y = as.numeric(y1), n_trials = rep(1L, N),
                       X = cbind(intercept = 1, x = x1),
                       spatial_idx = s_idx_1,
                       re_idx = rep(0L, N), n_re_groups = 0L,
                       sigma_re = 1.0,
                       family = "binomial", phi = 1.0),
            pos = list(y = as.numeric(y2), n_trials = rep(1L, N),
                       X = cbind(intercept = 1, x = x2),
                       spatial_idx = s_idx_2,
                       re_idx = rep(0L, N), n_re_groups = 0L,
                       sigma_re = 1.0,
                       family = "gaussian", phi = 0.16)
        )
    )
}

.expect_equiv_fits <- function(fit_dense, fit_sparse,
                                 lm_tol = 1e-6, mode_tol = 1e-5) {
    expect_equal(length(fit_dense$log_marginal),
                 length(fit_sparse$log_marginal))
    # Per-cell log_marginal must agree.
    expect_equal(
        as.numeric(fit_sparse$log_marginal),
        as.numeric(fit_dense$log_marginal),
        tolerance = lm_tol,
        info = "per-cell log_marginal disagreement between dense and sparse paths"
    )
    # Modes should agree elementwise when both fits stored them.
    if (!is.null(fit_dense$modes) && !is.null(fit_sparse$modes) &&
        is.matrix(fit_dense$modes) && is.matrix(fit_sparse$modes)) {
        expect_equal(dim(fit_sparse$modes), dim(fit_dense$modes))
        expect_equal(as.numeric(fit_sparse$modes),
                     as.numeric(fit_dense$modes),
                     tolerance = mode_tol,
                     info = "per-cell mode disagreement between dense and sparse paths")
    }
}

test_that("sparse path matches dense on joint BYM2 with copy", {
    sim <- .sim_joint_small(seed = 11L, alpha_true = 1.2)
    prior <- c(
        list(type = "bym2",
             sigma_grid = c(0.5, 0.6),
             rho_grid   = c(0.7)),
        sim$adj
    )
    responses <- sim$responses
    responses$pos$field_coef <- list(name = "alpha", grid = c(1.0, 1.2))

    fit_dense  <- tulpa_nested_laplace_joint(
        responses = responses, prior = prior,
        control = list(max_iter = 40L, tol = 1e-8, n_threads = 1L, verbose = FALSE,
                       force_sparse = FALSE)
    )
    fit_sparse <- tulpa_nested_laplace_joint(
        responses = responses, prior = prior,
        control = list(max_iter = 40L, tol = 1e-8, n_threads = 1L, verbose = FALSE,
                       force_sparse = TRUE)
    )
    .expect_equiv_fits(fit_dense, fit_sparse)
})

test_that("sparse path matches dense on joint ICAR (no copy)", {
    sim <- .sim_joint_small(seed = 12L, alpha_true = 1.0)
    prior <- c(
        list(type = "icar",
             sigma_grid = c(0.5, 0.7)),
        sim$adj
    )

    fit_dense  <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = prior, copy = NULL,
        control = list(max_iter = 40L, tol = 1e-8, n_threads = 1L, verbose = FALSE,
                       force_sparse = FALSE)
    )
    fit_sparse <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = prior, copy = NULL,
        control = list(max_iter = 40L, tol = 1e-8, n_threads = 1L, verbose = FALSE,
                       force_sparse = TRUE)
    )
    .expect_equiv_fits(fit_dense, fit_sparse)
})

test_that("sparse path matches dense on joint CAR_proper", {
    sim <- .sim_joint_small(seed = 13L)
    prior <- c(
        list(type = "car_proper",
             sigma_grid   = c(0.5),
             rho_car_grid = c(0.9, 0.95)),
        sim$adj
    )

    fit_dense  <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = prior, copy = NULL,
        control = list(max_iter = 40L, tol = 1e-8, n_threads = 1L, verbose = FALSE,
                       force_sparse = FALSE)
    )
    fit_sparse <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = prior, copy = NULL,
        control = list(max_iter = 40L, tol = 1e-8, n_threads = 1L, verbose = FALSE,
                       force_sparse = TRUE)
    )
    .expect_equiv_fits(fit_dense, fit_sparse)
})

# --- gcol33/tulpa#432: what the two paths EXPORT, not only what they report ---
#
# The comparisons above read log_marginal and the modes. The sparse loop also
# hands out a precision (store_Q) and a fixed-effect covariance block, and it
# used to hand out `H_builder` AFTER joint_pd_step_solve had run: in LM mode
# that call loads the diagonal on every failed factorization and never takes the
# load back off, so the export was of H + lambda I. A covariance read off that is
# smaller than the one read off H, so the standard errors come out low with
# nothing on the result saying so. The dense sibling gated the same export on a
# clean factorization; the sparse loop, whose header says it mirrors it exactly,
# carried no equivalent condition.
#
# The export is now a snapshot of the values the scatter left behind, taken
# before the conditioning step, and `pd_conditioned` records whether that step
# had to do anything.

.expect_equiv_fixed <- function(fit_dense, fit_sparse, tol = 1e-7) {
    sd_ <- summary(fit_dense)
    ss_ <- summary(fit_sparse)
    expect_equal(as.numeric(ss_$estimate), as.numeric(sd_$estimate),
                 tolerance = tol, info = "fixed-effect estimates")
    expect_equal(as.numeric(ss_$std.error), as.numeric(sd_$std.error),
                 tolerance = tol, info = "fixed-effect standard errors")
    expect_equal(as.numeric(ss_[["2.5%"]]), as.numeric(sd_[["2.5%"]]),
                 tolerance = tol, info = "lower bounds")

    # The retained per-cell fixed-effect precisions, which is what the summary
    # marginalizes. Compared relatively: these run to 1e3 on this fixture.
    expect_false(is.null(fit_dense$grid_hessians))
    expect_false(is.null(fit_sparse$grid_hessians))
    expect_length(fit_sparse$grid_hessians, length(fit_dense$grid_hessians))
    for (k in seq_along(fit_dense$grid_hessians)) {
        hd <- fit_dense$grid_hessians[[k]]
        hs <- fit_sparse$grid_hessians[[k]]
        expect_equal(dim(hs), dim(hd), info = paste("cell", k))
        expect_lt(max(abs(hs - hd)) / max(abs(hd)), tol)
    }
}

.fit_joint_pair <- function(prior, responses, ...) {
    ctl <- utils::modifyList(
        list(max_iter = 40L, tol = 1e-8, n_threads = 1L, verbose = FALSE,
             keep_grid_hessians = TRUE, diagnose_k = FALSE, progress = FALSE),
        list(...))
    lapply(c(FALSE, TRUE), function(fs) {
        tulpa_nested_laplace_joint(
            responses = responses, prior = prior, copy = NULL,
            control = utils::modifyList(ctl, list(force_sparse = fs)))
    })
}

test_that("the two paths export the same fixed-effect block and precision", {
    sim <- .sim_joint_small(seed = 12L, alpha_true = 1.0)
    for (prior in list(
            c(list(type = "icar", sigma_grid = c(0.5, 0.7)), sim$adj),
            c(list(type = "bym2", sigma_grid = c(0.5, 0.6),
                   rho_grid = c(0.7)), sim$adj),
            c(list(type = "car_proper", sigma_grid = c(0.5),
                   rho_car_grid = c(0.9, 0.95)), sim$adj))) {
        fits <- .fit_joint_pair(prior, sim$responses)
        .expect_equiv_fixed(fits[[1]], fits[[2]])
    }
})

test_that("the export survives the sum-to-zero rank-1 storage", {
    # With the augmentation's 1 1' left OFF the stored H and folded in at solve
    # time instead, the sparse loop's factorization is of a matrix whose
    # constant direction is unpinned -- the arrangement the export was taken
    # from. TULPA_S2Z_DENSIFY_MAX = 0 forces it at any field size.
    old <- Sys.getenv("TULPA_S2Z_DENSIFY_MAX", unset = NA)
    on.exit(if (is.na(old)) Sys.unsetenv("TULPA_S2Z_DENSIFY_MAX")
            else Sys.setenv(TULPA_S2Z_DENSIFY_MAX = old), add = TRUE)

    sim   <- .sim_joint_small(seed = 12L, alpha_true = 1.0)
    prior <- c(list(type = "icar", sigma_grid = c(0.5, 0.7)), sim$adj)

    dense_store <- .fit_joint_pair(prior, sim$responses)[[2]]
    Sys.setenv(TULPA_S2Z_DENSIFY_MAX = "0")
    fits <- .fit_joint_pair(prior, sim$responses)
    .expect_equiv_fixed(fits[[1]], fits[[2]], tol = 1e-6)

    # ... and the rank-1 storage gives the same standard errors as storing the
    # augmentation exactly, so the fold is not silently shrinking them.
    expect_equal(as.numeric(summary(fits[[2]])$std.error),
                 as.numeric(summary(dense_store)$std.error),
                 tolerance = 1e-6)
})

test_that("a conditioned factorization is reported rather than silent", {
    coupled_occ_register()
    # The occupancy mixture's dark-cell term is not concave, so a fit stopped
    # after one Newton iteration sits at a point whose Hessian is indefinite and
    # the final factorization has to condition it. That is the state the export
    # gate exists for, and the flag is what makes it readable from R.
    d <- coupled_occ_data(seed = 500001L, n_cells = 100L, n_visits = 4L,
                          b_occ = -2.5, b_det = -1.0)
    fit_at <- function(fs, max_iter) suppressWarnings(tulpa_nested_laplace_joint(
        responses = coupled_occ_arms(d, beta_prec = 0.25),
        prior = coupled_occ_flat_prior(d),
        cell_coupling = "test_occupancy_mixture",
        control = list(max_iter = max_iter, tol = 1e-10, n_threads = 1L,
                       keep_grid_hessians = TRUE, diagnose_k = FALSE,
                       diagnose_skew = FALSE, auto_recenter = FALSE,
                       progress = FALSE, force_sparse = fs)))

    for (fs in c(FALSE, TRUE)) {
        stalled <- fit_at(fs, 1L)
        expect_false(as.logical(stalled$converged)[1L], info = paste("sparse", fs))
        expect_true(all(stalled$pd_conditioned), info = paste("sparse", fs))
        # Nothing conditioned reaches a coefficient table: the retention
        # declines upstream, on the cause that stopped the solve.
        expect_identical(stalled$grid_fixed_declined, "not_converged")
        expect_true(all(is.na(coef(stalled))))

        converged <- fit_at(fs, 300L)
        expect_true(as.logical(converged$converged)[1L])
        expect_false(any(converged$pd_conditioned), info = paste("sparse", fs))
        expect_true(is.na(converged$grid_fixed_declined))
    }
})

test_that("the two backends agree on whether they had to condition", {
    # A verdict that depended on the backend would put the dense and sparse
    # reads of one model into disagreement about whether its export is clean.
    sim   <- .sim_joint_small(seed = 12L)
    prior <- c(list(type = "icar", sigma_grid = c(0.5, 0.7)), sim$adj)
    fits  <- .fit_joint_pair(prior, sim$responses)
    expect_identical(as.logical(fits[[2]]$pd_conditioned),
                     as.logical(fits[[1]]$pd_conditioned))
    # This fixture factorizes on the first attempt, which is what makes it the
    # control arm for the coupled test above.
    expect_false(any(fits[[1]]$pd_conditioned))
})
