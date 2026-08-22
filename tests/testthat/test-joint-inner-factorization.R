# The dense joint inner Newton's factorization backend
# (gcol33/tulpa#471 item 2, gcol33/tulpa#601).
#
# `control$force_sparse` chooses which DRIVER assembles the joint Hessian --
# dense or sparse. `control$inner_factorization` chooses what factorizes the
# dense driver's Hessian once it is assembled: the dense Cholesky or CHOLMOD.
# The two are independent, and until this control existed the second choice was
# fixed by the latent dimension, so no joint problem could be driven through
# both backends the way the single-arm loop's `sparse_override` allows.
#
# The Hessian is the same matrix on both settings, so the fits must agree to
# factorization noise. That is the gate.

skip_on_cran()

.chain_adj_if <- function(n_s) {
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

.if_fixture <- function(seed = 71L, n_s = 18L, N1 = 120L, N2 = 100L) {
    set.seed(seed)
    adj <- .chain_adj_if(n_s)
    rw <- cumsum(rnorm(n_s, 0, 0.6 / sqrt(n_s)))
    phi_s <- rw - mean(rw)
    s1 <- sample.int(n_s, N1, replace = TRUE)
    s2 <- sample.int(n_s, N2, replace = TRUE)
    X1 <- cbind(1, rnorm(N1))
    X2 <- cbind(1, rnorm(N2))
    y1 <- rbinom(N1, 1, plogis(X1 %*% c(-0.2, 0.4) + phi_s[s1]))
    y2 <- rnorm(N2, X2 %*% c(0.1, -0.3) + phi_s[s2], 0.5)
    list(
        adj = adj,
        responses = list(
            occ = list(y = as.numeric(y1), n_trials = rep(1L, N1), X = X1,
                       spatial_idx = s1, re_idx = rep(0, N1),
                       n_re_groups = 0L, sigma_re = 1.0,
                       family = "binomial", phi = 1.0),
            pos = list(y = y2, n_trials = rep(1L, N2), X = X2,
                       spatial_idx = s2, re_idx = rep(0, N2),
                       n_re_groups = 0L, sigma_re = 1.0,
                       family = "gaussian", phi = 1.0)
        ),
        prior = list(list(
            type = "icar",
            n_spatial_units = adj$n_spatial_units,
            adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
            n_neighbors = adj$n_neighbors,
            tau_grid = 1 / c(0.4, 0.7, 1.0)^2,
            spatial_idx = list(s1, s2)
        ))
    )
}

.if_fit <- function(fx, inner) {
    tulpa_nested_laplace_joint(
        responses = fx$responses, prior = fx$prior,
        control = list(max_iter = 40L, tol = 1e-9, n_threads = 1L,
                       verbose = FALSE, force_sparse = FALSE,
                       inner_factorization = inner)
    )
}

test_that("the dense joint loop reaches the same mode through both backends", {
    fx <- .if_fixture()
    fit_dense  <- .if_fit(fx, "dense")
    fit_sparse <- .if_fit(fx, "sparse")

    expect_equal(as.numeric(fit_sparse$log_marginal),
                 as.numeric(fit_dense$log_marginal),
                 tolerance = 1e-8)
    expect_equal(as.numeric(fit_sparse$modes), as.numeric(fit_dense$modes),
                 tolerance = 1e-7)
    # "auto" at this latent dimension is the dense factorization, so it must
    # reproduce the explicit setting exactly.
    fit_auto <- .if_fit(fx, "auto")
    expect_identical(as.numeric(fit_auto$log_marginal),
                     as.numeric(fit_dense$log_marginal))
})

test_that("an unknown inner_factorization is refused by name", {
    fx <- .if_fixture()
    expect_error(.if_fit(fx, "cholmod"), "inner_factorization")
})

test_that("the override is refused where it would not be applied", {
    fx <- .if_fixture()
    # A single-block prior takes the other dispatch, which does not thread the
    # override; a knob that silently does nothing reads as a setting that was
    # honoured.
    prior_single <- list(
        type = "icar",
        n_spatial_units = fx$adj$n_spatial_units,
        adj_row_ptr = fx$adj$adj_row_ptr, adj_col_idx = fx$adj$adj_col_idx,
        n_neighbors = fx$adj$n_neighbors,
        sigma_grid = c(0.4, 0.7, 1.0))
    expect_error(
        tulpa_nested_laplace_joint(
            responses = fx$responses, prior = prior_single,
            control = list(verbose = FALSE, inner_factorization = "sparse")
        ),
        "multi-block joint path"
    )
})

test_that("a healthy joint grid reports no sum-to-zero log-determinant fallback", {
    # gcol33/tulpa#601: the flag rides the grid per cell, so a fit that never
    # fell back carries an all-FALSE vector rather than nothing at all, and
    # diagnostic_summary() stays silent about it.
    fx <- .if_fixture()
    fit <- .if_fit(fx, "auto")
    fb <- fit$s2z_log_det_fallback
    expect_false(is.null(fb))
    expect_length(fb, length(fit$log_marginal))
    expect_false(any(as.logical(fb)))
    expect_null(tulpa:::.tulpa_s2z_fallback_cells(fit))
})

test_that("the fallback count is what diagnostic_summary reports", {
    # The count reads off the per-cell vector, so it is testable without
    # forcing a non-PD pinned matrix: a fit carrying two flagged cells must
    # report two, and one carrying none must report nothing.
    expect_null(tulpa:::.tulpa_s2z_fallback_cells(
        list(s2z_log_det_fallback = c(FALSE, FALSE, FALSE))))
    got <- tulpa:::.tulpa_s2z_fallback_cells(
        list(s2z_log_det_fallback = c(TRUE, FALSE, TRUE, FALSE)))
    expect_identical(got$n, 2L)
    expect_identical(got$n_grid, 4L)
    # A fit that predates the flag carries no vector and declines rather than
    # reporting zero fallbacks as a fact.
    expect_null(tulpa:::.tulpa_s2z_fallback_cells(list()))
})
