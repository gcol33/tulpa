# Outer-grid refinement on the registry door (gcol33/tulpa#927): the passes the
# joint drivers run, over the field SD axes `tulpa_nested_laplace()` can write a
# cell onto, with the per-cell side data of every added cell bound onto the base
# result.

.nrr_sim <- function(seed = 2L, nr = 12L) {
    set.seed(seed)
    n <- nr * nr
    adj <- lapply(grid_neighbours(nr, nr), sort)
    nn <- vapply(adj, length, integer(1))
    W <- matrix(0, n, n)
    for (i in seq_len(n)) W[i, adj[[i]]] <- 1
    ev <- eigen(diag(nn) - W, symmetric = TRUE)
    k <- seq_len(n - 1L)
    u <- as.numeric(ev$vectors[, k] %*% (rnorm(n - 1L) / sqrt(ev$values[k])))
    x <- rnorm(n)
    list(n = n, x = x, y = 0.2 + 0.5 * x + 2 * u + rnorm(n, 0, 0.3),
         prior = list(type = "icar", n_spatial_units = n,
                      spatial_idx = seq_len(n),
                      adj_row_ptr = c(0L, cumsum(nn)),
                      adj_col_idx = unlist(adj) - 1L, n_neighbors = nn))
}

.nrr_fit <- function(sim, prior, control = list()) {
    suppressWarnings(tulpa_nested_laplace(
        sim$y, rep(1L, sim$n), cbind(1, sim$x), prior = prior,
        family = "gaussian", phi = 0.3,
        control = c(list(progress = FALSE, diagnose_k = FALSE), control)))
}

test_that("a field SD collapsed onto few nodes is refined on the registry door", {
    skip_on_cran()
    sim <- .nrr_sim()
    held <- .nrr_fit(sim, sim$prior, list(axis_refine = "none"))
    fit  <- .nrr_fit(sim, sim$prior)
    min_ess <- tulpa:::.nl_diag("axis_sd_ess")

    expect_lt(held$theta_sd_ess[[1L]], min_ess)
    expect_false(any(nzchar(held$refining_axis %||% "")))
    expect_gte(fit$theta_sd_ess[[1L]], min_ess)
    expect_gt(length(fit$log_marginal), length(held$log_marginal))
    expect_true(any(fit$refining_axis == "tau"))
    expect_false(is.null(fit$var_of_means_consistency_info))

    # Every per-cell field has one entry per cell, and the base cells keep the
    # numbers the held fit solved.
    n <- length(fit$log_marginal)
    expect_equal(nrow(fit$modes), n)
    expect_length(fit$weights, n)
    expect_length(fit$log_quad, n)
    expect_length(fit$grid_hessians, n)
    expect_equal(nrow(fit$fitted_eta), n)
    expect_equal(sum(fit$weights), 1, tolerance = 1e-8)
    nb <- length(held$log_marginal)
    expect_equal(fit$log_marginal[seq_len(nb)], held$log_marginal)
    expect_equal(fit$modes[seq_len(nb), ], held$modes)
    # A cell the pass added is solved to the mode the dispatcher finds there.
    expect_true(all(is.finite(fit$log_marginal[-seq_len(nb)])))
    expect_true(all(is.finite(coef(fit))))
})

test_that("a multi-block registry fit refines the same cells as the single block", {
    skip_on_cran()
    sim <- .nrr_sim()
    one <- .nrr_fit(sim, sim$prior)
    multi <- .nrr_fit(sim, list(sim$prior))
    expect_equal(as.numeric(multi$log_marginal), as.numeric(one$log_marginal),
                 tolerance = 1e-8)
    expect_equal(as.numeric(multi$theta_grid[, "b1.tau"]),
                 as.numeric(one$theta_grid))
    expect_equal(as.numeric(multi$weights), as.numeric(one$weights),
                 tolerance = 1e-8)
    expect_equal(nrow(multi$modes), length(multi$log_marginal))
    expect_true(any(grepl("b1.tau", multi$refining_axis, fixed = TRUE)))
})

test_that("a registry axis the caller wrote down is densified, not extended", {
    skip_on_cran()
    sim <- .nrr_sim()
    pinned <- c(0.15, 0.2, 0.3, 0.45)
    fit <- .nrr_fit(sim, c(sim$prior, list(tau_grid = pinned)),
                    list(auto_recenter = FALSE))
    tau <- as.numeric(fit$theta_grid)
    expect_gte(min(tau), min(pinned) - 1e-12)
    expect_lte(max(tau), max(pinned) + 1e-12)
})

test_that("control$axis_refine names axes the registry fit has", {
    skip_on_cran()
    sim <- .nrr_sim()
    expect_error(.nrr_fit(sim, sim$prior, list(axis_refine = c(sigma = "none"))),
                 "does not have")
    expect_error(.nrr_fit(sim, list(sim$prior), list(axis_refine = c(tau = "none"))),
                 "does not have")
    expect_error(.nrr_fit(sim, sim$prior, list(axis_refine = c(tau = "widen"))),
                 "unknown mode")
})

test_that("only a field SD the dispatcher can write a cell onto is refinable", {
    ref <- tulpa:::.nl_refinable_registry_axes
    expect_identical(ref(list(type = "icar"), FALSE), "tau")
    expect_identical(ref(list(type = "bym2"), FALSE), "sigma")
    expect_identical(ref(list(list(type = "icar"), list(type = "iid")), TRUE),
                     c("b1.tau", "b2.sigma"))
    # The GP variance and lengthscale are not field SDs, and a bare `phi_gp` is
    # not a dispersion here.
    expect_length(ref(list(type = "nngp"), FALSE), 0L)
    expect_length(ref(list(type = "hsgp"), FALSE), 0L)
})

test_that("per-cell fields are bound by shape, constants left alone", {
    base <- list(a = c(1, 2, 3), m = matrix(1:6, 3), l = list(1, 2, 3),
                 flag = c(TRUE, FALSE, FALSE), const = c(9, 9, 9), k = 5)
    chunk <- function(n) list(res = list(a = rep(10, n), m = matrix(0L, n, 2L),
                                         l = as.list(seq_len(n)),
                                         const = c(9, 9, 9), k = 5), n = n)
    out <- tulpa:::.nl_bind_cell_results(base, list(chunk(2L), chunk(1L)), 3L)
    expect_equal(out$a, c(1, 2, 3, 10, 10, 10))
    expect_equal(dim(out$m), c(6L, 2L))
    expect_length(out$l, 6L)
    # A per-cell field the chunks do not return is filled.
    expect_identical(out$flag, c(TRUE, FALSE, FALSE, FALSE, FALSE, FALSE))
    # A constant keeps its value and length whatever the chunk sizes.
    expect_identical(out$const, c(9, 9, 9))
    expect_identical(out$k, 5)
})

test_that("a call as large as the base grid is split so shapes stay unambiguous", {
    seen <- integer(0)
    ck <- tulpa:::.nl_chunked_kernel(function(theta) {
        seen <<- c(seen, nrow(theta))
        list(log_marginal = seq_len(nrow(theta)))
    }, 4L)
    out <- ck$kernel_fn(matrix(1:8, 4, dimnames = list(NULL, c("a", "b"))))
    expect_identical(seen, c(3L, 1L))
    expect_length(out$log_marginal, 4L)
    expect_length(ck$chunks(), 2L)
})
