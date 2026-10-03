# An outer-grid cell whose inner Newton stops at `control$max_iter` without
# reaching a mode is re-solved from its nearest converged neighbour's mode; a
# cell that still has no mode stays in the measure at the value it stopped at,
# is listed on the fit with the posterior mass it carries, and is named by
# `diagnostics()`.

# A binomial occurrence arm and a beta cover arm sharing an ICAR field on an
# nr x nr lattice, the beta precision pinned to nodes away from its posterior
# so the consistency pass lays a dispersion ladder. From a cold start the inner
# Newton needs four to six steps on this fixture, so a cap of three leaves
# cells without a mode while a re-solve from a converged neighbour recovers
# some of them.
.uc_fit <- function(max_iter, nr = 8L, seed = 2L) {
    set.seed(seed)
    n <- nr * nr
    id <- function(r, c) (r - 1L) * nr + c
    adj <- lapply(seq_len(n), function(k) {
        r <- (k - 1L) %/% nr + 1L; c <- (k - 1L) %% nr + 1L
        sort(c(if (r > 1L) id(r - 1L, c), if (r < nr) id(r + 1L, c),
               if (c > 1L) id(r, c - 1L), if (c < nr) id(r, c + 1L)))
    })
    nn <- vapply(adj, length, integer(1))
    u <- as.numeric(scale(rnorm(n)))
    per_site <- 4L
    si <- rep(seq_len(n), each = per_site)
    X <- cbind(1, rnorm(n * per_site))
    occ <- rbinom(n * per_site, 1, plogis(0.3 + 0.5 * X[, 2] + 1.2 * u[si]))
    pos <- occ == 1L
    mu <- plogis(-0.2 + 0.4 * X[pos, 2] + u[si[pos]])
    y2 <- rbeta(sum(pos), mu * 3.25, (1 - mu) * 3.25)
    y2 <- pmin(pmax(y2, 1e-4), 1 - 1e-4)
    responses <- list(
        occ = list(y = as.numeric(occ), n_trials = rep(1L, n * per_site), X = X,
                   spatial_idx = si, family = "binomial"),
        pos = list(y = y2, n_trials = rep(1L, sum(pos)), X = X[pos, , drop = FALSE],
                   spatial_idx = si[pos], family = "beta", phi = 3.25,
                   field_coef = list(name = "alpha",
                                     grid = auto_grid(c(0.3, 0.6, 1, 1.5)))))
    prior <- list(type = "icar", n_spatial_units = n,
                  adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(adj) - 1L,
                  n_neighbors = nn, sigma_grid = auto_grid(c(0.5, 1, 2)))
    suppressMessages(tulpa_nested_laplace_joint(
        responses, prior, hyperprior = "proper",
        phi_grid = list(pos = c(1, 3.91, 15.3, 60)),
        control = list(diagnose_k = FALSE, progress = FALSE, max_iter = max_iter)))
}

test_that("a cell without a mode is re-solved, then recorded with its mass and named", {
    skip_on_cran()
    skip_if_fast()
    expect_warning(fit <- .uc_fit(max_iter = 3L, seed = 4L),
                   "without reaching a mode")
    solved <- is.finite(fit$log_marginal) & !as.logical(fit$prune_mask)
    stalled <- which(solved & !fit$converged)
    expect_gt(length(stalled), 0L)
    # The record is the per-cell flag, cell for cell, and the mass is the
    # weight those cells carry.
    expect_identical(fit$nonconverged_cells, stalled)
    expect_equal(fit$nonconverged_mass, sum(fit$weights[stalled]))
    expect_gt(fit$nonconverged_mass, .nl_screen("gate_mass"))
    expect_true(all(fit$n_iter[stalled] == 3L))
    expect_gt(fit$nonconverged_resolve$n_resolved, 0L)
    # The re-solve ran over every cell that first stalled; what it recovered is
    # converged now and no longer listed.
    rs <- fit$nonconverged_resolve
    expect_gte(rs$n_stalled, length(stalled))
    expect_identical(rs$n_stalled - rs$n_resolved, length(stalled))
    # The stalled cells keep their weight: nothing is dropped silently.
    expect_true(all(is.finite(fit$log_marginal[stalled])))
    expect_true(all(fit$weights[stalled] >= 0))
    # Both diagnostic doors name them.
    d <- diagnostics(fit)
    expect_match(attr(d, "unconverged_cells_note"), "without reaching a mode")
    expect_identical(attr(d, "unconverged_cells"), stalled)
    s <- diagnostic_summary(fit, quiet = TRUE)
    expect_identical(s$unconverged_cells, length(stalled))
    expect_identical(s$status, "WARN")
    expect_true(any(grepl("without reaching a mode", s$recommendations)))
})

test_that("a fit whose every cell converged records an empty list", {
    skip_on_cran()
    skip_if_fast()
    fit <- .uc_fit(max_iter = 300L)
    expect_identical(fit$nonconverged_cells, integer(0))
    expect_identical(fit$nonconverged_mass, 0)
    expect_identical(fit$nonconverged_resolve$n_stalled, 0L)
    expect_null(attr(diagnostics(fit), "unconverged_cells_note"))
    expect_identical(diagnostic_summary(fit, quiet = TRUE)$unconverged_cells, 0L)
})
