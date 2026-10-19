# A cell whose inner Hessian is not positive definite at the point the inner
# Newton returned is no Laplace approximation at a mode, and its stored
# precision cannot be factored. On the full 25 km occu_cover fit under a flat
# outer hyperprior one such cell carried 1.6% of the weight, and every posterior
# draw that landed in it failed its Cholesky. Such a cell is re-solved from a
# converged neighbour and, when it stays indefinite, left out of the measure
# with a record and a warning.

test_that("an indefinite cell counts as stalled for the re-solve", {
    ex <- list(list(converged = TRUE, hessian_indefinite = FALSE),
               list(converged = TRUE, hessian_indefinite = TRUE),
               list(converged = FALSE))
    cc <- tulpa:::.joint_cell_convergence(c(-1, -2, -3), ex)
    expect_identical(cc$converged, c(TRUE, FALSE, FALSE))
    expect_identical(cc$stalled,   c(FALSE, TRUE, TRUE))
})

skip_on_cran_budget()

test_that("a cell that stays indefinite leaves the measure, recorded", {
    skip_on_cran_budget()
    res <- list(log_marginal = c(0, -0.5, -1, -Inf),
                converged = rep(TRUE, 4),
                hessian_indefinite = c(FALSE, TRUE, FALSE, FALSE),
                prune_mask = c(FALSE, FALSE, FALSE, TRUE))
    out <- tulpa:::.nl_record_unconverged(res)
    expect_identical(out$indefinite_cells, 2L)
    w0 <- exp(c(0, -0.5, -1)); w0 <- w0 / sum(w0)
    expect_equal(out$indefinite_mass, w0[2])
    expect_identical(out$log_marginal[2], -Inf)
    expect_equal(out$log_marginal[c(1, 3)], c(0, -1))
    expect_length(out$nonconverged_cells, 0L)
    expect_warning(tulpa:::.nl_warn_indefinite(out, "f()"),
                   "1 outer-grid cell .*left out of the integration")
})

test_that("indefinite cells stay when nothing else was solved", {
    skip_on_cran_budget()
    res <- list(log_marginal = c(-1, -2), converged = c(TRUE, TRUE),
                hessian_indefinite = c(TRUE, TRUE))
    out <- tulpa:::.nl_record_unconverged(res)
    expect_identical(out$indefinite_cells, 1:2)
    expect_true(isTRUE(out$indefinite_kept))
    expect_equal(out$log_marginal, c(-1, -2))
    expect_warning(tulpa:::.nl_warn_indefinite(out, "f()"), "were kept")
})

test_that("a fit with no indefinite cell records none and is silent", {
    skip_on_cran_budget()
    res <- list(log_marginal = c(0, -1), converged = c(TRUE, TRUE),
                hessian_indefinite = c(FALSE, FALSE))
    out <- tulpa:::.nl_record_unconverged(res)
    expect_length(out$indefinite_cells, 0L)
    expect_identical(out$indefinite_mass, 0)
    expect_silent(tulpa:::.nl_warn_indefinite(out, "f()"))
})

skip_on_cran()

test_that("the grid kernel reports a per-cell indefinite flag", {
    skip_on_cran()
    set.seed(3)
    n_s <- 12L
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    sp <- sample.int(n_s, 300, replace = TRUE)
    y  <- rbinom(300, 1, 0.4)
    fit <- tulpa_nested_laplace_joint(
        list(occ = list(y = as.numeric(y), n_trials = rep(1L, 300),
                        X = matrix(1, 300, 1), spatial_idx = sp,
                        re_idx = rep(0, 300), n_re_groups = 0L, sigma_re = 1,
                        family = "binomial", phi = 1)),
        list(type = "icar", n_spatial_units = n_s,
             adj_row_ptr = as.integer(c(0L, cumsum(nn))),
             adj_col_idx = as.integer(unlist(nbr)) - 1L,
             n_neighbors = as.integer(nn), sigma_grid = c(0.3, 0.6, 1.2)),
        control = list(diagnose_k = FALSE))
    expect_length(fit$hessian_indefinite, nrow(as.matrix(fit$theta_grid)))
    expect_false(any(fit$hessian_indefinite))
    expect_length(fit$indefinite_cells, 0L)
})
