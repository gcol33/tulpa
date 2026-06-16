# Tests for the cell-coupling registry + R-side cell_coupling argument on
# tulpa_nested_laplace_joint(). Layer A of gcol33/tulpa#32 Change 2 (the
# per-cell branch in the inner Newton lands with Layer B; here we test
# only the plumbing: the registry, its default, and the R-side validator).

test_that("registry holds the separable default after first touch", {
    # ensure_defaults_registered() fires lazily; the size query touches it.
    sz <- cpp_cell_coupling_registry_size()
    expect_true(sz >= 1L)
    expect_true(cpp_cell_coupling_registry_has("separable"))
})

test_that("registry lookup returns FALSE on an unregistered name", {
    expect_false(cpp_cell_coupling_registry_has(
        "this_name_should_never_be_registered_xyzzy"))
})

test_that("tulpa_nested_laplace_joint() accepts cell_coupling = \"separable\" silently", {
    skip_on_cran()
    # 2-arm cover-hurdle on a tiny synthetic ICAR (smoke test the
    # cell_coupling default; cover hurdle stays on the per-obs path because
    # the separable sentinel has arm_ids().empty()).
    set.seed(1L)
    n_s <- 6L
    adj_row_ptr <- c(0L, 1L, 3L, 4L, 6L, 8L, 9L)
    adj_col_idx <- c(2L, 1L, 3L, 2L, 1L, 5L, 4L, 6L, 5L) - 1L
    n_neighbors <- c(1L, 2L, 1L, 2L, 2L, 1L)
    N <- 30L
    spatial_idx <- as.integer(rep(seq_len(n_s), length.out = N))
    X <- matrix(1, nrow = N, ncol = 1)
    arm1 <- list(
        y = rbinom(N, 1, 0.4), n_trials = rep(1L, N),
        X = X, spatial_idx = spatial_idx, family = "binomial", phi = 1
    )
    arm2 <- list(
        y = rnorm(N), n_trials = rep(1L, N),
        X = X, spatial_idx = spatial_idx, family = "gaussian", phi = 1
    )

    res <- tulpa_nested_laplace_joint(
        responses = list(occ = arm1, pos = arm2),
        prior = list(type = "icar", n_spatial_units = n_s,
                     adj_row_ptr = adj_row_ptr, adj_col_idx = adj_col_idx,
                     n_neighbors = n_neighbors,
                     sigma_grid = c(0.5, 1.0)),
        cell_coupling = "separable",
        control = list(max_iter = 10L)
    )

    expect_s3_class(res, "tulpa_nested_laplace_joint")
    expect_identical(res$cell_coupling, "separable")
})

test_that("tulpa_nested_laplace_joint() rejects an unknown cell_coupling name", {
    arm <- list(
        y = c(0, 1), n_trials = c(1L, 1L),
        X = matrix(1, 2, 1), spatial_idx = c(1L, 2L),
        family = "binomial", phi = 1
    )
    expect_error(
        tulpa_nested_laplace_joint(
            responses = list(occ = arm),
            prior = list(type = "icar", n_spatial_units = 2L,
                         adj_row_ptr = c(0L, 1L, 2L),
                         adj_col_idx = c(1L, 0L), n_neighbors = c(1L, 1L)),
            cell_coupling = "no_such_spec"
        ),
        regexp = "not registered"
    )
})

test_that("tulpa_nested_laplace_joint() rejects non-character cell_coupling", {
    arm <- list(
        y = c(0, 1), n_trials = c(1L, 1L),
        X = matrix(1, 2, 1), spatial_idx = c(1L, 2L),
        family = "binomial", phi = 1
    )
    expect_error(
        tulpa_nested_laplace_joint(
            responses = list(occ = arm),
            prior = list(type = "icar", n_spatial_units = 2L,
                         adj_row_ptr = c(0L, 1L, 2L),
                         adj_col_idx = c(1L, 0L), n_neighbors = c(1L, 1L)),
            cell_coupling = NA_character_
        ),
        regexp = "single non-empty character"
    )
})
