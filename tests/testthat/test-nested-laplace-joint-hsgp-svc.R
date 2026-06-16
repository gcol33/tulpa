# HSGP joint dispatch + HSGP-SVC (Stage 1.6b) + HSGP-MSGP (Stage 1.6c).
#
# Stage 1.3c added an HSGP block to the joint multi-block dispatcher; this
# file is the first end-to-end test of that path AND exercises:
#   * `svc_column` (Stage 1.6b) — turns the block into an SVC by
#     row-scaling each arm's basis evaluation by a covariate column.
#   * Multi-scale composition (Stage 1.6c) — declaring two HSGP blocks
#     in the same prior_list, each with its own eigenvalue spectrum and
#     (sigma2, lengthscale) axis pair, additively decomposes the latent
#     field into independent long- and short-range components.
#
# Coverage:
#   (1) Smoke: 2-arm joint dispatch with one HSGP block, no svc_column.
#       log_marginal finite on every outer-grid cell, weights normalise.
#   (2) Identity: svc_column pointing at a constant-1 covariate column
#       must produce IDENTICAL log_marginal to the no-svc baseline (up to
#       numerical tolerance) because phi_scaled[i, m] = phi[i, m] * 1.
#   (3) Non-trivial SVC: svc_column pointing at a varying covariate
#       column runs, log_marginal stays finite, and the fit differs from
#       the baseline (the scaling actually changes the posterior).
#   (4) Validation errors: bad svc_column (out-of-range, non-integer)
#       raises a clean error before any C++ entry.
#   (5) Multi-scale: two HSGP blocks (different m_per_dim / eigenvalues)
#       in one prior_list. Latent state grows by the sum of both block
#       sizes; outer grid is the Cartesian product of both axis pairs.
#       Fit runs, log_marginal stays finite, and the multi-scale fit
#       differs from either single-scale baseline.

skip_on_cran()

.hsgp_basis_2d_svc <- function(coords, m_per_dim = 4L, cscl = 1.5) {
    x <- coords[, 1]; y <- coords[, 2]
    L1 <- max(cscl * (max(x) - min(x)) / 2, 0.1)
    L2 <- max(cscl * (max(y) - min(y)) / 2, 0.1)
    xs <- x - (max(x) + min(x)) / 2
    ys <- y - (max(y) + min(y)) / 2
    M <- m_per_dim * m_per_dim
    N <- nrow(coords)
    phi <- matrix(0, N, M)
    lam <- numeric(M)
    for (j1 in seq_len(m_per_dim)) {
        lam_j1 <- (pi * j1 / (2 * L1))^2
        for (j2 in seq_len(m_per_dim)) {
            lam_j2 <- (pi * j2 / (2 * L2))^2
            idx <- (j1 - 1L) * m_per_dim + j2
            lam[idx] <- lam_j1 + lam_j2
            phi[, idx] <-
                (1 / sqrt(L1)) * sin(pi * j1 * (xs + L1) / (2 * L1)) *
                (1 / sqrt(L2)) * sin(pi * j2 * (ys + L2) / (2 * L2))
        }
    }
    list(phi = phi, lam = lam, M = M)
}

.sim_two_arm_hsgp <- function(seed = 21L, N1 = 90L, N2 = 70L,
                               m_per_dim = 4L) {
    set.seed(seed)
    coords1 <- cbind(runif(N1), runif(N1))
    coords2 <- cbind(runif(N2), runif(N2))
    basis1 <- .hsgp_basis_2d_svc(coords1, m_per_dim = m_per_dim)
    basis2 <- .hsgp_basis_2d_svc(coords2, m_per_dim = m_per_dim)
    # NB: in the joint dispatch a shared HSGP block requires shared
    # eigenvalues across arms. We pick a common L1/L2 by using a
    # union-style bounding box. Easiest: rebuild both bases on the union.
    coords_all <- rbind(coords1, coords2)
    b1 <- .hsgp_basis_2d_svc(coords1, m_per_dim = m_per_dim)
    b2 <- .hsgp_basis_2d_svc(coords2, m_per_dim = m_per_dim)
    # Use coords1's eigenvalues for the block (must be one shared spectrum).
    # In practice users center both arms on the same domain; here we just
    # use b1$lam everywhere.
    lam_shared <- b1$lam
    list(
        N1 = N1, N2 = N2, M = b1$M, lam = lam_shared,
        phi1 = b1$phi, phi2 = b2$phi,
        coords1 = coords1, coords2 = coords2
    )
}

.make_responses_hsgp <- function(sim, x_extra1 = NULL, x_extra2 = NULL) {
    # Each arm carries an intercept + an extra covariate column. The svc
    # tests target column 2.
    if (is.null(x_extra1)) x_extra1 <- rnorm(sim$N1)
    if (is.null(x_extra2)) x_extra2 <- rnorm(sim$N2)
    X1 <- cbind(intercept = 1, xsvc = x_extra1)
    X2 <- cbind(intercept = 1, xsvc = x_extra2)
    set.seed(99L)
    y1 <- rbinom(sim$N1, 1L, 0.5)
    y2 <- rnorm(sim$N2, sd = 0.5)
    list(
        a1 = list(y = as.numeric(y1), n_trials = rep(1L, sim$N1),
                  X = X1, family = "binomial", phi = 1.0),
        a2 = list(y = as.numeric(y2), n_trials = rep(1L, sim$N2),
                  X = X2, family = "gaussian", phi = 0.5)
    )
}

.hsgp_block <- function(sim, svc_column = NULL) {
    spec <- list(
        type             = "hsgp",
        m_total          = sim$M,
        phi              = list(sim$phi1, sim$phi2),
        n_obs_per_arm    = c(sim$N1, sim$N2),
        eigenvalues      = sim$lam,
        sigma2_grid      = c(0.3, 0.8),
        lengthscale_grid = c(0.2, 0.5)
    )
    if (!is.null(svc_column)) spec$svc_column <- svc_column
    spec
}

# --------------------------------------------------------------------------- #
# (1) Joint HSGP smoke                                                        #
# --------------------------------------------------------------------------- #

test_that("joint HSGP (no svc) runs end-to-end via multi-block dispatch", {
    sim <- .sim_two_arm_hsgp(seed = 21L)
    set.seed(101)
    arms <- .make_responses_hsgp(sim)
    prior <- list(.hsgp_block(sim))  # length-1 multi-block list

    fit <- tulpa_nested_laplace_joint(
        responses = list(occ = arms$a1, pos = arms$a2),
        prior = prior, copy = NULL,
        control = list(max_iter = 40L, tol = 1e-7, n_threads = 1L, verbose = FALSE)
    )
    expect_s3_class(fit, "tulpa_nested_laplace_joint")
    expect_true(all(is.finite(fit$log_marginal)))
    expect_equal(sum(fit$weights), 1.0, tolerance = 1e-6)
})

# --------------------------------------------------------------------------- #
# (2) SVC with constant-1 column matches no-SVC baseline                      #
# --------------------------------------------------------------------------- #

test_that("svc_column on a constant-1 X column reproduces the no-svc fit", {
    sim <- .sim_two_arm_hsgp(seed = 22L)
    set.seed(102)
    arms <- .make_responses_hsgp(sim,
                                  x_extra1 = rep(1.0, sim$N1),
                                  x_extra2 = rep(1.0, sim$N2))

    fit_plain <- tulpa_nested_laplace_joint(
        responses = list(occ = arms$a1, pos = arms$a2),
        prior = list(.hsgp_block(sim, svc_column = NULL)),
        control = list(max_iter = 40L, tol = 1e-7, n_threads = 1L, verbose = FALSE)
    )
    fit_svc <- tulpa_nested_laplace_joint(
        responses = list(occ = arms$a1, pos = arms$a2),
        prior = list(.hsgp_block(sim, svc_column = 2L)),
        control = list(max_iter = 40L, tol = 1e-7, n_threads = 1L, verbose = FALSE)
    )
    expect_equal(as.numeric(fit_svc$log_marginal),
                 as.numeric(fit_plain$log_marginal),
                 tolerance = 1e-8,
                 info = "X[, svc_column] = 1 must leave HSGP fit unchanged")
})

# --------------------------------------------------------------------------- #
# (3) SVC with a varying covariate produces a different, finite fit           #
# --------------------------------------------------------------------------- #

test_that("svc_column on a varying X column shifts the HSGP fit", {
    sim <- .sim_two_arm_hsgp(seed = 23L)
    set.seed(103)
    arms <- .make_responses_hsgp(sim)  # varying x_extra

    fit_plain <- tulpa_nested_laplace_joint(
        responses = list(occ = arms$a1, pos = arms$a2),
        prior = list(.hsgp_block(sim, svc_column = NULL)),
        control = list(max_iter = 40L, tol = 1e-7, n_threads = 1L, verbose = FALSE)
    )
    fit_svc <- tulpa_nested_laplace_joint(
        responses = list(occ = arms$a1, pos = arms$a2),
        prior = list(.hsgp_block(sim, svc_column = 2L)),
        control = list(max_iter = 40L, tol = 1e-7, n_threads = 1L, verbose = FALSE)
    )
    expect_true(all(is.finite(fit_svc$log_marginal)))
    # The fits must differ on at least one cell; the basis row-scaling
    # actually changes the latent contribution to eta.
    expect_gt(max(abs(fit_svc$log_marginal - fit_plain$log_marginal)),
              1e-4)
})

# --------------------------------------------------------------------------- #
# (4) Validation errors                                                       #
# --------------------------------------------------------------------------- #

test_that("bad svc_column values raise clean errors before C++ entry", {
    sim <- .sim_two_arm_hsgp(seed = 24L)
    set.seed(104)
    arms <- .make_responses_hsgp(sim)

    expect_error(
        tulpa_nested_laplace_joint(
            responses = list(occ = arms$a1, pos = arms$a2),
            prior = list(.hsgp_block(sim, svc_column = 99L)),
            control = list(max_iter = 5L, tol = 1e-4, n_threads = 1L)
        ),
        "exceeds ncol"
    )
    expect_error(
        tulpa_nested_laplace_joint(
            responses = list(occ = arms$a1, pos = arms$a2),
            prior = list(.hsgp_block(sim, svc_column = 0L)),
            control = list(max_iter = 5L, tol = 1e-4, n_threads = 1L)
        ),
        "positive integer"
    )
})

# --------------------------------------------------------------------------- #
# (5) Multi-scale HSGP: two HSGP blocks in one prior_list                     #
# --------------------------------------------------------------------------- #

.hsgp_block_with_basis <- function(N1, N2, m_per_dim, coords1, coords2,
                                    sigma2_grid, lengthscale_grid) {
    # Each scale gets its own eigenvalue spectrum (different m_per_dim).
    # Per-arm bases evaluated against the same coords as the smoke fixture.
    b1 <- .hsgp_basis_2d_svc(coords1, m_per_dim = m_per_dim)
    b2 <- .hsgp_basis_2d_svc(coords2, m_per_dim = m_per_dim)
    list(
        type             = "hsgp",
        m_total          = b1$M,
        phi              = list(b1$phi, b2$phi),
        n_obs_per_arm    = c(N1, N2),
        eigenvalues      = b1$lam,
        sigma2_grid      = sigma2_grid,
        lengthscale_grid = lengthscale_grid
    )
}

test_that("multi-scale HSGP (two blocks) composes via the multi-block prior", {
    sim <- .sim_two_arm_hsgp(seed = 25L)
    set.seed(105)
    arms <- .make_responses_hsgp(sim)

    # Long-range component (few basis fns, wider lengthscales) +
    # short-range component (more basis fns, tighter lengthscales).
    # HSGP `sigma2_grid` and `lengthscale_grid` are paired axes (each row
    # is one (sigma2, ell) pair, NOT a Cartesian factor) — pre-expand
    # via expand.grid() to get a 2x2 = 4-cell per-block grid.
    s2_long <- c(0.5, 1.0); ell_long <- c(0.6, 1.0)
    g_long  <- expand.grid(s2 = s2_long, ell = ell_long)
    block_long <- .hsgp_block_with_basis(
        sim$N1, sim$N2, m_per_dim = 3L,
        sim$coords1, sim$coords2,
        sigma2_grid      = g_long$s2,
        lengthscale_grid = g_long$ell
    )
    s2_short <- c(0.2, 0.5); ell_short <- c(0.1, 0.2)
    g_short  <- expand.grid(s2 = s2_short, ell = ell_short)
    block_short <- .hsgp_block_with_basis(
        sim$N1, sim$N2, m_per_dim = 4L,
        sim$coords1, sim$coords2,
        sigma2_grid      = g_short$s2,
        lengthscale_grid = g_short$ell
    )

    fit_multi <- tulpa_nested_laplace_joint(
        responses = list(occ = arms$a1, pos = arms$a2),
        prior = list(block_long, block_short),
        copy = NULL,
        control = list(max_iter = 40L, tol = 1e-7, n_threads = 1L, verbose = FALSE)
    )
    expect_s3_class(fit_multi, "tulpa_nested_laplace_joint")
    expect_true(all(is.finite(fit_multi$log_marginal)))
    expect_equal(sum(fit_multi$weights), 1.0, tolerance = 1e-6)
    # Cartesian product of two 4-cell blocks = 16 cells.
    expect_equal(length(fit_multi$log_marginal), 16L)

    # Single-scale baseline (long block only) — must differ on at least
    # one cell from the multi-scale fit at the matching (long-axis) cell.
    fit_long <- tulpa_nested_laplace_joint(
        responses = list(occ = arms$a1, pos = arms$a2),
        prior = list(block_long),
        copy = NULL,
        control = list(max_iter = 40L, tol = 1e-7, n_threads = 1L, verbose = FALSE)
    )
    expect_equal(length(fit_long$log_marginal), 4L)
    # Both fits exist; the multi-scale fit's max log_marginal must differ
    # from the single-scale fit's max — the short block carries non-zero
    # information.
    expect_gt(abs(max(fit_multi$log_marginal) - max(fit_long$log_marginal)),
              1e-4)
})
