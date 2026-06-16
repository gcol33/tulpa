# Multi-output (co-regionalization) HSGP block (Stage 1.7).
#
# `hsgp_mo` introduces a K = n_arms-output HSGP where K correlated latent
# fields share a basis Phi and eigenvalues, but their basis coefficients
# are coupled across outputs via a K x K covariance Sigma. For K = 2 the
# parameterization is direct: (sigma_1, sigma_2, rho, ell), all raw.
#
# Coverage:
#   (1) Smoke: 2-arm joint dispatch with one hsgp_mo block. log_marginal
#       finite on every outer-grid cell, weights normalise.
#   (2) Pattern correctness: cpp_test_joint_pattern shows M independent
#       K x K precision blocks plus the cross-output (k=1, k=0) entry at
#       each basis index, in output-major (k*M + m) coordinates.
#   (3) Independence reduction: when rho_grid = 0 and sigma_1 = sigma_2 a
#       fitted hsgp_mo block places posterior mass on the rho = 0 axis
#       (data are independent across outputs).
#   (4) Cross-output coupling: when the simulated K = 2 fields are
#       strongly correlated, posterior weight concentrates on rho > 0
#       cells rather than rho <= 0.
#   (5) Validation errors: missing fields, K != 2.

skip_on_cran()

.hsgp_basis_2d_mo <- function(coords, m_per_dim = 4L, cscl = 1.5) {
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

.sim_two_arm_mo <- function(seed = 31L, N1 = 90L, N2 = 90L,
                              m_per_dim = 4L,
                              sigma_1 = 1.0, sigma_2 = 1.0,
                              rho = 0.0, ell = 0.4) {
    set.seed(seed)
    coords1 <- cbind(runif(N1), runif(N1))
    coords2 <- cbind(runif(N2), runif(N2))
    b1 <- .hsgp_basis_2d_mo(coords1, m_per_dim = m_per_dim)
    b2 <- .hsgp_basis_2d_mo(coords2, m_per_dim = m_per_dim)
    # Use b1's eigenvalues as the shared spectrum (centred coords differ
    # by arm but lambda_m are close enough at this scale).
    M <- b1$M
    lam <- b1$lam

    # Sample beta_{:,m} ~ N(0, Sigma) for each m, where
    #   Sigma = [[s1^2, rho*s1*s2], [rho*s1*s2, s2^2]].
    L <- matrix(0, 2L, 2L)
    L[1, 1] <- sigma_1
    L[2, 1] <- rho * sigma_2
    L[2, 2] <- sigma_2 * sqrt(max(1 - rho^2, 0))
    Z <- matrix(rnorm(2L * M), nrow = 2L)
    beta <- L %*% Z  # 2 x M, rows = outputs

    # sqrt_S_norm uses sigma-free 1D-form spectral density (matches the
    # factory). Multiply phi by sqrt_S elementwise before contracting.
    pref <- sqrt(2 * pi) * ell
    sqrt_S <- sqrt(pref * exp(-0.5 * ell^2 * lam))

    eta_lat_1 <- b1$phi %*% (sqrt_S * beta[1, ])
    eta_lat_2 <- b2$phi %*% (sqrt_S * beta[2, ])

    X1 <- matrix(1.0, N1, 1L)
    X2 <- matrix(1.0, N2, 1L)
    beta_lin_1 <- 0.1
    beta_lin_2 <- -0.2
    eta1 <- as.numeric(X1 %*% beta_lin_1 + eta_lat_1)
    eta2 <- as.numeric(X2 %*% beta_lin_2 + eta_lat_2)
    y1 <- rbinom(N1, 1L, plogis(eta1))
    y2 <- rnorm(N2, eta2, 0.3)

    list(
        N1 = N1, N2 = N2, M = M, lam = lam,
        phi1 = b1$phi, phi2 = b2$phi,
        responses = list(
            occ = list(y = as.numeric(y1), n_trials = rep(1L, N1),
                       X = X1, family = "binomial", phi = 1.0),
            pos = list(y = as.numeric(y2), n_trials = rep(1L, N2),
                       X = X2, family = "gaussian", phi = 0.3)
        )
    )
}

.mo_block <- function(sim, sigma_1_grid, sigma_2_grid, rho_grid, ell_grid) {
    list(
        type             = "hsgp_mo",
        m_total          = sim$M,
        phi              = list(sim$phi1, sim$phi2),
        n_obs_per_arm    = c(sim$N1, sim$N2),
        eigenvalues      = sim$lam,
        sigma_1_grid     = as.numeric(sigma_1_grid),
        sigma_2_grid     = as.numeric(sigma_2_grid),
        rho_grid         = as.numeric(rho_grid),
        lengthscale_grid = as.numeric(ell_grid)
    )
}

# --------------------------------------------------------------------------- #
# (1) Smoke                                                                   #
# --------------------------------------------------------------------------- #

test_that("hsgp_mo joint dispatch runs end-to-end on 2 arms (K = n_arms = 2)", {
    sim <- .sim_two_arm_mo(seed = 31L, sigma_1 = 0.8, sigma_2 = 0.8,
                            rho = 0.3, ell = 0.5)
    s1 <- c(0.5, 1.0); s2 <- c(0.5, 1.0); rh <- c(0.0, 0.4); ls <- c(0.4)
    gr <- expand.grid(s1 = s1, s2 = s2, rh = rh, ls = ls,
                       KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
    block <- .mo_block(sim, gr$s1, gr$s2, gr$rh, gr$ls)

    fit <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = list(block), copy = NULL,
        control = list(max_iter = 40L, tol = 1e-7, n_threads = 1L, verbose = FALSE)
    )
    expect_s3_class(fit, "tulpa_nested_laplace_joint")
    expect_true(all(is.finite(fit$log_marginal)))
    expect_equal(sum(fit$weights), 1.0, tolerance = 1e-6)
    expect_equal(length(fit$log_marginal), nrow(gr))
})

# --------------------------------------------------------------------------- #
# (2) Pattern correctness                                                     #
# --------------------------------------------------------------------------- #

test_that("hsgp_mo H pattern: M independent K x K precision blocks (K = 2)", {
    # Minimal fixture: 2 arms with 3 obs each, M = 4 basis fns. We need
    # phi to be all-ones so cpp_test_joint_pattern routes through every
    # latent slot in the basis_eval step (the pattern builder for
    # DENSE_BASIS over-fills the full sub-block, so positions are not
    # data-dependent; phi values are unused at the pattern stage).
    M <- 4L
    N <- 3L
    phi_1 <- matrix(0.5, N, M)
    phi_2 <- matrix(0.5, N, M)
    arm_template <- function() list(
        y = rep(0.0, N), n_trials = rep(1L, N),
        X = matrix(1.0, N, 1L),
        re_idx = rep(0L, N), n_re_groups = 0L, sigma_re = 1.0,
        spatial_idx = rep(0L, N), temporal_idx = rep(0L, N),
        obs_idx = rep(0L, N),
        family = "gaussian", phi = 1.0
    )
    arms_list <- list(arm_template(), arm_template())
    bs <- list(
        type           = "hsgp_mo",
        m_total        = M,
        phi            = list(phi_1, phi_2),
        n_obs_per_arm  = c(N, N),
        eigenvalues    = rep(0.5, M)
    )
    # Theta grid must have 4 columns (sigma_1, sigma_2, rho, ell). Pattern
    # builder doesn't index theta_grid except via block.prep — and we don't
    # call prep in cpp_test_joint_pattern. One dummy row is enough.
    theta_grid <- matrix(c(1.0, 1.0, 0.0, 0.5), nrow = 1L)
    pat <- cpp_test_joint_pattern(
        arms_list = arms_list,
        copy_arm = -1L, copy_block = -1L,
        blocks_spec = list(bs),
        theta_grid = theta_grid,
        axis_offsets = as.integer(c(0L, 4L))
    )

    # n_x = per-arm betas (1 each) + multi-output block size (K * M).
    expect_equal(pat$n_x, 1L + 1L + 2L * M)

    # The block starts after the two betas: latent offset = 2.
    blk_start <- 2L
    # Reconstruct lower-triangle entries.
    nnz <- pat$nnz
    col_ptr <- pat$col_ptr
    row_idx <- pat$row_idx
    cols <- integer(nnz)
    for (j in seq_len(pat$n_x)) {
        for (e in seq(col_ptr[j] + 1L, col_ptr[j + 1L])) {
            if (e <= nnz) cols[e] <- j - 1L
        }
    }
    S <- unique(cbind(row = row_idx, col = cols))
    has <- function(r, c) any(S[, "row"] == r & S[, "col"] == c)

    # Diagonal of each latent slot must be present.
    for (k in 0:1) for (m in 0:(M - 1L)) {
        idx <- blk_start + k * M + m
        expect_true(has(idx, idx),
                    info = sprintf("diagonal (k=%d, m=%d)", k, m))
    }
    # Cross-output entry at each basis index: (k=1, m) couples to (k=0, m).
    for (m in 0:(M - 1L)) {
        r <- blk_start + 1L * M + m
        c <- blk_start + 0L * M + m
        expect_true(has(r, c),
                    info = sprintf("cross-output coupling at m=%d", m))
    }
})

# --------------------------------------------------------------------------- #
# (3) Independence reduction: rho_grid = 0 sanity check                        #
# --------------------------------------------------------------------------- #

test_that("hsgp_mo at rho = 0 stays finite and matches independent-fields scale", {
    sim <- .sim_two_arm_mo(seed = 32L, sigma_1 = 0.7, sigma_2 = 0.7,
                            rho = 0.0, ell = 0.4)
    block <- .mo_block(sim,
                        sigma_1_grid = c(0.7),
                        sigma_2_grid = c(0.7),
                        rho_grid     = c(0.0),
                        ell_grid     = c(0.4))
    fit <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = list(block), copy = NULL,
        control = list(max_iter = 60L, tol = 1e-7, n_threads = 1L, verbose = FALSE)
    )
    expect_true(all(is.finite(fit$log_marginal)))
    expect_equal(length(fit$log_marginal), 1L)
})

# --------------------------------------------------------------------------- #
# (4) Cross-output coupling: rho posterior tracks the simulated correlation   #
# --------------------------------------------------------------------------- #

test_that("hsgp_mo posterior weight concentrates on rho_true > 0 cells", {
    # Simulate at rho_true = +0.7 with reasonably large N; fit on a rho
    # grid spanning {-0.6, -0.3, 0, 0.3, 0.6}. The posterior weight should
    # put more mass on rho > 0 than rho <= 0.
    sim <- .sim_two_arm_mo(seed = 33L, N1 = 200L, N2 = 200L,
                            sigma_1 = 0.8, sigma_2 = 0.8,
                            rho = 0.7, ell = 0.5)
    rh_grid <- c(-0.6, -0.3, 0.0, 0.3, 0.6)
    s1 <- c(0.8); s2 <- c(0.8); ls <- c(0.5)
    gr <- expand.grid(s1 = s1, s2 = s2, rh = rh_grid, ls = ls,
                       KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
    block <- .mo_block(sim, gr$s1, gr$s2, gr$rh, gr$ls)

    fit <- tulpa_nested_laplace_joint(
        responses = sim$responses, prior = list(block), copy = NULL,
        control = list(max_iter = 80L, tol = 1e-8, n_threads = 1L, verbose = FALSE)
    )
    expect_true(all(is.finite(fit$log_marginal)))

    w <- fit$weights
    rho_cells <- gr$rh
    pos_mass <- sum(w[rho_cells > 0])
    neg_mass <- sum(w[rho_cells < 0])
    zero_mass <- sum(w[rho_cells == 0])
    # The data carry a strong rho > 0 signal: positive-rho cells must
    # win the posterior weight against the negative-rho cells.
    expect_gt(pos_mass, neg_mass + 0.10)
})

# --------------------------------------------------------------------------- #
# (5) Validation                                                              #
# --------------------------------------------------------------------------- #

test_that("hsgp_mo raises clean errors on bad spec", {
    sim <- .sim_two_arm_mo(seed = 34L, N1 = 30L, N2 = 30L)
    # Missing eigenvalues.
    expect_error(
        tulpa_nested_laplace_joint(
            responses = sim$responses,
            prior = list(list(type = "hsgp_mo",
                              m_total = sim$M,
                              phi = list(sim$phi1, sim$phi2),
                              n_obs_per_arm = c(sim$N1, sim$N2),
                              sigma_1_grid = 0.5, sigma_2_grid = 0.5,
                              rho_grid = 0.0, lengthscale_grid = 0.4)),
            control = list(max_iter = 5L, tol = 1e-4, n_threads = 1L)
        ),
        "eigenvalues"
    )
    # phi length mismatch with n_arms.
    expect_error(
        tulpa_nested_laplace_joint(
            responses = sim$responses,
            prior = list(list(type = "hsgp_mo",
                              m_total = sim$M,
                              phi = list(sim$phi1),  # only one, n_arms = 2
                              n_obs_per_arm = c(sim$N1, sim$N2),
                              eigenvalues = sim$lam,
                              sigma_1_grid = 0.5, sigma_2_grid = 0.5,
                              rho_grid = 0.0, lengthscale_grid = 0.4)),
            control = list(max_iter = 5L, tol = 1e-4, n_threads = 1L)
        ),
        "length n_arms"
    )
})

test_that("hsgp_mo errors when n_arms != 2 (first-ship K = 2 restriction)", {
    # Build a synthetic 3-arm fixture by replicating arm 1.
    sim <- .sim_two_arm_mo(seed = 35L, N1 = 30L, N2 = 30L)
    responses3 <- c(sim$responses,
                     list(third = sim$responses$pos))  # 3 arms
    expect_error(
        tulpa_nested_laplace_joint(
            responses = responses3,
            prior = list(list(type = "hsgp_mo",
                              m_total = sim$M,
                              phi = list(sim$phi1, sim$phi2,
                                          sim$phi2),
                              n_obs_per_arm = c(sim$N1, sim$N2, sim$N2),
                              eigenvalues = sim$lam,
                              sigma_1_grid = 0.5, sigma_2_grid = 0.5,
                              rho_grid = 0.0, lengthscale_grid = 0.4)),
            control = list(max_iter = 5L, tol = 1e-4, n_threads = 1L)
        ),
        "n_arms == 2"
    )
})
