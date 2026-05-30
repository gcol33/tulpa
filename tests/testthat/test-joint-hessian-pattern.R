# Direct C++ unit tests for build_joint_hessian_pattern (Stage 1.5a).
#
# Exercises the LatentBlock pattern enumerator without running any Newton
# iteration. cpp_test_joint_pattern returns the joint Hessian sparsity
# pattern as a CSC triple (col_ptr, row_idx, n_x); these tests reconstruct
# the lower-triangle pattern matrix and assert exact equality against
# hand-derived reference patterns for each prior shape.
#
# The pattern builder normalises (row, col) to lower triangle (row >= col),
# so every assertion below works on the lower triangle.

skip_on_cran()

.pattern_to_lower_set <- function(pat) {
    # Convert CSC (col_ptr, row_idx) to a set of (row, col) integer pairs
    # in the lower triangle. Returns a unique sorted matrix.
    n <- pat$n_x
    nnz <- pat$nnz
    col_ptr <- pat$col_ptr
    row_idx <- pat$row_idx
    cols <- integer(nnz)
    for (j in seq_len(n)) {
        for (e in seq(col_ptr[j] + 1L, col_ptr[j + 1L])) {
            if (e <= nnz) cols[e] <- j - 1L
        }
    }
    M <- cbind(row = row_idx, col = cols)
    M <- unique(M)
    M[order(M[, "col"], M[, "row"]), , drop = FALSE]
}

.has_entry <- function(pat_set, row, col) {
    # Lower triangle: caller passes (r, c) with r >= c.
    any(pat_set[, "row"] == row & pat_set[, "col"] == col)
}

.chain_adj <- function(n_s) {
    nbr <- vector("list", n_s)
    nbr[[1]] <- 2L
    if (n_s >= 2L) nbr[[n_s]] <- n_s - 1L
    if (n_s >= 3L) for (s in 2:(n_s - 1L)) nbr[[s]] <- c(s - 1L, s + 1L)
    list(adj_row_ptr = as.integer(c(0L, cumsum(vapply(nbr, length, integer(1))))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(vapply(nbr, length, integer(1))),
         n_spatial_units = n_s)
}

.single_arm <- function(N = 4L, p = 1L, spatial_idx = NULL,
                         temporal_idx = NULL, obs_idx = NULL) {
    list(
        y = rep(0.0, N), n_trials = rep(1L, N),
        X = matrix(1.0, N, p),
        re_idx = rep(0L, N), n_re_groups = 0L, sigma_re = 1.0,
        spatial_idx = if (!is.null(spatial_idx)) as.integer(spatial_idx)
                      else rep(0L, N),
        temporal_idx = if (!is.null(temporal_idx)) as.integer(temporal_idx)
                       else rep(0L, N),
        obs_idx = if (!is.null(obs_idx)) as.integer(obs_idx) else rep(0L, N),
        family = "gaussian", phi = 1.0
    )
}


test_that("ICAR pattern: chain adjacency + beta x spatial cross", {
    n_s <- 5L
    adj <- .chain_adj(n_s)
    # 3 obs mapping to sites 1, 3, 5 (only those should appear as β-cross
    # entries — sites 2 and 4 are unobserved but their priors still create
    # adjacency entries).
    arm <- .single_arm(N = 3L, p = 1L, spatial_idx = c(1, 3, 5))
    bs <- c(list(type = "icar",
                 spatial_idx = list(c(1L, 3L, 5L))),
            adj)
    pat <- cpp_test_joint_pattern(
        arms_list = list(arm),
        copy_arms = -1L, copy_blocks = -1L,
        blocks_spec = list(bs),
        theta_grid = matrix(0.5, nrow = 1, ncol = 1),
        axis_offsets = as.integer(c(0, 1))
    )
    expect_equal(pat$n_x, 1L + n_s)  # β + n_s

    S <- .pattern_to_lower_set(pat)

    # Beta diagonal: (0, 0).
    expect_true(.has_entry(S, 0L, 0L))

    # Latent diagonal: (1+s, 1+s) for all s.
    for (s in 0:(n_s - 1L)) {
        expect_true(.has_entry(S, 1L + s, 1L + s))
    }

    # Chain adjacency: (1+i, 1+i-1) for i = 1..n_s-1.
    for (i in 1:(n_s - 1L)) {
        expect_true(.has_entry(S, 1L + i, 1L + i - 1L))
    }

    # Beta x spatial cross: only for observed sites {0, 2, 4} (1-based 1,3,5).
    for (s in c(0L, 2L, 4L)) {
        expect_true(.has_entry(S, 1L + s, 0L))
    }
    # Sites 2 and 4 (1-based) unobserved -> no β-cross entries.
    # site index 1 (1-based 2) at row 1+1=2, col 0.
    expect_false(.has_entry(S, 1L + 1L, 0L))
    expect_false(.has_entry(S, 1L + 3L, 0L))
})


test_that("BYM2 pattern: ICAR on phi block + diagonal-only on theta block", {
    n_s <- 4L
    adj <- .chain_adj(n_s)
    arm <- .single_arm(N = 2L, p = 1L, spatial_idx = c(1, 3))
    bs <- c(list(type = "bym2",
                 spatial_idx = list(c(1L, 3L)),
                 scale_factor = 1.0),
            adj)
    # BYM2 axes: (sigma, rho).
    pat <- cpp_test_joint_pattern(
        arms_list = list(arm),
        copy_arms = -1L, copy_blocks = -1L,
        blocks_spec = list(bs),
        theta_grid = matrix(c(0.5, 0.7), nrow = 1, ncol = 2),
        axis_offsets = as.integer(c(0, 2))
    )
    expect_equal(pat$n_x, 1L + 2L * n_s)  # β + phi + theta
    S <- .pattern_to_lower_set(pat)

    # Layout: 0 = beta, 1..n_s = phi, n_s+1..2*n_s = theta.
    phi_start <- 1L
    theta_start <- 1L + n_s

    # ICAR adjacency on the phi block.
    for (i in 1:(n_s - 1L)) {
        expect_true(.has_entry(S, phi_start + i, phi_start + i - 1L))
    }
    # Theta block: NO adjacency, just diagonal.
    for (s in 0:(n_s - 1L)) {
        expect_true(.has_entry(S, theta_start + s, theta_start + s))
    }
    # No (theta_i, theta_{i-1}) entries — IID prior has no off-diagonal.
    for (i in 1:(n_s - 1L)) {
        expect_false(.has_entry(S, theta_start + i, theta_start + i - 1L))
    }

    # Same-obs phi/theta coupling: every observed site contributes
    # a (theta_s, phi_s) cross. obs map to sites 1 and 3 (0-based 0, 2).
    for (s in c(0L, 2L)) {
        # lower triangle: hi = max(phi, theta), lo = min
        hi <- max(phi_start + s, theta_start + s)
        lo <- min(phi_start + s, theta_start + s)
        expect_true(.has_entry(S, hi, lo))
    }
})


test_that("RW1 pattern: tridiagonal (t, t-1) on the temporal block", {
    n_t <- 6L
    arm <- .single_arm(N = 3L, p = 1L, temporal_idx = c(1, 3, 6))
    bs <- list(type = "rw1",
               temporal_idx = list(c(1L, 3L, 6L)),
               n_times = n_t,
               cyclic = FALSE)
    pat <- cpp_test_joint_pattern(
        arms_list = list(arm),
        copy_arms = -1L, copy_blocks = -1L,
        blocks_spec = list(bs),
        theta_grid = matrix(0.5, nrow = 1, ncol = 1),
        axis_offsets = as.integer(c(0, 1))
    )
    expect_equal(pat$n_x, 1L + n_t)
    S <- .pattern_to_lower_set(pat)

    # Tridiagonal: every (t, t-1) for t in 1..n_t-1.
    for (t in 1:(n_t - 1L)) {
        expect_true(.has_entry(S, 1L + t, 1L + t - 1L))
    }
    # Diagonal: every (t, t).
    for (t in 0:(n_t - 1L)) {
        expect_true(.has_entry(S, 1L + t, 1L + t))
    }
    # No (t, t-2) entries — RW1 is strictly first-order.
    for (t in 2:(n_t - 1L)) {
        expect_false(.has_entry(S, 1L + t, 1L + t - 2L))
    }
})


test_that("RW2 pattern: pentadiagonal (t, t-1) and (t, t-2)", {
    n_t <- 5L
    arm <- .single_arm(N = 2L, p = 1L, temporal_idx = c(1, 5))
    bs <- list(type = "rw2",
               temporal_idx = list(c(1L, 5L)),
               n_times = n_t)
    pat <- cpp_test_joint_pattern(
        arms_list = list(arm),
        copy_arms = -1L, copy_blocks = -1L,
        blocks_spec = list(bs),
        theta_grid = matrix(0.5, nrow = 1, ncol = 1),
        axis_offsets = as.integer(c(0, 1))
    )
    S <- .pattern_to_lower_set(pat)

    # RW2 generates entries at lags 1 and 2.
    for (t in 1:(n_t - 1L)) {
        expect_true(.has_entry(S, 1L + t, 1L + t - 1L))
    }
    for (t in 2:(n_t - 1L)) {
        expect_true(.has_entry(S, 1L + t, 1L + t - 2L))
    }
    # No lag-3 entries.
    if (n_t >= 4L) {
        for (t in 3:(n_t - 1L)) {
            expect_false(.has_entry(S, 1L + t, 1L + t - 3L))
        }
    }
})


test_that("AR1 pattern: tridiagonal Q", {
    n_t <- 5L
    arm <- .single_arm(N = 2L, p = 1L, temporal_idx = c(1, 4))
    bs <- list(type = "ar1",
               temporal_idx = list(c(1L, 4L)),
               n_times = n_t)
    # AR1 axes: (sigma_t, rho).
    pat <- cpp_test_joint_pattern(
        arms_list = list(arm),
        copy_arms = -1L, copy_blocks = -1L,
        blocks_spec = list(bs),
        theta_grid = matrix(c(0.5, 0.7), nrow = 1, ncol = 2),
        axis_offsets = as.integer(c(0, 2))
    )
    S <- .pattern_to_lower_set(pat)
    for (t in 1:(n_t - 1L)) {
        expect_true(.has_entry(S, 1L + t, 1L + t - 1L))
    }
    if (n_t >= 3L) {
        for (t in 2:(n_t - 1L)) {
            # AR1 is tridiagonal: no (t, t-2)
            expect_false(.has_entry(S, 1L + t, 1L + t - 2L))
        }
    }
})


test_that("IID pattern: diagonal only, no off-diagonal entries", {
    n_u <- 4L
    arm <- .single_arm(N = 3L, p = 1L, obs_idx = c(1, 2, 4))
    bs <- list(type = "iid",
               obs_idx = list(c(1L, 2L, 4L)),
               n_units = n_u)
    pat <- cpp_test_joint_pattern(
        arms_list = list(arm),
        copy_arms = -1L, copy_blocks = -1L,
        blocks_spec = list(bs),
        theta_grid = matrix(0.5, nrow = 1, ncol = 1),
        axis_offsets = as.integer(c(0, 1))
    )
    S <- .pattern_to_lower_set(pat)

    # IID diagonal must be there.
    for (u in 0:(n_u - 1L)) {
        expect_true(.has_entry(S, 1L + u, 1L + u))
    }
    # No off-diagonal latent-block entries from IID prior. (Diagonal already
    # checked.) Restrict the lower-triangle scan to the latent block region.
    latent_lo <- 1L
    latent_hi <- 1L + n_u - 1L
    off_latent <- S[S[, "row"] >= latent_lo & S[, "col"] >= latent_lo &
                    S[, "row"] != S[, "col"], , drop = FALSE]
    # The only off-diagonal entries on the latent block should come from
    # observed same-site coupling — but IID has one DOF per obs, no within-
    # obs cross. So this should be empty.
    expect_equal(nrow(off_latent), 0L)
})


test_that("ICAR x RW1 cross-block pattern: only for obs hitting both", {
    n_s <- 3L
    n_t <- 4L
    adj <- .chain_adj(n_s)
    # 2 obs with both spatial_idx and temporal_idx.
    arm <- .single_arm(N = 2L, p = 1L,
                        spatial_idx = c(1, 3),
                        temporal_idx = c(2, 4))
    bs_icar <- c(list(type = "icar",
                       spatial_idx = list(c(1L, 3L))),
                  adj)
    bs_rw1 <- list(type = "rw1",
                    temporal_idx = list(c(2L, 4L)),
                    n_times = n_t,
                    cyclic = FALSE)
    pat <- cpp_test_joint_pattern(
        arms_list = list(arm),
        copy_arms = -1L, copy_blocks = -1L,
        blocks_spec = list(bs_icar, bs_rw1),
        theta_grid = matrix(c(0.5, 0.5), nrow = 1, ncol = 2),
        axis_offsets = as.integer(c(0, 1, 2))
    )
    expect_equal(pat$n_x, 1L + n_s + n_t)
    S <- .pattern_to_lower_set(pat)

    icar_start <- 1L
    rw1_start <- 1L + n_s

    # Cross-block entries: obs 1 -> (icar_start + 0, rw1_start + 1) and
    # obs 2 -> (icar_start + 2, rw1_start + 3). Lower triangle: hi >= lo.
    for (k in seq_along(c(1L, 3L))) {
        s <- c(1L, 3L)[k] - 1L  # 0-based
        t <- c(2L, 4L)[k] - 1L
        hi <- max(icar_start + s, rw1_start + t)
        lo <- min(icar_start + s, rw1_start + t)
        expect_true(.has_entry(S, hi, lo),
                    info = sprintf("expected ICAR site %d x RW1 time %d", s, t))
    }
    # Cross entries for obs that don't co-occur should NOT be present.
    # E.g. (icar site 1, rw1 time 0): no obs hits both -> not in pattern.
    expect_false(.has_entry(S, icar_start + 1L, rw1_start + 0L))
})
