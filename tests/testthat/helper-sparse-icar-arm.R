# The sparse single-arm ICAR fixture whose field-SD posterior rails its own
# grid. Shared rather than copied: `test-nested-laplace-joint-auto-grid.R`
# needs it to exercise the recenter, and `test-outer-grid-collapse-reporting.R`
# needs the SAME collapse to check what the reporting doors say about it, and a
# second copy is how one of them stops collapsing and silently stops testing.

.chain_adj_ag <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn),
         n_spatial_units = n_s)
}

# A sparse, strongly separated occurrence pattern -- a few units almost
# always positive, the rest almost always negative -- the "sparse,
# weakly-identified species" regime gcol33/tulpa#289 targets: the field-SD
# posterior wants to sit well past the old fixed ceiling of 3.0.
.sparse_icar_arm <- function(n_s = 20L, n_per = 6L, seed = 11) {
    set.seed(seed)
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    base_p <- rep(0.02, n_s); base_p[1:5] <- 0.95
    y <- rbinom(length(spatial_idx), 1, base_p[spatial_idx])
    X <- cbind(1, rnorm(length(y), 0, 0.05))
    list(
        arm = list(y = as.numeric(y), n_trials = rep(1L, length(y)),
                  X = X, spatial_idx = as.integer(spatial_idx),
                  re_idx = rep(0, length(y)), n_re_groups = 0L, sigma_re = 1.0,
                  family = "binomial", phi = 1.0),
        adj = .chain_adj_ag(n_s)
    )
}

# The pinned-axis prior that keeps that fixture collapsed: an explicit
# `sigma_grid` is a user override the recenter never touches, so the fit stays
# `collapsed_edge` on the sigma axis.
.sparse_icar_pinned_prior <- function(sim, sigma_grid = c(0.1, 0.5, 1, 2, 3)) {
    list(type = "icar", n_spatial_units = sim$adj$n_spatial_units,
         adj_row_ptr = sim$adj$adj_row_ptr,
         adj_col_idx = sim$adj$adj_col_idx,
         n_neighbors = sim$adj$n_neighbors,
         sigma_grid = sigma_grid)
}
