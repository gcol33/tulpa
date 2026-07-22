# Replicated areal field: spatial(graph, ~ ... || cell, by = level).
#
# `by` builds one independent copy of the whole field per factor level -- the
# field over the block-diagonal Kronecker graph I_L (x) Q -- with the
# hyperparameters shared across levels. It is orthogonal to the bar character
# (| / ||): `by` sets how many replicates exist, the bar sets the covariance
# among the coefficient columns within a replicate.
#
# Coverage, decisive-first:
#   (0) KERNEL INVARIANT (regression guard). The intrinsic ICAR / MCAR kernels
#       must treat each connected component's rank-deficiency separately: a
#       block-diagonal L-component log-prior MUST equal the sum of the L
#       independent single-component log-priors. The pre-fix kernels hardcoded a
#       single component (rank n-1, one global sum-to-zero), so they FAILED this
#       additivity (the (n-1) vs (n-L) normalizer differs by 0.5(L-1)log(tau)).
#   (1) Replication helper: I_L (x) Q + level-offset index; L = 1 is the identity.
#   (2) .spatial_field_blocks carries n_components; L = 1 is byte-identical to no-by.
#   (3) Constructor: `by` stores the column name; |+proper gate holds with by.
#   (4) Parameter recovery: L independent per-level fields sharing sigma, with no
#       cross-level contamination (the centering must be per component).

.chain_adj_by <- function(n_s) {
  adj <- matrix(0, n_s, n_s)
  for (i in seq_len(n_s - 1L)) adj[i, i + 1L] <- adj[i + 1L, i] <- 1
  adj
}

# Block-diagonal from a list of per-component adjacency blocks; the blocks may
# be UNEQUAL in size (a genuine disconnected map is not the equal-size I_L (x) Q
# of a replicate). Components come from the graph, so no count is passed.
.blockdiag_list <- function(adjs) {
  as.matrix(Matrix::bdiag(lapply(adjs, function(a) Matrix::Matrix(a, sparse = TRUE))))
}

.icar_lp1 <- function(x, adj, tau) {
  csr <- tulpa:::adjacency_to_csr_tulpa(adj)
  tulpa:::cpp_test_log_prior_icar(x, nrow(adj), tau,
    csr$row_ptr, csr$col_idx, csr$n_neighbors)
}

# --------------------------------------------------------------------------- #
# (0) Kernel invariant: per-component rank deficiency (the regression guard)   #
# --------------------------------------------------------------------------- #

test_that("ICAR log-prior is additive across connected components (equal and unequal)", {
  set.seed(11)
  # UNEQUAL sizes are the discriminating case: the pre-fix kernel split the field
  # into n_components EQUAL contiguous blocks, so a 5+3 graph pinned {1..4},{5..8}
  # instead of {1..5},{6..8} and broke additivity. Equal sizes are covered too so
  # the replicate path stays byte-identical.
  for (sizes in list(c(7L, 7L), c(5L, 3L), c(6L, 4L, 3L))) {
    adjs    <- lapply(sizes, .chain_adj_by)
    adj_rep <- .blockdiag_list(adjs)
    xs      <- lapply(sizes, rnorm)
    x_full  <- unlist(xs)
    for (tau in c(0.5, 1.7, 3.3)) {
      full  <- .icar_lp1(x_full, adj_rep, tau)
      parts <- sum(mapply(function(a, x) .icar_lp1(x, a, tau), adjs, xs))
      expect_equal(full, parts, tolerance = 1e-9,
                   info = paste("sizes", paste(sizes, collapse = "+"), "tau", tau))
    }
  }
})

test_that("ICAR log-prior is invariant to node relabeling (non-contiguous components)", {
  set.seed(12)
  # A 5+3 disconnected graph, then a permutation that INTERLEAVES the two
  # components so neither is contiguous in the node ordering. The log-prior is a
  # relabeling invariant (edge sums + per-component constant pins), so the
  # permuted value must match -- this is what exercises the arbitrary node-index
  # path rather than the contiguous fast path.
  adj_rep <- .blockdiag_list(list(.chain_adj_by(5L), .chain_adj_by(3L)))
  n <- nrow(adj_rep); x <- rnorm(n)
  perm <- c(1L, 6L, 2L, 7L, 3L, 8L, 4L, 5L)   # comp1 -> {1,3,5,7,8}, comp2 -> {2,4,6}
  adj_perm <- adj_rep[perm, perm]
  x_perm   <- x[perm]
  for (tau in c(0.7, 2.1)) {
    expect_equal(.icar_lp1(x, adj_rep, tau), .icar_lp1(x_perm, adj_perm, tau),
                 tolerance = 1e-10, info = paste("tau", tau))
  }
})

test_that("MCAR log-prior is additive across connected components (unequal)", {
  set.seed(13)
  p     <- 2L
  theta <- c(log(1.1), 0.4, log(0.8))       # log-Cholesky of a 2x2 Sigma
  mcar1 <- function(x, adj) {
    csr <- tulpa:::adjacency_to_csr_tulpa(adj)
    tulpa:::cpp_test_mcar_prior(theta, p, nrow(adj),
      csr$row_ptr, csr$col_idx, csr$n_neighbors, x)$log_prior
  }
  sizes   <- c(5L, 3L)                        # unequal -> discriminates equal-split
  adjs    <- lapply(sizes, .chain_adj_by)
  adj_rep <- .blockdiag_list(adjs)
  # Per-component latents are field-major within their own single-component
  # problem (field a over s cells); the full latent is field-major over all
  # cells (field a = concat of each component's field-a slice).
  xs <- lapply(sizes, function(s) rnorm(p * s))
  fields <- lapply(seq_len(p), function(a)
    unlist(lapply(seq_along(sizes), function(ci) {
      s <- sizes[ci]; xs[[ci]][((a - 1L) * s + 1L):(a * s)]
    })))
  x_full <- unlist(fields)
  full   <- mcar1(x_full, adj_rep)
  parts  <- sum(mapply(function(a, x) mcar1(x, a), adjs, xs))
  expect_equal(full, parts, tolerance = 1e-9)
})

# --------------------------------------------------------------------------- #
# (1) Replication helper                                                      #
# --------------------------------------------------------------------------- #

test_that("tulpa_bar_field_replicate builds I_L (x) Q and the level-offset index", {
  adj <- .chain_adj_by(4L)
  node <- rep(1:4, times = 3L)
  lev  <- rep(c("a", "b", "c"), each = 4L)
  ri <- tulpa_bar_field_replicate(adj, node, lev)

  expect_identical(ri$n_levels, 3L)
  expect_identical(ri$n_nodes, 4L)
  expect_identical(dim(ri$adjacency), c(12L, 12L))
  # Block-diagonal: the (a, b) cross-block is all zero.
  expect_true(all(ri$adjacency[1:4, 5:8] == 0))
  expect_equal(ri$adjacency[1:4, 1:4], adj)
  expect_equal(ri$adjacency[5:8, 5:8], adj)
  # Level offset: node n in level k -> n + (k-1) * n_nodes.
  expect_identical(ri$index, as.integer(node + (rep(0:2, each = 4L)) * 4L))
})

test_that("a single-level by is the identity (byte-identical graph + index)", {
  adj <- .chain_adj_by(5L)
  node <- rep(1:5, times = 2L)
  lev  <- rep("only", 10L)
  ri <- tulpa_bar_field_replicate(adj, node, lev)
  expect_identical(ri$n_levels, 1L)
  expect_equal(ri$adjacency, adj)
  expect_identical(ri$index, as.integer(node))
})

test_that("tulpa_bar_field_replicate validates its inputs", {
  adj <- .chain_adj_by(4L)
  expect_error(tulpa_bar_field_replicate(adj, 1:3, rep("a", 4L)),
               "same length")
  expect_error(tulpa_bar_field_replicate(adj, c(1L, 5L, 2L, 3L), rep("a", 4L)),
               "1-based indices")
  expect_error(tulpa_bar_field_replicate(matrix(0, 2, 3), 1:2, c("a", "b")),
               "square")
})

# --------------------------------------------------------------------------- #
# (2) Block expansion carries n_components; L = 1 is byte-identical            #
# --------------------------------------------------------------------------- #

test_that(".spatial_field_blocks replicates the graph and tags n_components", {
  adj <- .chain_adj_by(6L)
  d <- data.frame(cell = rep(1:6, each = 4L),
                  hab  = rep(c("x", "y"), each = 12L),
                  time = rnorm(24L))
  f <- spatial(graph = adj, formula = ~ 1 || cell, by = hab)
  blks <- tulpa:::.spatial_field_blocks(f, d)
  expect_length(blks, 1L)
  b <- blks[[1L]]
  expect_identical(b$type, "icar")
  expect_identical(b$n_components, 2L)
  expect_identical(b$n_spatial_units, 12L)          # 2 levels x 6 cells
  # The per-observation index is offset into the level's copy.
  idx <- b$spatial_idx[[1L]]
  expect_identical(max(idx), 12L)
  expect_true(all(idx[d$hab == "x"] <= 6L))
  expect_true(all(idx[d$hab == "y"] > 6L))
})

test_that("a single-level by produces blocks byte-identical to no-by", {
  adj <- .chain_adj_by(6L)
  d <- data.frame(cell = rep(1:6, each = 4L), time = rnorm(24L))
  d$one <- "only"
  f_by   <- spatial(graph = adj, formula = ~ 1 + time || cell, by = one)
  f_no   <- spatial(graph = adj, formula = ~ 1 + time || cell)
  expect_identical(tulpa:::.spatial_field_blocks(f_by, d),
                   tulpa:::.spatial_field_blocks(f_no, d))
})

test_that("correlated (|) by builds one mcar block over the replicated graph", {
  adj <- .chain_adj_by(5L)
  d <- data.frame(cell = rep(1:5, each = 6L),
                  hab  = rep(c("x", "y", "z"), each = 10L),
                  time = rnorm(30L))
  f <- spatial(graph = adj, formula = ~ 1 + time | cell, by = hab)
  blks <- tulpa:::.spatial_field_blocks(f, d)
  expect_length(blks, 1L)
  expect_identical(blks[[1L]]$type, "mcar")
  expect_identical(blks[[1L]]$n_components, 3L)
  expect_identical(blks[[1L]]$n_spatial_units, 15L)   # 3 levels x 5 cells
})

# --------------------------------------------------------------------------- #
# (3) Constructor + gates                                                     #
# --------------------------------------------------------------------------- #

test_that("spatial() records the by column (bare or string)", {
  adj <- .chain_adj_by(5L)
  expect_identical(spatial(adj, ~ 1 || cell, by = hab)$by_var, "hab")
  expect_identical(spatial(adj, ~ 1 || cell, by = "hab")$by_var, "hab")
  expect_null(spatial(adj, ~ 1 || cell)$by_var)
})

test_that("the |+proper gate holds with or without by", {
  adj <- .chain_adj_by(5L)
  expect_error(
    spatial(adj, ~ 1 + time | cell, proper = TRUE, by = hab),
    "Correlated proper CAR")
})

test_that("a by column missing from the data errors at fit time", {
  adj <- .chain_adj_by(5L)
  d <- data.frame(cell = rep(1:5, each = 3L))
  f <- spatial(graph = adj, formula = ~ 1 || cell, by = nope)
  expect_error(tulpa:::.spatial_field_blocks(f, d), "not found in the data")
})

# --------------------------------------------------------------------------- #
# (4) Parameter recovery                                                      #
# --------------------------------------------------------------------------- #

test_that("by recovers L independent per-level fields sharing sigma", {
  skip_on_cran()
  set.seed(3)

  n_s <- 14L
  L   <- 2L
  adj <- .chain_adj_by(n_s)

  # Two independent ICAR fields (smooth chains), each centred + scaled to the
  # SAME sigma -- the replicate model.
  mk_field <- function() {
    u <- cumsum(rnorm(n_s)); (u - mean(u)) / sd(u)
  }
  u1 <- mk_field(); u2 <- mk_field()
  sig_u <- 1.1; b0 <- 0.3

  n_per <- 30L
  grid <- expand.grid(cell = seq_len(n_s), level = seq_len(L),
                      rep = seq_len(n_per))
  cell <- grid$cell; level <- grid$level
  N <- nrow(grid)
  U <- cbind(u1, u2)
  eta <- b0 + sig_u * U[cbind(cell, level)]
  y <- rnorm(N, eta, 0.4)
  d <- data.frame(y = y, cell = cell, level = factor(level))

  fit <- suppressWarnings(tulpa(
    y ~ spatial(graph = adj, formula = ~ 1 || cell, by = level),
    data = d, family = "gaussian", mode = "laplace",
    control = list(n_draws = 400L)))

  expect_s3_class(fit, "tulpa_fit")
  fld <- fit$spatial_fields[["cell.Intercept"]]$mean
  expect_length(fld, L * n_s)
  u1_hat <- fld[1:n_s]
  u2_hat <- fld[(n_s + 1):(2 * n_s)]

  # Each level's field is recovered ...
  expect_gt(cor(u1_hat, u1), 0.7)
  expect_gt(cor(u2_hat, u2), 0.7)
  # ... and the two levels do NOT contaminate each other (a single global
  # sum-to-zero centering across both copies would couple them).
  expect_lt(abs(cor(u1_hat, u2)), 0.55)
  expect_lt(abs(cor(u2_hat, u1)), 0.55)

  # Fixed-effect intercept recovered.
  expect_lt(abs(coef(fit)[["(Intercept)"]] - b0), 0.2)

  # The shared sigma is recovered in a sensible band (the rank-deficiency
  # normalizer must use (n - L); a wrong (n - 1) biases it).
  sig_hat <- fit$spatial_field_hypers[["cell.Intercept"]]$sigma[2L]
  expect_gt(sig_hat, 0.6 * sig_u)
  expect_lt(sig_hat, 1.8 * sig_u)
})
