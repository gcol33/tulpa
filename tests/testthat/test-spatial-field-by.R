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

# Block-diagonal I_L (x) Q as a plain matrix.
.blockdiag_by <- function(adj, L) {
  as.matrix(Matrix::bdiag(rep(list(Matrix::Matrix(adj, sparse = TRUE)), L)))
}

# --------------------------------------------------------------------------- #
# (0) Kernel invariant: per-component rank deficiency (the regression guard)   #
# --------------------------------------------------------------------------- #

test_that("ICAR log-prior is additive across connected components", {
  set.seed(11)
  n_base <- 7L
  adj_base <- .chain_adj_by(n_base)
  csr_base <- tulpa:::adjacency_to_csr_tulpa(adj_base)

  adj_rep <- .blockdiag_by(adj_base, 2L)
  csr_rep <- tulpa:::adjacency_to_csr_tulpa(adj_rep)

  x1 <- rnorm(n_base); x2 <- rnorm(n_base)
  x_full <- c(x1, x2)

  # tau != 2*pi so the normalizer term is non-trivial: the old (n-1) kernel
  # would be off by exactly 0.5*log(tau / (2*pi)) here.
  for (tau in c(0.5, 1.7, 3.3)) {
    full <- tulpa:::cpp_test_log_prior_icar(
      x_full, 2L * n_base, tau,
      csr_rep$row_ptr, csr_rep$col_idx, csr_rep$n_neighbors, n_components = 2L)
    parts <-
      tulpa:::cpp_test_log_prior_icar(x1, n_base, tau,
        csr_base$row_ptr, csr_base$col_idx, csr_base$n_neighbors, 1L) +
      tulpa:::cpp_test_log_prior_icar(x2, n_base, tau,
        csr_base$row_ptr, csr_base$col_idx, csr_base$n_neighbors, 1L)
    expect_equal(full, parts, tolerance = 1e-9,
                 info = paste("tau =", tau))
  }

  # And the component-aware value MUST differ from the single-component
  # treatment of the same 2n graph (the bug). With each component summing to
  # zero the sum-to-zero penalty is identical both ways, so the gap is purely
  # the rank-deficiency normalizer: (n - 2) vs (n - 1) => -0.5 log(tau / 2pi),
  # which is non-zero for any tau != 2*pi.
  tau <- 1.7
  xc_full <- c(x1 - mean(x1), x2 - mean(x2))
  comp_aware <- tulpa:::cpp_test_log_prior_icar(
    xc_full, 2L * n_base, tau,
    csr_rep$row_ptr, csr_rep$col_idx, csr_rep$n_neighbors, n_components = 2L)
  one_comp <- tulpa:::cpp_test_log_prior_icar(
    xc_full, 2L * n_base, tau,
    csr_rep$row_ptr, csr_rep$col_idx, csr_rep$n_neighbors, n_components = 1L)
  expect_equal(comp_aware - one_comp, -0.5 * log(tau / (2 * pi)),
               tolerance = 1e-9)
})

test_that("ICAR n_components = 1 is byte-identical to the historical path", {
  set.seed(12)
  n <- 9L
  adj <- .chain_adj_by(n)
  csr <- tulpa:::adjacency_to_csr_tulpa(adj)
  x <- rnorm(n)
  a <- tulpa:::cpp_test_log_prior_icar(x, n, 1.3,
         csr$row_ptr, csr$col_idx, csr$n_neighbors)            # default 1
  b <- tulpa:::cpp_test_log_prior_icar(x, n, 1.3,
         csr$row_ptr, csr$col_idx, csr$n_neighbors, 1L)
  expect_identical(a, b)
})

test_that("MCAR log-prior is additive across connected components", {
  set.seed(13)
  n_base <- 6L
  p <- 2L
  adj_base <- .chain_adj_by(n_base)
  csr_base <- tulpa:::adjacency_to_csr_tulpa(adj_base)
  adj_rep <- .blockdiag_by(adj_base, 2L)
  csr_rep <- tulpa:::adjacency_to_csr_tulpa(adj_rep)

  # log-Cholesky coords of a 2x2 Sigma (log-diag + strict lower).
  theta <- c(log(1.1), 0.4, log(0.8))

  # Per-component latent: field-major within each single-component problem
  # (field0 over n_base, field1 over n_base).
  x_c1 <- rnorm(p * n_base)
  x_c2 <- rnorm(p * n_base)
  # Full (replicated) latent is field-major over 2*n_base cells:
  #   field0 = [c1 field0, c2 field0]; field1 = [c1 field1, c2 field1].
  f0 <- c(x_c1[1:n_base],               x_c2[1:n_base])
  f1 <- c(x_c1[(n_base + 1):(2 * n_base)], x_c2[(n_base + 1):(2 * n_base)])
  x_full <- c(f0, f1)

  lp_full <- tulpa:::cpp_test_mcar_prior(
    theta, p, 2L * n_base,
    csr_rep$row_ptr, csr_rep$col_idx, csr_rep$n_neighbors,
    x_full, n_components = 2L)$log_prior
  lp_parts <-
    tulpa:::cpp_test_mcar_prior(theta, p, n_base,
      csr_base$row_ptr, csr_base$col_idx, csr_base$n_neighbors, x_c1, 1L)$log_prior +
    tulpa:::cpp_test_mcar_prior(theta, p, n_base,
      csr_base$row_ptr, csr_base$col_idx, csr_base$n_neighbors, x_c2, 1L)$log_prior
  expect_equal(lp_full, lp_parts, tolerance = 1e-9)
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
  skip_if_fast()
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
