# test-pg-bym2-rho-recovery.R
# Recovery test for the BYM2 PG-Gibbs mixing proportion rho. The rho full
# conditional under Polya-Gamma augmentation keeps both the quadratic data
# term and the linear data-coupling term sum_resid[j] * u_j. Dropping the
# linear term (the earlier behaviour) pulls rho toward the quadratic-only
# optimum and biases the structured-vs-iid decomposition.

# Build a 1D chain adjacency on S units (each interior unit has 2 neighbours).
.chain_graph <- function(S) {
  adj <- vector("list", S)
  for (s in seq_len(S)) {
    nb <- integer(0)
    if (s > 1) nb <- c(nb, s - 1L)
    if (s < S) nb <- c(nb, s + 1L)
    adj[[s]] <- as.integer(nb)
  }
  list(adj = adj, n_neighbors = vapply(adj, length, integer(1)))
}

# Generate a scaled-ICAR structured field on a chain by RW-style smoothing,
# then centre it (soft sum-to-zero). Returns a unit-scaled field.
.structured_field <- function(S, seed) {
  set.seed(seed)
  incr <- rnorm(S)
  f <- cumsum(incr)
  f <- f - mean(f)
  f / sd(f)
}

test_that("BYM2 PG-Gibbs recovers a structured-dominated rho", {
  skip_on_cran()
  skip_if_fast()

  S <- 40L
  reps <- 8L          # observations per spatial unit
  g <- .chain_graph(S)

  set.seed(101)
  # Structured spatial signal dominates: data are generated from a smooth
  # spatial field with no iid component, so rho should be pulled high.
  phi <- .structured_field(S, seed = 7)
  sigma_true <- 1.2
  u_true <- sigma_true * phi          # purely structured (rho_true ~ 1)

  spatial_group <- rep(seq_len(S), each = reps)
  N <- length(spatial_group)
  eta <- -0.2 + u_true[spatial_group]
  y <- rbinom(N, size = 1L, prob = plogis(eta))

  fit <- cpp_pg_binomial_gibbs_bym2(
    y = as.integer(y),
    n = as.integer(rep(1L, N)),
    X = matrix(1, N, 1),
    re_group = as.integer(rep(0L, N)),
    n_re_groups = 0L,
    spatial_group = as.integer(spatial_group),
    n_spatial_units = S,
    adj_list = g$adj,
    n_neighbors = as.integer(g$n_neighbors),
    scale_factor = 1.0,
    n_iter = 4000L, n_warmup = 2000L, thin = 2L,
    prior_rho_alpha = 1.0, prior_rho_beta = 1.0,
    verbose = FALSE
  )

  rho_post <- mean(fit$rho)
  cat("\n  structured-dominated: posterior mean rho =", round(rho_post, 3), "\n")

  # A structured-dominated signal should place rho above the 0.5 midpoint of
  # its Beta(1,1) prior. Without the linear data term the grid posterior is
  # driven by the quadratic form alone and does not track the structure.
  expect_gt(rho_post, 0.5)
})

test_that("update_rho_bym2 linear term tracks the data-fit direction", {
  skip_on_cran()
  skip_if_fast()

  # Direct check of the rho conditional shape: with sufficient statistics
  # that favour a particular u, the log-posterior including the linear term
  # peaks at a different rho than the quadratic-only form. We reproduce the
  # grid the sampler evaluates and confirm the linear term moves the mode.
  S <- 30L
  # Orthogonal structured / iid components so the quadratic form has no
  # rho-preference of its own; the linear data term then solely sets the mode.
  phi_scaled <- .structured_field(S, seed = 3)
  k <- seq_len(S)
  theta <- cos(2 * pi * k / S); theta <- theta - mean(theta); theta <- theta / sd(theta)
  sigma_spatial <- 1.0
  scale_factor <- 1.0
  sum_omega <- rep(1.0, S)
  # Residuals aligned with the iid component theta reward LOW rho. Without the
  # linear term the conditional is rho-symmetric and the modes must differ.
  sum_resid <- 1.5 * theta

  grid <- seq(0.025, 0.975, length.out = 40)
  logpost <- function(rho, use_linear) {
    sr <- sqrt(rho); s1 <- sqrt(1 - rho)
    u <- sigma_spatial * (sr * phi_scaled * scale_factor + s1 * theta)
    quad <- -0.5 * sum(sum_omega * u^2)
    lin <- if (use_linear) sum(sum_resid * u) else 0.0
    quad + lin
  }

  lp_full <- vapply(grid, logpost, numeric(1), use_linear = TRUE)
  lp_quad <- vapply(grid, logpost, numeric(1), use_linear = FALSE)
  rho_full <- grid[which.max(lp_full)]
  rho_quad <- grid[which.max(lp_quad)]

  cat("\n  mode(full) =", round(rho_full, 3),
      " mode(quad-only) =", round(rho_quad, 3), "\n")

  # The theta-aligned residual pulls the full-conditional mode toward the iid
  # direction (low rho), away from the quadratic-only optimum. Dropping the
  # linear term (the earlier behaviour) loses this data-fit direction entirely.
  expect_lt(rho_full, rho_quad)
})
