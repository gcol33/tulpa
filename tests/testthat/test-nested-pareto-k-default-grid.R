# The single-block outer Pareto-k must be computed whenever the realised grid is
# a single positive-scale axis -- including when the user relied on the default
# grid (so the input prior carries no `*_grid` field). Before #203 eligibility
# keyed on grepping the un-defaulted input prior, so a default-grid fit silently
# declined to pareto_k = NA even though an explicit-grid fit on the same axis
# produced a real k-hat.

rook_pk <- function(nr, nc) {
  n <- nr * nc; W <- matrix(0, n, n); id <- function(r, c) (c - 1) * nr + r
  for (r in seq_len(nr)) for (c in seq_len(nc)) {
    if (r < nr) { W[id(r, c), id(r + 1, c)] <- 1; W[id(r + 1, c), id(r, c)] <- 1 }
    if (c < nc) { W[id(r, c), id(r, c + 1)] <- 1; W[id(r, c + 1), id(r, c)] <- 1 }
  }
  W
}

test_that("default-grid single-block Pareto-k is computed, not declined (#203)", {
  skip_on_cran()
  set.seed(5)
  nr <- nc <- 5L; S <- nr * nc; reps <- 4L
  W <- rook_pk(nr, nc)
  unit <- rep(seq_len(S), each = reps); N <- length(unit)
  x <- rnorm(N); ntr <- rep(3L, N)
  y <- rbinom(N, ntr, plogis(-0.3 + 0.6 * x))
  X <- cbind(1, x)
  idx <- tulpa:::.resolve_unit_index(factor(unit), "region", S)
  csr <- tulpa:::adjacency_to_csr_tulpa(W)
  base <- list(type = "icar", spatial_idx = idx, n_spatial_units = S,
               adj_row_ptr = csr$row_ptr, adj_col_idx = csr$col_idx,
               n_neighbors = csr$n_neighbors)

  f_def <- tulpa_nested_laplace(y = y, n_trials = ntr, X = X, prior = base,
                                family = "binomial")
  f_exp <- tulpa_nested_laplace(
    y = y, n_trials = ntr, X = X, family = "binomial",
    prior = c(base, list(tau_grid = tulpa:::.default_tau_grid())))

  expect_false(is.na(f_def$pareto_k))                    # was NA before the fix
  # Same axis, same (default) grid -> identical k-hat whether named or defaulted.
  expect_equal(f_def$pareto_k, f_exp$pareto_k)
})
