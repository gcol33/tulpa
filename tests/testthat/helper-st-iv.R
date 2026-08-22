# Shared scaffolding for the Knorr-Held Type-IV fixture in src/test_st_iv_fixture.cpp.
#
# A helper file rather than the top of one test file: both the
# precision-informed diagonal (gcol33/tulpa#585) and the margin low-rank metric
# (gcol33/tulpa#597) are scored against the SAME numerical Hessian of the
# engine's own log-posterior, and a second copy of the fixture is a second
# thing to keep in step with the C++ entry points.

# ---------------------------------------------------------------------------
# Fixture: an S1 x S2 rook-adjacency grid observed at T times, one observation
# per (unit, time) cell.
# ---------------------------------------------------------------------------
st_iv_fixture <- function(S1 = 3, S2 = 3, T = 4, family = "poisson",
                          temporal = "rw1", temporal_cyclic = FALSE,
                          st_parameterization = 0,
                          drop_edges_of = integer(), seed = 1) {
  S <- S1 * S2
  nb <- vector("list", S)
  for (s in seq_len(S)) {
    r <- ((s - 1) %/% S2) + 1
    c <- ((s - 1) %% S2) + 1
    cand <- c(
      if (r > 1)  s - S2 else integer(),
      if (r < S1) s + S2 else integer(),
      if (c > 1)  s - 1  else integer(),
      if (c < S2) s + 1  else integer()
    )
    nb[[s]] <- sort(cand)
  }
  # Isolating a unit exercises the disconnected-graph case: st_spatial_rank
  # reads the component count and the assembly reads a zero degree.
  for (s in drop_edges_of) {
    for (o in nb[[s]]) nb[[o]] <- setdiff(nb[[o]], s)
    nb[[s]] <- integer()
  }

  adj_row_ptr <- c(0L, cumsum(vapply(nb, length, 1L)))
  adj_col_idx <- as.integer(unlist(nb))
  if (length(adj_col_idx) == 0) adj_col_idx <- integer()

  set.seed(seed)
  grid <- expand.grid(t = seq_len(T), s = seq_len(S))
  N <- nrow(grid)
  X <- cbind(1, stats::rnorm(N))
  y <- if (family == "gaussian") stats::rnorm(N) else stats::rpois(N, 3)

  list(y = as.numeric(y), X = X,
       s_idx = as.integer(grid$s), t_idx = as.integer(grid$t),
       adj_row_ptr = as.integer(adj_row_ptr), adj_col_idx = adj_col_idx,
       S = S, T = T, family = family, temporal = temporal,
       temporal_cyclic = temporal_cyclic,
       st_parameterization = st_parameterization)
}

st_iv_layout <- function(f) {
  cpp_test_st_iv_layout(
    f$y, f$X, f$s_idx, f$t_idx, f$adj_row_ptr, f$adj_col_idx, f$S, f$T,
    family = f$family, temporal = f$temporal,
    temporal_cyclic = f$temporal_cyclic,
    st_parameterization = f$st_parameterization)
}

st_iv_gmrf <- function(f, q, with_eta_weights = TRUE) {
  cpp_test_st_iv_gmrf_mass(
    f$y, f$X, f$s_idx, f$t_idx, f$adj_row_ptr, f$adj_col_idx, f$S, f$T,
    q = q, family = f$family, temporal = f$temporal,
    temporal_cyclic = f$temporal_cyclic,
    st_parameterization = f$st_parameterization,
    with_eta_weights = with_eta_weights)
}

st_iv_lp <- function(f, q) {
  cpp_test_st_iv_log_post(
    f$y, f$X, f$s_idx, f$t_idx, f$adj_row_ptr, f$adj_col_idx, f$S, f$T,
    q = q, family = f$family, temporal = f$temporal,
    temporal_cyclic = f$temporal_cyclic,
    st_parameterization = f$st_parameterization)
}

# Central-difference Hessian of the log-posterior over the coordinates `idx`.
#
# h = 1e-3 rather than the usual 1e-6. The sum-to-zero margins put the
# log-posterior at ~1e5 on this fixture while its second derivatives are ~10,
# so the second difference is a cancellation and the roundoff floor is
# |f| * eps / h^2. Measured over h in {1e-3, 3e-4, 1e-4, 3e-5} the agreement
# with the override degrades monotonically as h shrinks -- 1.3e-05, 1.3e-04,
# 1.6e-03, 2.1e-02 -- which is that floor and not a step-size the difference
# has yet to resolve. The prior is exactly quadratic here, so the truncation
# term a smaller h would buy is zero for all of it but the likelihood.
st_iv_num_hessian <- function(f, q, idx, h = 1e-3) {
  k <- length(idx)
  H <- matrix(0, k, k)
  f0 <- st_iv_lp(f, q)
  bump <- function(a, da, b, db) {
    qq <- q
    qq[idx[a]] <- qq[idx[a]] + da
    qq[idx[b]] <- qq[idx[b]] + db
    st_iv_lp(f, qq)
  }
  for (a in seq_len(k)) {
    H[a, a] <- (bump(a, h, a, 0) - 2 * f0 + bump(a, -h, a, 0)) / (h * h)
    if (a < k) for (b in (a + 1):k) {
      v <- (bump(a, h, b, h) - bump(a, h, b, -h) -
            bump(a, -h, b, h) + bump(a, -h, b, -h)) / (4 * h * h)
      H[a, b] <- v
      H[b, a] <- v
    }
  }
  H
}

