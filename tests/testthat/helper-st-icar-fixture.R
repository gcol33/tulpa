# A Poisson st_icar fit (cpp_nested_laplace_st_icar): an ICAR chain over n_s
# units plus an RW1 over n_t times, both intrinsic and loading every row, on
# a two-cell (tau_spatial, tau_temporal) grid. `A` is the loading matrix of
# eta on the latent vector [beta, ICAR, RW1].

.st_icar_chain_adj <- function(n_s) {
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  list(adj_row_ptr = as.integer(c(0L, cumsum(lengths(nbr)))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(lengths(nbr)),
       n_spatial_units = n_s)
}

.st_icar_fixture <- function(n_s, n_t, N, seed) {
  set.seed(seed)
  adj <- .st_icar_chain_adj(n_s)
  s_idx <- sample.int(n_s, N, replace = TRUE)
  t_idx <- sample.int(n_t, N, replace = TRUE)
  X <- cbind(1, rnorm(N))
  eta <- 0.3 + 0.5 * X[, 2] + sin(s_idx / 3) + cos(t_idx / 2)
  args <- c(list(
    y = as.numeric(rpois(N, exp(eta))), n = rep(1L, N), X = X,
    re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1,
    spatial_idx = as.integer(s_idx),
    temporal_idx = as.integer(t_idx), n_times = n_t,
    tau_spatial_grid = c(1, 3), temporal_type = "rw1",
    tau_temporal_grid = c(2, 4), rho_temporal_grid = NULL, cyclic = FALSE,
    family = "poisson", phi = 1, max_iter = 100L, tol = 1e-10, n_threads = 1L),
    adj)
  A <- cbind(X,
             outer(s_idx, seq_len(n_s), `==`) * 1,
             outer(t_idx, seq_len(n_t), `==`) * 1)
  # The RW1 chain's pin, which the sparse driver always folds.
  rw_fold <- list(idx = ncol(X) + n_s + seq_len(n_t),
                  coef = args$tau_temporal_grid / n_t)
  list(args = args, A = A, rw_fold = rw_fold)
}
