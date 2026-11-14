skip_on_cran()

device_sparse_door <- function(n, kind, choice) {
  idx <- seq_len(n)
  x <- cos(idx * 0.1)
  response <- list(y = 0.3 + 0.2*x + 0.1*sin(idx * 0.17),
    n_trials = rep(1L, n), X = cbind(intercept = 1, x = x),
    family = 'gaussian', phi = 1)
  temporal <- list(type = 'ar1', n_times = n, tau_grid = c(1,2),
                   rho_grid = c(0.4,0.6), temporal_idx = list(idx))
  neighbors <- lapply(idx, function(k) intersect(c(k-1L,k+1L),idx))
  count <- lengths(neighbors)
  areal <- list(type = 'car_proper', n_spatial_units = n,
    adj_row_ptr = as.integer(c(0,cumsum(count))),
    adj_col_idx = as.integer(unlist(neighbors)-1L), n_neighbors = as.integer(count),
    tau_grid = c(1,2), rho_grid = c(0.4,0.6), spatial_idx = list(idx))
  if (kind == 'spde') {
    fem <- device_sparse_fem(n)
    G <- methods::as(methods::as(fem$G, 'generalMatrix'), 'CsparseMatrix')
    A <- methods::as(methods::as(Matrix::Diagonal(n), 'generalMatrix'), 'CsparseMatrix')
    prior <- list(list(type = 'spde', n_mesh = n, nu = 1,
      range_grid = c(1.3,1.8), sigma_grid = c(0.7,0.9),
      C0_diag = fem$C, G1_p = G@p, G1_i = G@i, G1_x = G@x,
      A_p = list(A@p), A_i = list(A@i), A_x = list(A@x), n_obs_per_arm = n))
  } else prior <- switch(kind, areal = list(areal), temporal = list(temporal), joint = list(areal, temporal))
  tulpa_nested_laplace_joint(responses = list(y = response), prior = prior,
    control = list(device = choice, force_sparse = TRUE, integration = 'grid',
      max_iter = 100L, tol = 1e-9, n_threads = 1L, diagnose_k = FALSE, prune = FALSE,
      axis_refine = 'none', auto_recenter = FALSE, adaptive_grid = FALSE, store_Q = TRUE))
}

test_that('SPDE areal temporal and joint sparse doors agree on modes and variances', {
  skip_on_cran()
  skip_if_no_cuda('cudss')
  for (kind in c('spde','areal','temporal','joint')) for (n in c(63L,257L)) {
    cpu <- device_sparse_door(n, kind, 'cpu')
    cuda <- device_sparse_door(n, kind, 'cuda')
    expect_true(all(cpu$converged) && all(cuda$converged))
    for (q in c('log_marginal','modes','fitted_eta_var')) {
      expect_true(length(cpu[[q]]) > 0L)
      cat(sprintf('%s n=%d %s scale=%.17g tolerance=1e-7\n',kind,n,q,max(1,abs(cpu[[q]]))))
      expect_device_equal(cuda[[q]], cpu[[q]], 1e-7, paste(kind,n,q))
    }
    stage <- cuda$device$stages
    expect_true(any(stage$stage == 'sparse_cholesky' & stage$cuda_calls > 0))
    n_x <- ncol(cpu$modes)
    for (cell in seq_len(nrow(cpu$modes))) {
      run <- function(fit, choice) {
        session <- tulpa:::cpp_device_begin(choice)
        on.exit(tulpa:::cpp_device_end(session))
        tulpa_joint_inner_vcov_blocks(fit$Q_csc_p_per_grid[cell], fit$Q_csc_i_per_grid[cell],
          fit$Q_csc_x_per_grid[cell], n_x, seq_len(n_x), 2L, list(), n_threads = 1L)[[1L]]
      }
      a <- diag(run(cpu,'cpu')); b <- diag(run(cuda,'cuda'))
      expect_device_equal(b, a, 1e-7, paste(kind,n,cell,'latent marginal variance'))
    }
  }
})
