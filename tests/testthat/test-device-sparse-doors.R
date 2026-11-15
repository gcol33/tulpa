skip_on_cran()

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
