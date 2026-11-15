skip_on_cran()

test_that('complete sparse fits agree below and above the automatic factor gate', {
  skip_if_not_slow()
  skip_if_no_cuda(c('cudss', 'nvrtc', 'cublas'))
  gate <- device_sparse_factor_gate()
  sizes <- device_sparse_fit_sizes()
  for (kind in c('spde', 'icar', 'areal', 'temporal', 'joint')) for (nodes in sizes) {
    run <- function(choice) do.call(tulpa_nested_laplace_joint,
      device_sparse_fit_inputs(nodes, kind, choice))
    cpu <- run('cpu')
    expect_true(all(cpu$converged))
    variance <- device_sparse_latent_variance(cpu, 'cpu')
    expect_true(all(is.finite(variance)) && all(variance > 0))
    for (choice in c('cuda', 'auto')) {
      fit <- run(choice)
      expect_true(all(fit$converged))
      expect_identical(fit$theta_grid, cpu$theta_grid)
      expect_false(any(fit$device_cpu_recovered))
      for (q in c('log_marginal', 'modes', 'fitted_eta_var')) {
        label <- paste('fit', kind, nodes, choice, q)
        cat(sprintf('%s scale=%.17g relative_tolerance=1e-7\n', label, max(1, abs(cpu[[q]]))))
        expect_device_equal(fit[[q]], cpu[[q]], 1e-7, label)
      }
      actual <- device_sparse_latent_variance(fit, choice)
      label <- paste('fit', kind, nodes, choice, 'latent marginal variances')
      cat(sprintf('%s scale=%.17g relative_tolerance=1e-7\n', label, max(1, abs(variance))))
      expect_device_equal(actual, variance, 1e-7, label)
      stages <- fit$device$stages
      used <- any(stages$stage == 'sparse_cholesky' & stages$cuda_calls > 0)
      expect_identical(used, choice == 'cuda' || (length(gate) == 1L && ncol(fit$modes) >= gate))
      if (choice == 'auto') {
        expect_true(any(stages$stage == 'sparse_selected_inverse' & stages$cpu_calls > 0))
        expect_false(any(stages$stage == 'sparse_selected_inverse' & stages$cuda_calls > 0))
      }
    }
    rm(cpu, fit, variance, actual); gc(FALSE)
  }
})

test_that('registry areal and temporal fits retain equivalent latent marginals across the gate', {
  skip_if_not_slow()
  skip_if_no_cuda(c('cudss','nvrtc','cublas'))
  gate <- device_sparse_factor_gate()
  for(kind in c('areal','temporal')) for(nodes in device_sparse_fit_sizes()) {
    run <- function(choice) device_sparse_fit(device_sparse_fit_inputs(nodes,kind,choice),'registry')
    cpu <- run('cpu')
    expect_true(all(cpu$converged))
    variance <- if(kind=='areal') 'grid_spatial_var' else 'grid_field_var'
    expect_true(length(cpu[[variance]])>0L && all(cpu[[variance]]>0))
    for(choice in c('auto','cuda')) {
      fit <- run(choice)
      expect_true(all(fit$converged))
      expect_false(any(fit$device_cpu_recovered))
      stages <- fit$device$stages
      used <- any(stages$stage=='sparse_cholesky' & stages$cuda_calls>0)
      expect_identical(used,choice=='cuda' || (length(gate)==1L && ncol(fit$modes)>=gate))
      for(q in c('log_marginal','modes',variance)) {
        label <- paste('registry',kind,nodes,choice,q)
        cat(sprintf('%s scale=%.17g relative_tolerance=1e-7\n',label,max(1,abs(cpu[[q]]))))
        expect_device_equal(fit[[q]],cpu[[q]],1e-7,label)
      }
    }
  }
})
