test_that('Laplace batches validate dimensions before allocating device storage', {
  f <- device_laplace_fixture(3L, 2L)
  f$rhs <- f$rhs[, 1L, drop = FALSE]
  expect_error(device_laplace_run(f, 'cpu'), 'invalid Laplace batch dimensions')
})

test_that('CPU batch solves agree with dense reference for every cell', {
  f <- device_laplace_fixture(17L, 13L)
  r <- device_laplace_run(f, 'cpu')
  expect_true(all(r$positive))
  expect_identical(r$errors, rep('', 13L))
  for (k in seq_len(nrow(f$rhs))) {
    h <- matrix(f$matrices[k, ], f$n, f$n, byrow = TRUE)
    expect_device_equal(r$solution[k, ], as.vector(solve(h, f$rhs[k, ])), 1e-10, 'CPU batch solve')
    expect_device_equal(r$logdet[k], as.numeric(determinant(h, logarithm = TRUE)$modulus), 1e-10, 'CPU batch logdet')
  }
})

test_that('full CSC fixtures preserve each cell Hessian and right-hand side', {
  f <- device_laplace_fixture(17L,13L)
  csc <- device_full_csc_fixture(f)
  dense <- device_laplace_run(f,'cpu')
  sparse <- do.call(tulpa:::cpp_device_sparse_batch,c(csc,list(choice='cpu')))
  expect_device_equal(t(matrix(sparse$solution,17L)),dense$solution,1e-10,'CSC cell RHS')
  expect_device_equal(sparse$logdet,dense$logdet,1e-10,'CSC cell logdet')
  csc$b <- csc$b[,-1L,drop=FALSE]
  expect_error(do.call(tulpa:::cpp_device_sparse_batch,c(csc,list(choice='cpu'))),
               'invalid sparse RHS batches')
})

test_that('cell recovery flags survive joint and registry refinement packing', {
  res <- list(log_marginal=c(-1,-2),device_cpu_recovered=c(FALSE,TRUE))
  extra <- tulpa:::.joint_extras_from_res(res,2L)
  packed <- tulpa:::.joint_glue_extras_to_res(res,matrix(1:3,ncol=1L),c(-1,-2,-3),
    c(extra,list(list(device_cpu_recovered=FALSE))),rep('',3L))
  expect_identical(packed$device_cpu_recovered,c(FALSE,TRUE,FALSE))
  bound <- tulpa:::.nl_bind_cell_results(res,list(list(
    res=list(log_marginal=-3,device_cpu_recovered=TRUE),n=1L)),2L)
  expect_identical(bound$device_cpu_recovered,c(FALSE,TRUE,TRUE))
})

skip_on_cran()

test_that('dense Laplace factors agree on both sides of every worker gate', {
  skip_if_no_cuda(c('cusolver','nvrtc'))
  gates <- tulpa:::.device_thresholds(read.csv(system.file('device','benchmarks.csv',package='tulpa')))
  gates <- gates[gates$operation == 'laplace_dense', ]
  for (row in seq_len(nrow(gates))) {
    gate <- gates$threshold[row]
    sizes <- if (is.finite(gate)) as.integer(c(max(1,gate-1),gate+1)) else c(31L,129L)
    for (n in sizes) {
      f <- device_laplace_fixture(n, gates$variant[row] + 1L)
      cpu <- device_laplace_run(f,'cpu',gates$variant[row])
      cuda <- device_laplace_run(f,'cuda',gates$variant[row])
      auto <- device_laplace_run(f,'auto',gates$variant[row])
      for (r in list(cuda,auto)) {
        expect_true(all(r$positive)); expect_false(any(r$recovered))
        expect_true(all(r$errors == ''))
        for (q in c('solution','logdet')) expect_device_equal(r[[q]],cpu[[q]],1e-8,
          paste('dense Laplace gate',gates$variant[row],n,q))
      }
      used <- any(auto$stages$stage == 'laplace_dense' & auto$stages$cuda_calls > 0)
      expect_identical(used,n >= gate)
    }
  }
})

test_that('uniform sparse factors agree across every worker threshold', {
  skip_if_no_cuda(c('cudss','nvrtc'))
  gates <- tulpa:::.device_thresholds(read.csv(system.file('device','benchmarks.csv',package='tulpa')))
  gates <- gates[gates$operation == 'sparse_uniform', ]
  for (row in seq_len(nrow(gates))) {
    gate <- gates$threshold[row]
    sizes <- if (is.finite(gate)) as.integer(c(max(2,gate-1),gate+1)) else c(63L,257L)
    for (n in sizes) {
      f <- device_sparse_fixture(n,'spde')
      values <- outer(f$values[,1L],1+seq_len(gates$variant[row]+1L)/100)
      run <- function(choice) tulpa:::cpp_device_sparse_batch(f$p,f$i,values,f$b,
        choice,gates$variant[row],TRUE,FALSE)
      cpu <- run('cpu'); cuda <- run('cuda'); auto <- run('auto')
      for (r in list(cuda,auto)) {
        expect_true(all(r$positive)); expect_false(any(r$recovered))
        expect_true(all(r$errors == ''))
        for(q in c('solution','logdet')) expect_device_equal(r[[q]],cpu[[q]],1e-8,
          paste('uniform sparse gate',gates$variant[row],n,q))
      }
      used <- any(auto$stages$stage == 'sparse_uniform_analysis' & auto$stages$cuda_calls > 0)
      expect_true(any(cuda$stages$stage == 'sparse_uniform_analysis' & cuda$stages$cuda_calls > 0))
      expect_identical(used,n >= gate && n >= gates$batch_threshold[row])
    }
  }
})

test_that('joint sparse outer workers preserve each declared cell on CUDA', {
  skip_if_no_cuda(c('cudss','nvrtc'))
  for (kind in c('spde', 'joint')) for (prune in c(FALSE, TRUE)) {
    input <- device_sparse_door_inputs(47L, kind, 'cpu', family = 'poisson', outer = 4L)
    input$control$prune <- prune
    input$control$prune_tol <- 0.01
    for (b in seq_along(input$prior)) for (axis in intersect(
      c('range_grid', 'sigma_grid', 'tau_grid', 'rho_grid'), names(input$prior[[b]]))) {
      values <- input$prior[[b]][[axis]]
      input$prior[[b]][[axis]] <- seq(min(values), max(values), length.out = 8L)
    }
    cpu <- do.call(tulpa_nested_laplace_joint, input)
    input$control$device <- 'cuda'
    cuda <- do.call(tulpa_nested_laplace_joint, input)
    expect_identical(cpu$theta_grid, cuda$theta_grid)
    expect_identical(cpu$converged, cuda$converged)
    active <- is.finite(cpu$log_marginal)
    if (!prune) expect_true(all(active))
    expect_true(any(active))
    expect_true(all(cpu$converged[active]) && all(cuda$converged[active]))
    for (q in c('log_marginal', 'modes', 'fitted_eta', 'fitted_eta_var'))
      expect_device_equal(cuda[[q]], cpu[[q]], 1e-7, paste(kind, q))
    for (q in intersect(c('prune_cheap_log_marginal', 'prune_screen_decrement', 'log_ref'), names(cpu))) {
      finite <- is.finite(cpu[[q]])
      expect_identical(is.finite(cuda[[q]]), finite)
      if (any(finite)) expect_device_equal(cuda[[q]][finite], cpu[[q]][finite], 1e-7, paste(kind, 'screen', q))
    }
  }
})

test_that('HSGP cells agree on the registry and joint doors with CUDA', {
  skip_if_no_cuda(c('cusolver','nvrtc'))
  n <- 80L
  idx <- seq_len(n)
  coords <- cbind(idx/n, (idx %% 13L)/13)
  basis <- tulpa:::cpp_hsgp_basis_2d(coords, 6L, 1.5)
  response <- list(y = as.numeric(idx %% 3L), n_trials = rep(1L, n),
    X = cbind(1, cos(idx)), family = 'poisson')
  axes <- expand.grid(sigma = c(0.3, 0.8), range = c(0.2, 0.5))
  block <- list(type = 'hsgp', sigma2_grid = axes$sigma, lengthscale_grid = axes$range)
  registry <- c(block, list(phi_basis = basis$phi_basis, lambda_eig = basis$lambda_eig))
  joint <- c(block, list(m_total = ncol(basis$phi_basis), phi = list(basis$phi_basis),
    eigenvalues = basis$lambda_eig, n_obs_per_arm = n))
  control <- list(axis_refine = 'none', prune = FALSE, diagnose_k = FALSE,
    auto_recenter = FALSE, n_threads_outer = 4L, progress = FALSE)
  for (door in c('registry', 'joint')) {
    run <- function(choice) {
      control$device <- choice
      if (door == 'registry') do.call(tulpa_nested_laplace,
        c(response[c('y','n_trials','X','family')], list(prior = list(registry), control = control)))
      else tulpa_nested_laplace_joint(list(y = response), list(joint), control = control)
    }
    cpu <- run('cpu'); cuda <- run('cuda')
    expect_identical(cpu$theta_grid, cuda$theta_grid)
    if (door == 'registry') {
      stages <- cuda$device$stages
      for (operation in c('likelihood_weights', 'laplace_scatter'))
        expect_true(any(stages$stage == operation & stages$cuda_calls > 0), info = operation)
      expect_false(any(cuda$device_cpu_recovered))
    }
    for (q in c('log_marginal', 'modes', 'fitted_eta', 'fitted_eta_var'))
      expect_device_equal(cuda[[q]], cpu[[q]], 1e-7, paste(door, 'HSGP', q))
  }
})

test_that('full CSC Hessians use their own CPU crossover', {
  skip_if_no_cuda(c('cusolver','nvrtc'))
  table <- tulpa:::.device_thresholds(read.csv(system.file('device','benchmarks.csv',package='tulpa')))
  gates <- table[table$operation=='laplace_full_csc' & is.finite(table$threshold),]
  for(row in seq_len(nrow(gates)))for(n in unique(as.integer(c(max(1,gates$threshold[row]-1),gates$threshold[row])))) {
    f <- device_full_csc_fixture(device_laplace_fixture(n,13L))
    run <- function(choice)do.call(tulpa:::cpp_device_sparse_batch,c(f,list(
      choice=choice,workers=gates$variant[row],batched=TRUE,selected_inverse=FALSE)))
    cpu <- run('cpu');auto <- run('auto')
    used <- sum(auto$stages$cuda_calls[auto$stages$stage=='laplace_full_csc'])>0
    expect_identical(used,n>=gates$threshold[row])
    expect_false(any(auto$recovered))
    expect_identical(auto$errors,rep('',13L))
    for(q in c('solution','logdet'))expect_device_equal(auto[[q]],cpu[[q]],1e-8,paste('full CSC crossover',q))
  }
})

test_that('dense device inverse diagonals agree for every cell', {
  skip_if_no_cuda(c('cusolver','nvrtc'))
  for(width in c(1L,4L,8L)) for(n in c(31L,74L,252L)) {
    f <- device_full_csc_fixture(device_laplace_fixture(n,13L))
    f$b <- matrix(numeric(),n,0L)
    run <- function(choice,batched=TRUE)do.call(tulpa:::cpp_device_sparse_batch,
      c(f,list(choice=choice,workers=width,batched=batched,selected_inverse=TRUE)))
    cpu <- run('cpu')
    for(batched in c(FALSE,TRUE)) {
      cuda <- run('cuda',batched)
      expect_true(all(cuda$positive))
      expect_false(any(cuda$recovered))
      expect_identical(cuda$errors,rep('',13L))
      for(q in c('logdet','diagonal'))expect_device_equal(cuda[[q]],cpu[[q]],1e-8,paste('dense inverse',q))
    }
  }
})

test_that('dense inverse selection follows its own measured crossover', {
  skip_if_no_cuda(c('cusolver','nvrtc'))
  table <- tulpa:::.device_thresholds(read.csv(system.file('device','benchmarks.csv',package='tulpa')))
  gates <- table[table$operation=='laplace_dense_inverse' & is.finite(table$threshold),]
  for(row in seq_len(nrow(gates)))for(n in unique(as.integer(c(max(1,gates$threshold[row]-1),gates$threshold[row])))) {
    f <- device_full_csc_fixture(device_laplace_fixture(n,13L))
    f$b <- matrix(numeric(),n,0L)
    run <- function(choice)do.call(tulpa:::cpp_device_sparse_batch,c(f,list(
      choice=choice,workers=gates$variant[row],selected_inverse=TRUE)))
    cpu <- run('cpu');auto <- run('auto')
    used <- sum(auto$stages$cuda_calls[auto$stages$stage=='laplace_dense_inverse'])>0
    expect_identical(used,n>=gates$threshold[row])
    expect_false(any(auto$recovered))
    expect_identical(auto$errors,rep('',13L))
    for(q in c('logdet','diagonal'))expect_device_equal(auto[[q]],cpu[[q]],1e-8,paste('inverse crossover',q))
  }
})

test_that('uniform sparse cells share analysis and retain independent numeric factors', {
  skip_if_no_cuda(c('cudss','nvrtc'))
  r <- tulpa:::cpp_device_sparse_state('cuda', 4L, TRUE)
  expect_identical(r$errors, rep('', 4L))
  expect_identical(r$success, rep(1L, 4L))
  expect_equal(r$stages$cuda_calls[r$stages$stage == 'sparse_uniform_analysis'], 1)
})

test_that('sparse streams and uniform batches agree cell by cell', {
  skip_if_no_cuda(c('cudss','nvrtc'))
  for (kind in c('spde','areal','temporal')) {
    f <- device_sparse_fixture(31L,kind)
    values <- outer(f$values[,1L], 1 + seq_len(13L)/10)
    run <- function(choice, batched) tulpa:::cpp_device_sparse_batch(
      f$p,f$i,values,f$b,choice,4L,batched,TRUE)
    cpu <- run('cpu',FALSE)
    for(batched in c(FALSE,TRUE)) {
      cuda <- run('cuda',batched)
      expect_true(all(cuda$positive))
      expect_false(any(cuda$recovered))
      expect_identical(cuda$errors,rep('',13L))
      for(q in c('solution','logdet','diagonal'))
        expect_device_equal(cuda[[q]],cpu[[q]],1e-8,paste(kind,batched,q))
    }
  }
})

test_that('a nonfinite sparse solution replays its cell and preserves shared factors', {
  skip_if_no_cuda(c('cudss','nvrtc'))
  f <- device_sparse_fixture(31L,'spde')
  values <- outer(f$values[,1L],1+seq_len(13L)/10)
  b <- matrix(rep(f$b[,1L],13L),nrow=31L)
  b[1L,4L] <- Inf
  for(batched in c(FALSE,TRUE)) {
    run <- function(choice)tulpa:::cpp_device_sparse_batch(f$p,f$i,values,b,
      choice,4L,batched,FALSE,TRUE)
    cpu <- run('cpu');cuda <- run('cuda')
    expect_identical(which(cuda$recovered),4L)
    expect_true(all(cuda$positive[-4L]))
    expect_identical(cuda$errors,rep('',13L))
    expect_true(all(is.finite(matrix(cuda$solution,31L)[,-4L])))
    expect_device_equal(cuda$solution,cpu$solution,1e-8,'sparse cell replay')
  }
})

test_that('batched and singleton device launches preserve each cell and recover only the failed cell', {
  skip_if_no_cuda(c('cusolver','nvrtc'))
  for (n in c(31L, 74L, 128L)) {
    f <- device_laplace_fixture(n, 37L)
    cpu <- device_laplace_run(f, 'cpu')
    for (batched in c(FALSE, TRUE)) {
      cuda <- device_laplace_run(f, 'cuda', batched = batched)
      expect_true(all(cuda$positive))
      expect_identical(cuda$errors, rep('', 37L))
      expect_false(any(cuda$recovered))
      for (q in c('solution', 'logdet')) {
        cat(sprintf('dense n=%d batched=%s %s scale=%.17g tolerance=1e-8\n',
          n, batched, q, max(1, abs(cpu[[q]]))))
        expect_device_equal(cuda[[q]], cpu[[q]], 1e-8, paste(n, batched, q))
      }
    }
    recovered <- device_laplace_run(f, 'cuda', fail_cell = 3L)
    expect_identical(which(recovered$recovered), 4L)
    expect_true(all(recovered$positive))
    expect_device_equal(recovered$solution, cpu$solution, 1e-8, 'cell CPU replay')
    expect_identical(recovered$stages$cpu_calls[recovered$stages$stage == 'outer_cell_cpu_recovery'], 1)
  }
})

test_that('a rejected dense pivot leaves its cell unusable and preserves the remaining factors', {
  skip_if_no_cuda(c('cusolver','nvrtc'))
  f <- device_laplace_fixture(17L,13L)
  f$matrices[4L,1L] <- -1
  for(choice in c('cpu','cuda')) {
    r <- device_laplace_run(f,choice)
    expect_identical(which(!r$positive),4L)
    expect_true(is.nan(r$logdet[4L]))
    expect_true(all(is.nan(r$solution[4L,])))
    expect_true(all(is.finite(r$solution[-4L,])))
  }
})

test_that('a nonfinite device solution restarts its cell and preserves the worker context', {
  skip_if_no_cuda(c('cusolver','nvrtc'))
  f <- device_laplace_fixture(17L,13L)
  f$rhs[4L,1L] <- Inf
  cpu <- device_laplace_run(f,'cpu')
  cuda <- device_laplace_run(f,'cuda')
  expect_identical(which(!cpu$positive),4L)
  expect_identical(which(!cuda$positive),4L)
  expect_identical(which(cuda$recovered),4L)
  expect_true(all(is.finite(cuda$solution[-4L,])))
  expect_identical(cuda$errors,rep('',13L))
})
