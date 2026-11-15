device_sparse_fem <- function(n) {
  width <- max(2L, as.integer(sqrt(n)))
  cells <- seq_len(n - width - 1L)
  cells <- cells[cells %% width != 0L]
  right <- c(cells, cells + width + 1L)
  along_x <- c(cells + 1L, cells + width)
  along_y <- c(cells + width, cells + 1L)
  from <- rep(right, 2L)
  to <- c(along_x, along_y)
  incidence <- Matrix::sparseMatrix(i = rep(seq_along(from), 2L),
    j = c(from, to), x = rep(c(1, -1), each = length(from)),
    dims = c(length(from), n))
  G <- Matrix::crossprod(incidence) / 2
  mass <- tabulate(c(right, along_x, along_y), nbins = n) / 6
  list(C = mass, G = G)
}

device_sparse_fixture <- function(n, kind = 'areal') {
  stopifnot(n >= 2L, kind %in% c('spde', 'areal', 'temporal', 'joint'))
  node <- seq_len(n)
  width <- max(2L, as.integer(sqrt(n)))
  from <- seq_len(n - 1L)
  if (kind %in% c('spde', 'areal')) from <- from[from %% width != 0L]
  to <- from + 1L
  if (kind %in% c('spde', 'areal')) {
    from <- c(from, seq_len(n - width))
    to <- c(to, seq_len(n - width) + width)
  }
  incidence <- Matrix::sparseMatrix(i = rep(seq_along(from), 2L),
    j = c(from, to), x = c(rep(1, length(from)), rep(-1, length(from))),
    dims = c(length(from), n))
  q <- Matrix::crossprod(incidence) + Matrix::Diagonal(n, 0.3)
  if (kind == 'spde') {
    fem <- device_sparse_fem(n)
    inverse_mass <- ifelse(fem$C > 0, 1 / fem$C, 0)
    q <- Matrix::Diagonal(x = fem$C) + 2 * fem$G +
      fem$G %*% Matrix::Diagonal(x = inverse_mass) %*% fem$G +
      Matrix::Diagonal(x = as.numeric(fem$C == 0))
  }
  if (kind == 'joint') {
    count <- n %/% 2L
    coupling <- Matrix::sparseMatrix(i = seq_len(count), j = n - seq_len(count) + 1L,
                                    x = 0.17, dims = c(n, n))
    q <- q + coupling + Matrix::t(coupling) + Matrix::Diagonal(n, 0.4)
  }
  h <- Matrix::forceSymmetric(q + Matrix::Diagonal(n, 1.2 + 0.1 * sin(node)), uplo = 'L')
  h <- methods::as(h, 'dsCMatrix')
  stopifnot(nrow(h) == n, length(h@p) == n + 1L, h@p[n + 1L] == length(h@i))
  list(p = h@p, i = h@i, values = cbind(h@x, h@x * 1.07),
       b = cbind(sin(node * 0.1), cos(node * 0.13)), matrix = h)
}

device_sparse_run <- function(fixture, choice, full_inverse = FALSE, repetitions = 1L) {
  tulpa:::cpp_device_sparse(fixture$p, fixture$i, fixture$values, fixture$b,
                            choice, full_inverse, repetitions)
}

device_sparse_door_inputs <- function(n, kind, choice, family = 'gaussian', outer = 1L,
                               areal_grid = FALSE, extra_control = list()) {
  idx <- seq_len(n)
  x <- cos(idx * 0.1)
  y <- if (family == 'gaussian') 0.3 + 0.2*x + 0.1*sin(idx * 0.17) else
    as.numeric(idx %% 3L) + as.numeric(idx %% 7L == 0L)
  response <- list(y = y, n_trials = rep(1L, n), X = cbind(intercept = 1, x = x),
    family = family, phi = 1)
  temporal <- list(type = 'ar1', n_times = n, tau_grid = c(1,2),
    rho_grid = c(0.4,0.6), temporal_idx = list(idx))
  width <- max(2L, as.integer(sqrt(n)))
  neighbors <- lapply(idx, function(k) {
    from <- c(if (k %% width != 1L || !areal_grid) k-1L,
              if (k %% width != 0L || !areal_grid) k+1L,
              if (areal_grid) c(k-width,k+width))
    from[from >= 1L & from <= n]
  })
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
  stopifnot(length(response$y) == n, nrow(response$X) == n,
    length(areal$adj_row_ptr) == n+1L)
  cat(sprintf('nested fixture kind=%s family=%s rows=%d field_nodes=%d outer=%d\n',
    kind, family, n, n, outer))
  control <- utils::modifyList(list(device = choice, force_sparse = TRUE, integration = 'grid',
    max_iter = 100L, tol = 1e-9, n_threads = 1L, n_threads_outer = outer,
    diagnose_k = FALSE, prune = FALSE, axis_refine = 'none', auto_recenter = FALSE,
    adaptive_grid = FALSE, store_Q = TRUE), extra_control)
  list(responses = list(y = response), prior = prior, control = control)
}

device_sparse_door <- function(...) do.call(tulpa_nested_laplace_joint, device_sparse_door_inputs(...))

device_sparse_fit_inputs <- function(nodes, kind, choice, cells = 2L) {
  n <- as.integer(nodes %/% if (kind == 'joint') 2L else 1L)
  input <- device_sparse_door_inputs(n, if (kind == 'icar') 'areal' else kind,
    choice, family = 'poisson', areal_grid = TRUE,
    extra_control = list(progress = FALSE))
  if (kind == 'icar') {
    input$prior[[1L]]$type <- 'icar'
    input$prior[[1L]]$rho_grid <- NULL
  }
  for (b in seq_along(input$prior)) for (axis in intersect(
    c('range_grid', 'sigma_grid', 'tau_grid', 'rho_grid'), names(input$prior[[b]]))) {
    input$prior[[b]][[axis]] <- input$prior[[b]][[axis]][seq_len(cells)]
  }
  input
}

device_sparse_fit <- function(input, door = 'joint') {
  if (door == 'joint') return(do.call(tulpa_nested_laplace_joint, input))
  stopifnot(door == 'registry', length(input$responses) == 1L)
  prior <- lapply(input$prior, function(block) {
    for (index in intersect(c('spatial_idx','temporal_idx'),names(block)))
      block[[index]] <- block[[index]][[1L]]
    block
  })
  control <- input$control
  control$force_sparse <- control$store_Q <- NULL
  do.call(tulpa_nested_laplace, c(input$responses[[1L]][c('y','n_trials','X','family','phi')],
    list(prior = if(length(prior)==1L) prior[[1L]] else prior, control = control)))
}

device_sparse_factor_gate <- function() {
  gates <- tulpa:::.device_thresholds(read.csv(system.file('device','benchmarks.csv',package='tulpa')))
  gates$threshold[gates$operation=='sparse_cholesky' & gates$variant==0L]
}

device_sparse_fit_sizes <- function() {
  gate <- device_sparse_factor_gate()
  unique(as.integer(c(2000L,if(length(gate) && is.finite(gate)) max(15000,gate*1.5) else 15000L)))
}

device_sparse_latent_variance <- function(fit, choice) {
  session <- tulpa:::cpp_device_begin(choice, inherit = FALSE)
  on.exit(tulpa:::cpp_device_end(session))
  nx <- ncol(fit$modes)
  idx <- seq.int(3L, nx)
  tulpa:::cpp_loading_marginal_var(fit$Q_csc_p_per_grid,
    fit$Q_csc_i_per_grid, fit$Q_csc_x_per_grid, nx,
    as.integer(idx), seq.int(0L, length(idx)),
    matrix(1, nrow(fit$modes), length(idx)),
    tulpa:::.constraint_rows_for_cpp(fit$constraint_cols), 1L,
    if (!is.null(fit$Q_soft_D_per_grid)) tulpa:::.constraint_rows_for_cpp(fit$Q_soft_cols),
    fit$Q_soft_D_per_grid)
}
