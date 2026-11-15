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
