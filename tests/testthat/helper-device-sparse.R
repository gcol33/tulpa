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
