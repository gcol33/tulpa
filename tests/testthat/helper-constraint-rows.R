# Reference builders for the constraint rows a nested driver hands back
# (`constraint_cols`): an integer vector of 1-based latent indices is a
# sum-to-zero group (every weight 1); a list(cols, w) is a weighted row. The
# tests build dense / sparse constraint matrices from them to check the
# constrained reads against.
.constraint_row_dense <- function(cc, n_x) {
  a <- numeric(n_x)
  if (is.list(cc)) {
    cols <- as.integer(cc$cols)
    a[sort(unique(cols))] <- as.numeric(rowsum(as.numeric(cc$w), cols))
  } else {
    a[unique(as.integer(cc))] <- 1
  }
  a
}

# The rows as the columns of a sparse n_x x k matrix (A').
.constraint_rows_matrix <- function(constr, n_x) {
  ij <- lapply(seq_along(constr), function(g) {
    cc <- constr[[g]]
    if (is.list(cc)) {
      data.frame(i = as.integer(cc$cols), j = g, x = as.numeric(cc$w))
    } else {
      u <- unique(as.integer(cc))
      data.frame(i = u, j = rep(g, length(u)), x = rep(1, length(u)))
    }
  })
  ij <- do.call(rbind, ij)
  Matrix::sparseMatrix(i = ij$i, j = ij$j, x = ij$x,
                       dims = c(n_x, length(constr)))
}