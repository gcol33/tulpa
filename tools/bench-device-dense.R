device_dense_runners <- function(append_row, basis_columns = c(16L, 64L, 144L, 225L, 256L, 400L, 900L)) {
  stopifnot(length(basis_columns) > 0L, all(is.finite(basis_columns)), all(basis_columns > 0))
  paired <- function(operation, variant, size, run, quantities) {
    cpu <- run("cpu", 1L); cuda <- run("cuda", 1L)
    if (!is.null(cpu$first_seconds)) cat(sprintf("first evaluation %s variant=%d size=%d CPU=%.6fs CUDA=%.6fs\n",
      operation, variant, size, cpu$first_seconds, cuda$first_seconds))
    for (quantity in quantities) {
      scale <- max(1, abs(cpu[[quantity]]))
      error <- max(abs(cpu[[quantity]] - cuda[[quantity]]))
      stopifnot(is.finite(error), error <= 3e-6 * scale)
      cat(sprintf("equivalence %s variant=%d size=%d %s scale=%.17g error=%.6g\n",
                  operation, variant, size, quantity, scale, error))
    }
    repetitions <- max(1L, min(100L, as.integer(ceiling(0.1 / min(cpu$seconds, cuda$seconds)))))
    for (trial in seq_len(5L)) {
      order <- if (trial %% 2L) c("cpu", "cuda") else c("cuda", "cpu")
      measured <- setNames(numeric(2L), c("cpu", "cuda"))
      for (choice in order) measured[choice] <- run(choice, repetitions)$seconds
      append_row(operation, variant, size, measured["cpu"], measured["cuda"])
    }
  }
  dense <- function(operation, ladder) function() {
    for (n in ladder) {
      a <- outer(seq_len(n), seq_len(n), function(i, j) exp(-abs(i - j) / 4)) + diag(0.1, n)
      rhs <- cbind(sin(seq_len(n) * 0.31), cos(seq_len(n) * 0.17))
      stopifnot(nrow(a) == n, ncol(a) == n, nrow(rhs) == n)
      run <- function(choice, repetitions) tulpa:::cpp_device_dense(a, rhs, operation, choice, repetitions)
      paired(operation, 0L, n, run, c("log_density", "gradient", "solution", "draw"))
    }
  }
  rsr <- function(operation) function() {
    for (n in c(16L, 32L, 64L, 128L, 256L, 512L, 1024L, 2048L)) {
      x <- cbind(1, seq_len(n) / n)
      p <- diag(n) - tcrossprod(qr.Q(qr(x)))
      prior <- diag(0.4, n)
      w <- 0.5 + seq_len(n) / n; r <- cos(seq_len(n) * 0.17)
      stopifnot(nrow(p) == n, length(w) == n, length(r) == n)
      run <- function(choice, repetitions) tulpa:::cpp_device_projected_dense(p, prior, w, r,
        operation, choice, repetitions, changing_prior = operation == "gp_rsr_gibbs")
      paired(operation, 0L, n, run, c("precision", "log_density", "gradient", "draw"))
    }
  }
  hsgp <- function(action) function() {
    operation <- switch(action, product = "hsgp_product", transpose = "hsgp_transpose", crossprod = "hsgp_crossprod")
    for (m in basis_columns) for (n in c(64L, 256L, 1024L, 4096L, 16384L, 65536L)) {
      basis <- matrix(sin(seq_len(n * m) * 0.0031), n, m)
      w <- 0.5 + seq_len(n) / n
      scale <- exp(-seq_len(m) / m); beta <- cos(seq_len(m)) / sqrt(m)
      stopifnot(nrow(basis) == n, ncol(basis) == m, length(w) == n, length(beta) == m)
      run <- function(choice, repetitions) tulpa:::cpp_device_hsgp_dense(basis, w, scale, beta,
        choice, repetitions, action)
      quantities <- switch(action, product = c("field", "log_density"), transpose = "gradient", crossprod = "gram")
      paired(operation, m, n * m, run, quantities)
      rm(basis); gc(FALSE)
    }
  }
  list(
    dense_gp = dense("dense_gp", c(16L, 32L, 64L, 128L, 256L, 512L, 1024L, 2048L, 4096L)),
    dense_spec = dense("dense_spec", c(8L, 16L, 32L, 64L, 128L, 160L, 199L)),
    dense_precision = dense("dense_precision", c(2L, 8L, 16L, 32L, 64L, 128L, 256L, 512L, 1024L)),
    rsr_gibbs = rsr("rsr_gibbs"),
    gp_rsr_gibbs = rsr("gp_rsr_gibbs"),
    hsgp_product = hsgp("product"),
    hsgp_transpose = hsgp("transpose"),
    hsgp_crossprod = hsgp("crossprod"),
    hsgp_cross = function() {
      for (m in basis_columns) for (p in c(2L, m)) for (n in c(64L, 256L, 1024L, 4096L, 16384L, 65536L)) {
        left <- matrix(sin(seq_len(n * m) * 0.0031), n, m)
        right <- matrix(cos(seq_len(n * p) * 0.0017), n, p)
        stopifnot(nrow(left) == n, ncol(left) == m, nrow(right) == n, ncol(right) == p)
        run <- function(choice, repetitions) tulpa:::cpp_device_hsgp_cross(left, right, choice, repetitions)
        variant <- run("cpu", 1L)$variant
        paired("hsgp_cross", variant, n * m, run, "value")
        rm(left, right); gc(FALSE)
      }
    },
    temporal_gp = function() {
      for (tvc in c(FALSE, TRUE)) for (centered in c(TRUE, FALSE)) {
        variant <- as.integer(2L * tvc + !centered)
        for (n in c(8L, 16L, 32L, 64L, 128L, 256L, 512L, 1024L)) {
          times <- cumsum(0.6 + (sin(seq_len(n) * 0.71) + 1) / 2) / n
          q <- c(sin(seq_len(n) * 0.31) / 10, log(0.8), log(0.2))
          stopifnot(length(times) == n, length(q) == n + 2L)
          run <- function(choice, repetitions) tulpa:::cpp_device_temporal_dense(times, q,
            2L, centered, if (tvc) 0 else 1e-6, tvc, choice, repetitions)
          paired("temporal_gp", variant, n, run, c("log_density", "arena_value", "gradient"))
        }
      }
    },
    dense_adjoint = function() {
      for (n in c(8L, 16L, 32L, 64L, 128L, 256L, 512L, 1024L, 2048L)) {
        a <- outer(seq_len(n), seq_len(n), function(i, j) exp(-abs(i - j) / 4)) + diag(n)
        lower <- t(chol(a)); adjoint <- matrix(cos(seq_len(n * n)), n)
        run <- function(choice, repetitions) tulpa:::cpp_device_dense_adjoint(lower, adjoint, choice, repetitions)
        paired("dense_adjoint", 0L, n, run, "gradient")
      }
    }
  )
}
