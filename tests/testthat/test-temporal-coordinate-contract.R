test_that("standalone temporal innovations reproduce dense precisions", {
  n <- 6L
  groups <- 2L
  set.seed(191)
  z <- rnorm(n * groups)
  log_tau <- .7
  for (structure in c("rw1", "rw2", "ar1", "iid")) {
    for (cyclic in if (structure %in% c("rw1", "rw2")) c(FALSE, TRUE) else FALSE) {
      r <- tulpa:::cpp_test_temporal_coordinates(z, n, groups, structure, cyclic, log_tau)
      D <- if (structure == "rw2") diff(diag(n), differences = 2) else diff(diag(n))
      if (cyclic) {
        D <- diag(n) - diag(n)[c(n, seq_len(n - 1L)), ]
        if (structure == "rw2") D <- D %*% D
      }
      Q <- crossprod(D) + matrix(1 / n, n, n)
      if (structure == "ar1") {
        rho <- 2 * plogis(.3) - 1
        D <- diag(n)
        D[cbind(2:n, 1:(n - 1L))] <- -rho
        D[1L, 1L] <- sqrt(1 - rho^2)
        Q <- crossprod(D)
      }
      if (structure == "iid") Q <- diag(n)
      field <- matrix(r$field, nrow = n)
      reference_quad <- sum(field * (Q %*% field)) * exp(log_tau)
      normal <- if (structure == "rw2" && !cyclic) n - 1L else n
      nc_applies <- structure %in% c("rw1", "ar1") || (structure == "rw2" && !cyclic)
      if (nc_applies) {
        expect_equal(reference_quad, sum(matrix(z, n)[seq_len(normal), ]^2), tolerance = 1e-10)
        eigenvalues <- eigen(Q, symmetric = TRUE, only.values = TRUE)$values
        log_pdet <- sum(log(eigenvalues[eigenvalues > 1e-10]))
        jacobian <- -.5 * groups * (normal * log_tau + log_pdet)
        constant <- if (structure == "ar1") 0 else .5 * groups * log_pdet
        expect_equal(r$nc, r$centered + jacobian + constant, tolerance = 1e-10)
      } else expect_equal(r$nc, r$centered)
    }
  }
})

test_that("grouped temporal, IID and restriction gradients pass the runtime gate", {
  set.seed(192)
  n <- 6L
  time <- rep(seq_len(n), 4L)
  group <- rep(rep(1:2, each = n), 2L)
  X <- cbind(1, rnorm(length(time)))
  y <- rnorm(length(time))
  basis <- qr.Q(qr(X))
  for (structure in c("rw1", "rw2", "ar1", "iid")) {
    for (coordinate in c("centered", "noncentered", "auto")) {
      spec <- list(type = structure, parameterization = coordinate,
        group_centering = TRUE, n_groups = 2L, n_times = n,
        time_idx = as.integer(time), group_idx = as.integer(group),
        restriction_basis = as.numeric(t(basis)), restriction_rank = ncol(basis))
      r <- tulpa:::cpp_test_field_coordinates(y, X, temporal_spec = spec)
      expect_true(r$verified)
      expect_equal(r$gradient, r$numerical, tolerance = 2e-5)
      expect_equal(r$prior_gradient, r$numerical_prior, tolerance = 2e-5)
      if (structure %in% c("rw1", "rw2"))
        expect_equal(colSums(matrix(tail(r$stored, 2L * n), n)), c(0, 0), tolerance = 1e-12)
    }
  }
  for (structure in c("rw1", "rw2", "ar1", "iid")) {
    tvc <- list(n_times = n, n_groups = 2L, n_tvc = 1L,
      time_index = as.integer(time), group_index = as.integer(group),
      tvc_indices = 2L, X_tvc = X[, 2L], structure = structure)
    r <- tulpa:::cpp_test_field_coordinates(y, X, tvc_spec = tvc)
    expect_true(r$verified)
    expect_equal(r$gradient, r$numerical, tolerance = 2e-5)
  }
  spec$restriction_basis[1L] <- NA_real_
  expect_error(tulpa:::cpp_test_field_coordinates(y, X, temporal_spec = spec), "finite")
})
