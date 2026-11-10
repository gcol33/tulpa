test_that("restrictions normalize exactly removed RW2 slopes", {
  n <- 6L
  linear <- seq_len(n) - (n + 1) / 2
  norm <- sqrt(sum(linear^2))
  unit <- linear / norm
  for (groups in 1:2) for (multiscale in c(FALSE, TRUE)) {
    if (multiscale && groups > 1L) next
    time <- rep(seq_len(n), 4L * groups)
    group <- rep(rep(seq_len(groups), each = n), 4L)
    X <- cbind(1, linear[time])
    basis <- qr.Q(qr(X))
    for (coordinate in c("centered", "noncentered")) {
      spec <- list(type = "rw2", parameterization = coordinate,
        group_centering = TRUE, n_groups = groups, n_times = n,
        time_idx = as.integer(time), group_idx = as.integer(group),
        restriction_basis = as.numeric(t(basis)), restriction_rank = ncol(basis))
      if (multiscale) {
        spec$type <- "multiscale"
        spec$time_index <- as.integer(time)
        spec$trend <- "rw2"
        spec$seasonal <- 0L
        spec$short_term <- "none"
      }
      evaluate <- function(point = NULL) tulpa:::cpp_test_field_coordinates(
        rep(0, length(time)), X, temporal_spec = spec, point = point)
      initial <- evaluate()
      q <- numeric(length(initial$q))
      center <- evaluate(q)
      offset <- length(q) - n * groups
      direction <- numeric(length(q))
      for (g in seq_len(groups)) {
        block <- offset + (g - 1L) * n + seq_len(n)
        if (coordinate == "centered") direction[block] <- unit / sqrt(groups)
        else if (multiscale) direction[block[2L]] <- 1
        else direction[block[n]] <- 1 / (norm * sqrt(groups))
      }
      plus <- evaluate(q + .7 * direction)
      minus <- evaluate(q - .7 * direction)
      expect_equal(plus$raw_lp + minus$raw_lp - 2 * center$raw_lp, -.7^2, tolerance = 1e-9)
      expect_equal(tail(plus$stored, n * groups), numeric(n * groups), tolerance = 1e-12)
      expect_true(plus$verified)
      expect_equal(plus$gradient, plus$numerical, tolerance = 2e-5)
      if (groups == 2L) {
        contrast <- direction
        contrast[offset + n + seq_len(n)] <- -contrast[offset + n + seq_len(n)]
        retained <- evaluate(q + .7 * contrast)
        field <- matrix(tail(retained$stored, n * groups), n)
        expect_equal(drop(crossprod(unit, field)), c(.7, -.7) / sqrt(2), tolerance = 1e-12)
      }
    }
  }
})

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
      nc_applies <- structure %in% c("rw1", "rw2", "ar1")
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

test_that("global intrinsic innovations retain the flat group contrasts", {
  n <- 6L
  groups <- 2L
  z <- sin(seq_len(n * groups))
  for (structure in c("rw1", "rw2")) {
    order <- if (structure == "rw1") 1L else 2L
    D <- diff(diag(n), differences = order)
    Q <- kronecker(diag(groups), crossprod(D)) + matrix(1 / (n * groups), n * groups, n * groups)
    r <- tulpa:::cpp_test_temporal_coordinates(z, n, groups, structure, FALSE,
      .7, group_centering = FALSE)
    zz <- matrix(z, n)
    expected_quad <- sum(zz[seq_len(n - order), ]^2) + sum(zz[n - order + 1L, ])^2 / groups
    expect_equal(exp(.7) * drop(crossprod(r$field, Q %*% r$field)), expected_quad,
      tolerance = 1e-10)
    rank <- groups * (n - order) + 1L
    expect_equal(r$nc - r$centered, -.5 * rank * .7, tolerance = 1e-10)
  }
})
