dense_gate_sizes <- function(operation, variant = 0L, fallback = c(7L, 33L)) {
  table <- tulpa:::.device_thresholds(utils::read.csv(system.file("device", "benchmarks.csv", package = "tulpa")))
  gate <- table$threshold[table$operation == operation & table$variant == variant]
  if (length(gate) && is.finite(gate)) unique(as.integer(c(fallback, max(1, gate - 1), gate, gate + 1)))
  else fallback
}

dense_temporal_fixture <- function(n, phi = 0.2) {
  times <- cumsum(0.6 + (sin(seq_len(n) * 0.71) + 1) / 2) / n
  list(times = times, q = c(sin(seq_len(n) * 0.31) / 10, log(0.8), log(phi)))
}

test_that("dense automatic dispatch follows every measured shape boundary", {
  skip_if_no_cuda("cusolver")
  skip_if_no_cuda("cublas")
  skip_if_no_cuda("nvrtc")
  timings <- utils::read.csv(system.file("device", "benchmarks.csv", package = "tulpa"))
  gates <- tulpa:::.device_thresholds(timings)
  gates <- gates[!gates$operation %in% c("nngp_cholesky", "axpy"), ]
  expect_true(nrow(gates) > 0L)
  for (i in seq_len(nrow(gates))) {
    g <- gates[i, ]
    sizes <- if (is.finite(g$threshold)) c(g$threshold - 1, g$threshold, g$threshold + 1) else 1e9
    for (size in sizes) {
      actual <- tulpa:::cpp_device_probe(g$operation, size, g$variant, "auto")$device
      expect_identical(actual, if (size >= g$threshold) "cuda" else "cpu",
                       info = paste(g$operation, g$variant, size))
    }
    expect_identical(tulpa:::cpp_device_probe(g$operation, 1e9, -987L, "auto")$device, "cpu")
  }
})

test_that("shared basis moments retain row weights and signed curvature", {
  n <- 37L; m <- 16L
  basis <- matrix(sin(seq_len(n * m) * .031), n, m)
  amplitude <- sin(seq_len(n) * .2)
  scale <- exp(-seq_len(m) / m); score <- cos(seq_len(n) * .3)
  design <- amplitude * sweep(basis, 2, scale, "*")
  choices <- "cpu"
  info <- gpu_info()
  if (info$available && !any(nzchar(info$library_reasons[c("cublas", "nvrtc")]))) choices <- c(choices, "cuda")
  for (w in list(seq_len(n) / n, sin(seq_len(n)))) for (choice in choices) {
    actual <- tulpa:::cpp_device_basis_moments(basis, w, score, scale, amplitude, choice)
    expect_device_equal(actual$gram, crossprod(design, w * design), 1e-10, "weighted Gram")
    expect_device_equal(drop(actual$gradient), drop(crossprod(design, score)), 1e-10, "weighted gradient")
    expect_device_equal(actual$weighted_basis, sqrt(abs(w)) * design, 1e-12, "weighted basis")
  }
})

test_that("threaded workers reuse independent dense storage", {
  cpu <- tulpa:::cpp_device_dense_threads(33L, 8L, "cpu")
  expect_true(all(is.finite(cpu)))
  skip_if_no_cuda("cusolver")
  skip_if_no_cuda("cublas")
  skip_if_no_cuda("nvrtc")
  cuda <- tulpa:::cpp_device_dense_threads(33L, 8L, "cuda")
  expect_device_equal(cuda, cpu, 1e-10, "threaded dense operations")
})

test_that("fixed-hyperparameter HSGP fits use the shared basis products", {
  skip_on_cran()
  n <- 37L; m <- 16L
  basis <- matrix(sin(seq_len(n * m) * .031), n, m)
  y <- sin(seq_len(n) * .13); x <- cbind(1, cos(seq_len(n) * .2))
  run <- function(choice, phi_basis = basis) {
    session <- tulpa:::cpp_device_begin(choice)
    on.exit(tulpa:::cpp_device_end(session))
    fit <- tulpa:::cpp_laplace_fit_hsgp(y, rep(1, n), x, numeric(), 0L, 1,
      phi_basis, seq_len(m) * .2, 0L, .8, .3, "gaussian", phi = .4, tol = 1e-10)
    list(fit = fit, device = tulpa:::cpp_device_snapshot()$stages)
  }
  cpu <- run("cpu")
  expect_true(cpu$fit$converged)
  expect_true(all(c("hsgp_product", "hsgp_crossprod") %in% cpu$device$stage))
  skip_if_no_cuda("cusolver")
  skip_if_no_cuda("cublas")
  skip_if_no_cuda("nvrtc")
  cuda <- run("cuda")
  expect_true(cuda$fit$converged)
  expect_device_equal(cuda$fit$mode, cpu$fit$mode, 1e-8, "fixed HSGP mode")
  expect_device_equal(cuda$fit$log_marginal, cpu$fit$log_marginal, 1e-9, "fixed HSGP marginal")
  for (operation in c("hsgp_product", "hsgp_crossprod", "dense_spec"))
    expect_true(any(cuda$device$stage == operation & cuda$device$cuda_calls > 0))
})

expect_dense_parameter_gradient <- function(actual, expected, tolerance, label, variance_tolerance = tolerance) {
  n <- length(expected) - 2L
  expect_device_equal(actual[seq_len(n)], expected[seq_len(n)], tolerance, paste(label, "field"))
  expect_device_equal(actual[n + 1L], expected[n + 1L], variance_tolerance, paste(label, "log variance"))
  expect_device_equal(actual[n + 2L], expected[n + 2L], tolerance, paste(label, "log lengthscale"))
}

test_that("dense factor, log determinant, solves and Gaussian gradients match R", {
  for (operation in c("dense_gp", "dense_spec", "dense_precision")) for (n in c(1L, 7L, 33L)) {
    a <- outer(seq_len(n), seq_len(n), function(i, j) exp(-abs(i - j) / 4)) + diag(0.1, n)
    rhs <- cbind(sin(seq_len(n)), cos(seq_len(n)))
    actual <- tulpa:::cpp_device_dense(a, rhs, operation, "cpu")
    expect_device_equal(actual$lower, t(chol(a)), 1e-12, paste(operation, "lower"))
    expect_device_equal(actual$solution, solve(a, rhs), 1e-12, paste(operation, "solve"))
    expect_device_equal(actual$logdet, as.numeric(determinant(a, logarithm = TRUE)$modulus), 1e-12, "logdet")
    expect_device_equal(actual$gradient, -drop(solve(a, rhs[, 1])), 1e-12, "Gaussian gradient")
  }
  expect_error(tulpa:::cpp_device_dense(matrix(c(1, 2, 2, 1), 2), matrix(1, 2),
    "dense_gp", "cpu"), "not positive definite")
})

test_that("temporal covariance pullbacks agree with independent finite differences", {
  fixture <- dense_temporal_fixture(7L)
  for (kind in 1:4) for (centered in c(FALSE, TRUE)) for (tvc in c(FALSE, TRUE)) {
    run <- function(q) tulpa:::cpp_device_temporal_dense(fixture$times, q, kind,
      centered, 1e-6, tvc, "cpu")
    actual <- run(fixture$q)
    expect_device_equal(actual$arena_value, actual$log_density, 1e-12, "arena value")
    numerical <- vapply(seq_along(fixture$q), function(j) {
      h <- 1e-5; plus <- minus <- fixture$q
      plus[j] <- plus[j] + h; minus[j] <- minus[j] - h
      (run(plus)$log_density - run(minus)$log_density) / (2 * h)
    }, numeric(1))
    expect_dense_parameter_gradient(actual$gradient, numerical, 2e-6, paste("temporal gradient", kind, centered, tvc))
  }
})

test_that("HSGP signed Gram products and coefficient pullbacks match R", {
  n <- 17L; m <- 9L
  basis <- matrix(sin(seq_len(n * m) * 0.31), n, m)
  scale <- exp(-seq_len(m) / m); beta <- cos(seq_len(m))
  for (weights in list(seq_len(n) / n, sin(seq_len(n)))) {
    actual <- tulpa:::cpp_device_hsgp_dense(basis, weights, scale, beta, "cpu")
    design <- sweep(basis, 2, scale, "*")
    field <- drop(design %*% beta)
    expect_device_equal(actual$gram, crossprod(design, weights * design), 1e-12, "HSGP Gram")
    expect_device_equal(drop(actual$field), field, 1e-12, "HSGP field")
    expect_device_equal(drop(actual$gradient), -drop(crossprod(design, weights * field)), 1e-12, "HSGP gradient")
  }
})

test_that("dense CPU and CUDA agree across their measured gates", {
  skip_if_no_cuda("cusolver")
  skip_if_no_cuda("cublas")
  skip_if_no_cuda("nvrtc")
  for (operation in c("dense_gp", "dense_spec", "dense_precision")) for (n in dense_gate_sizes(operation)) {
    a <- outer(seq_len(n), seq_len(n), function(i, j) exp(-abs(i - j) / 4)) + diag(0.1, n)
    rhs <- cbind(sin(seq_len(n)), cos(seq_len(n)))
    cpu <- tulpa:::cpp_device_dense(a, rhs, operation, "cpu")
    cuda <- tulpa:::cpp_device_dense(a, rhs, operation, "cuda")
    expect_true(cuda$used_cuda)
    for (quantity in c("lower", "solution", "logdet", "log_density", "gradient", "draw"))
      expect_device_equal(cuda[[quantity]], cpu[[quantity]], 1e-9, paste(operation, n, quantity))
  }
  expect_error(tulpa:::cpp_device_dense(matrix(c(1, 2, 2, 1), 2), matrix(1, 2),
    "dense_gp", "cuda"), "not positive definite")
})

test_that("rectangular HSGP products use their own measured shape gate", {
  choices <- "cpu"
  info <- gpu_info()
  if (info$available && !nzchar(info$library_reasons[["cublas"]])) choices <- c(choices, "cuda")
  for (m in c(16L, 64L, 144L, 225L, 256L, 400L, 900L)) for (p in c(2L, m)) {
    initial <- tulpa:::cpp_device_hsgp_cross(matrix(1, 1, m), matrix(1, 1, p), "cpu")
    sizes <- if (length(choices) > 1L) dense_gate_sizes("hsgp_cross", initial$variant, c(7L * m, 33L * m)) else c(7L * m, 33L * m)
    rows <- unique(pmax(1L, as.integer(c(floor(sizes / m), ceiling(sizes / m)))))
    for (n in rows) {
      left <- matrix(sin(seq_len(n * m) * 0.03), n, m)
      right <- matrix(cos(seq_len(n * p) * 0.07), n, p)
      expected <- crossprod(left, right)
      for (choice in choices) {
        actual <- tulpa:::cpp_device_hsgp_cross(left, right, choice)
        expect_device_equal(actual$value, expected, 1e-10, paste("rectangular cross-product", n, m, p, choice))
      }
    }
  }
})

test_that("temporal GP and TVC values and gradients agree on CUDA with binding nuggets", {
  skip_if_no_cuda("cusolver")
  skip_if_no_cuda("cublas")
  skip_if_no_cuda("nvrtc")
  for (centered in c(FALSE, TRUE)) for (tvc in c(FALSE, TRUE)) {
    variant <- as.integer(2L * tvc + !centered)
    for (n in dense_gate_sizes("temporal_gp", variant, fallback = c(7L, 20L))) for (kind in 1:4) {
      for (phi in c(0.2, 5)) {
        fixture <- dense_temporal_fixture(n, phi)
        run <- function(choice) tulpa:::cpp_device_temporal_dense(fixture$times,
          fixture$q, kind, centered, if (tvc) 0 else 1e-6, tvc, choice)
        cpu <- run("cpu"); cuda <- run("cuda")
        expect_true(cuda$used_cuda)
        tolerance <- if ((centered && tvc) || phi == 5) 3e-6 else 1e-7
        variance_tolerance <- if (centered && tvc && phi == 5) 1e-5 else tolerance
        for (quantity in c("log_density", "arena_value"))
          expect_device_equal(cuda[[quantity]], cpu[[quantity]], tolerance,
            paste("temporal", n, kind, centered, tvc, phi, quantity))
        expect_dense_parameter_gradient(cuda$gradient, cpu$gradient, tolerance,
          paste("temporal", n, kind, centered, tvc, phi), variance_tolerance)
      }
    }
  }
})

test_that("projected RSR conditionals agree with R and CUDA", {
  choices <- "cpu"
  info <- gpu_info()
  if (info$available && !any(nzchar(info$library_reasons[c("cusolver", "cublas", "nvrtc")]))) choices <- c(choices, "cuda")
  for (operation in c("rsr_gibbs", "gp_rsr_gibbs")) for (n in if (length(choices) > 1L) dense_gate_sizes(operation) else c(7L, 33L)) {
    x <- cbind(1, seq_len(n) / n)
    p <- diag(n) - tcrossprod(qr.Q(qr(x)))
    prior <- diag(0.4, n)
    w <- seq_len(n) / n; residual <- cos(seq_len(n))
    expected <- crossprod(p, w * p) + prior
    reference <- NULL
    for (choice in choices) {
      actual <- tulpa:::cpp_device_projected_dense(p, prior, w, residual, operation, choice)
      expect_device_equal(actual$precision, expected, 1e-10, "projected precision")
      expect_device_equal(drop(actual$mean), drop(solve(expected, p %*% residual)), 1e-10, "conditional mean")
      if (is.null(reference)) reference <- actual
      else for (quantity in c("logdet", "log_density", "gradient", "draw"))
        expect_device_equal(actual[[quantity]], reference[[quantity]], 1e-9, paste(operation, n, quantity))
    }
  }
})

test_that("HSGP products agree on both sides of each basis-size gate", {
  skip_if_no_cuda("cublas")
  skip_if_no_cuda("nvrtc")
  for (m in c(16L, 64L, 144L, 225L, 256L, 400L, 900L)) {
    sizes <- unique(c(dense_gate_sizes("hsgp_product", m), dense_gate_sizes("hsgp_transpose", m), dense_gate_sizes("hsgp_crossprod", m)))
    rows <- unique(pmax(1L, as.integer(c(floor(sizes / m), ceiling(sizes / m)))))
    for (n in rows) for (signed in c(FALSE, TRUE)) {
      basis <- matrix(sin(seq_len(n * m) * 0.0031), n, m)
      scale <- exp(-seq_len(m) / m); beta <- cos(seq_len(m)) / sqrt(m)
      w <- if (signed) sin(seq_len(n)) else seq_len(n) / n
      run <- function(choice) tulpa:::cpp_device_hsgp_dense(basis, w, scale, beta, choice)
      cpu <- run("cpu"); cuda <- run("cuda")
      for (quantity in c("gram", "field", "log_density", "gradient"))
        expect_device_equal(cuda[[quantity]], cpu[[quantity]], 1e-9, paste("HSGP", n, m, signed, quantity))
    }
  }
})

test_that("HSGP sampler gradients include spectral scale and range on CUDA", {
  basis <- matrix(sin(seq_len(37L * 16L) * 0.031), 37L, 16L)
  eigenvalues <- seq_len(16L) * 0.2
  q <- c(cos(seq_len(16L)) / 4, log(0.8), log(0.3))
  choices <- "cpu"
  info <- gpu_info()
  if (info$available && !any(nzchar(info$library_reasons[c("cublas", "nvrtc")]))) choices <- c(choices, "cuda")
  for (kernel in 0:2) {
    run <- function(q, choice = "cpu") tulpa:::cpp_device_hsgp_sampler(basis, eigenvalues, q, kernel, choice)
    cpu <- run(q)
    numerical <- vapply(seq_along(q), function(j) {
      h <- 1e-5; plus <- minus <- q; plus[j] <- plus[j] + h; minus[j] <- minus[j] - h
      (run(plus)$log_density - run(minus)$log_density) / (2 * h)
    }, numeric(1))
    expect_dense_parameter_gradient(cpu$gradient, numerical, 1e-7, paste("HSGP spectral gradient", kernel))
    for (choice in choices) {
      actual <- run(q, choice)
      for (quantity in c("log_density", "arena_value"))
        expect_device_equal(actual[[quantity]], cpu[[quantity]], 1e-10, paste("HSGP sampler", kernel, quantity))
      expect_dense_parameter_gradient(actual$gradient, cpu$gradient, 1e-10, paste("HSGP sampler", kernel))
    }
  }
})

test_that("the standardized long-lengthscale fixture retains its binding jitter on CUDA", {
  skip_if_no_cuda("cusolver")
  skip_if_no_cuda("cublas")
  skip_if_no_cuda("nvrtc")
  set.seed(851)
  times <- as.numeric(scale(cumsum(rexp(24L))))
  q <- c(sin(seq_len(24L) * 0.31) / 10, log(0.8), log(5.005))
  for (kind in c(2L, 3L)) for (relative in c(0, 1e-6)) {
    distance <- abs(outer(times, times, "-"))
    r <- distance / 5.005
    k <- if (kind == 3L) 0.8 * exp(-r * r) else 0.8 * (1 + sqrt(5) * r + 5 * r * r / 3) * exp(-sqrt(5) * r)
    nugget <- if (relative > 0) 0.8 * relative else 1e-8
    expect_lt(min(eigen(k, symmetric = TRUE, only.values = TRUE)$values), nugget)
    for (centered in c(FALSE, TRUE)) {
      run <- function(choice) tulpa:::cpp_device_temporal_dense(times, q, kind, centered, relative, FALSE, choice)
      cpu <- run("cpu"); cuda <- run("cuda")
      expect_device_equal(cuda$log_density, cpu$log_density, 3e-6, "binding jitter log density")
      expect_dense_parameter_gradient(cuda$gradient, cpu$gradient, 3e-6, "binding jitter gradient")
    }
  }
})

test_that("Cholesky adjoints agree with finite differences and CUDA across the gate", {
  choices <- "cpu"
  info <- gpu_info()
  if (info$available && !any(nzchar(info$library_reasons[c("cublas", "nvrtc")]))) choices <- c(choices, "cuda")
  for (n in if (length(choices) > 1L) dense_gate_sizes("dense_adjoint") else c(7L, 33L)) {
    a <- outer(seq_len(n), seq_len(n), function(i, j) exp(-abs(i - j) / 4)) + diag(n)
    lower <- t(chol(a)); adjoint <- matrix(cos(seq_len(n * n)), n)
    direction <- outer(seq_len(n), seq_len(n), function(i, j) sin(i + j))
    h <- 1e-5
    numerical <- sum((t(chol(a + h * direction)) - t(chol(a - h * direction))) * adjoint) / (2 * h)
    reference <- NULL
    for (choice in choices) {
      actual <- tulpa:::cpp_device_dense_adjoint(lower, adjoint, choice)$gradient
      expect_equal(sum(actual * direction), numerical, tolerance = 2e-7)
      if (is.null(reference)) reference <- actual
      else expect_device_equal(actual, reference, 1e-9, paste("Cholesky adjoint", n))
    }
  }
})
