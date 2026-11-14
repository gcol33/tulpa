skip_on_cran()

test_that("CPU and CUDA factors reconstruct each SPD matrix", {
  skip_on_cran()
  skip_if_no_cuda("cusolver")
  set.seed(1179)
  for (nn in c(10L, 15L, 30L)) {
    batch <- 64L
    matrices <- lapply(seq_len(batch), function(i) {
      x <- matrix(rnorm(nn * nn), nn)
      crossprod(x) + diag(nn) * (0.5 + i / batch)
    })
    flat <- t(vapply(matrices, as.numeric, numeric(nn * nn)))
    cpu <- tulpa:::cpp_device_cholesky(flat, nn, "cpu")
    cuda <- tulpa:::cpp_device_cholesky(flat, nn, "cuda")
    expect_true(cuda$used_cuda)
    expect_device_equal(cuda$factors, cpu$factors, 1e-12, "Cholesky factors")
    for (i in c(1L, 31L, 64L)) {
      L <- matrix(cuda$factors[i, ], nn, byrow = TRUE)
      expect_device_equal(tcrossprod(L), matrices[[i]], 1e-12, "L L'")
    }
    flat[batch, ] <- 0
    expect_error(tulpa:::cpp_device_cholesky(flat, nn, "cuda"), "batch index 63")
  }
})

test_that("NNGP moments agree across the measured size gates", {
  skip_on_cran()
  skip_if_no_cuda("cusolver")
  table <- utils::read.csv(system.file("device", "benchmarks.csv", package = "tulpa"))
  gates <- tulpa:::.device_thresholds(table)
  for (nn in c(10L, 15L, 30L)) {
    threshold <- gates$threshold[gates$operation == "nngp_cholesky" & gates$variant == nn]
    sizes <- if (length(threshold) && is.finite(threshold))
      unique(as.integer(c(max(nn + 1L, threshold - 1L), threshold, threshold + 1L))) else c(49L, 51L, 256L)
    for (batch in sizes) {
      args <- device_nngp_fixture(batch, nn)
      cpu <- do.call(tulpa:::cpp_device_nngp, c(args, list(choice = "cpu")))
      cuda <- do.call(tulpa:::cpp_device_nngp, c(args, list(choice = "cuda")))
      auto <- do.call(tulpa:::cpp_device_nngp, c(args, list(choice = "auto")))
      expect_false(cpu$used_cuda)
      expect_true(cuda$used_cuda)
      want <- length(threshold) == 1L && is.finite(threshold) && batch >= threshold
      expect_identical(auto$used_cuda, want)
      for (quantity in c("mean", "variance", "alpha")) {
        expect_device_equal(cuda[[quantity]], cpu[[quantity]], 1e-8, quantity)
        expect_device_equal(auto[[quantity]], cpu[[quantity]], 1e-8, quantity)
      }
    }
  }
})
