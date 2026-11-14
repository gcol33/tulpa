skip_if_no_cuda <- function(capability = "driver") {
  info <- tulpa::gpu_info()
  if (!isTRUE(info$available)) testthat::skip(paste("CUDA unavailable:", info$reason))
  reason <- info$library_reasons[[capability]]
  if (is.null(reason)) stop("unknown CUDA capability")
  if (nzchar(reason)) testthat::skip(paste("CUDA", capability, "unavailable:", reason))
  invisible(info)
}

device_nngp_fixture <- function(batch, nn) {
  n <- as.integer(batch + 1L)
  i <- seq_len(n)
  coords <- cbind(i * 0.02, sin(i * 0.73) * 0.1)
  idx <- matrix(0L, n, nn)
  distances <- matrix(0, n, nn)
  for (k in seq_len(nn)) {
    rows <- seq.int(k + 1L, n)
    idx[rows, k] <- rows - k
    distances[rows, k] <- sqrt(rowSums((coords[rows, , drop = FALSE] -
                                       coords[rows - k, , drop = FALSE])^2))
  }
  list(w = sin(i * 0.33), coords = coords, nn_idx = idx, nn_dist = distances,
       nn_order = seq.int(0L, n - 1L), sigma2 = 0.9, phi = 0.4, cov_type = 0L)
}

expect_device_equal <- function(actual, expected, tolerance, quantity) {
  testthat::expect_identical(length(actual), length(expected), info = quantity)
  testthat::expect_identical(dim(actual), dim(expected), info = quantity)
  if (length(actual) != length(expected) || !identical(dim(actual), dim(expected)))
    return(invisible(NULL))
  scale <- max(abs(expected), 1)
  testthat::expect_true(max(abs(actual - expected)) <= tolerance * scale,
    info = sprintf("%s: reference scale %.17g, relative tolerance %.3g",
                   quantity, scale, tolerance))
}
