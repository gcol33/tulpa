# CUDA routes come from tulpaGPU; a CUDA test runs only when it is installed
# and its device and the named libraries are usable.
cuda_unavailable <- function(capability = "driver") {
  if (!requireNamespace("tulpaGPU", quietly = TRUE)) return("tulpaGPU is not installed")
  info <- tulpaGPU::gpu_info()
  if (!isTRUE(info$available)) return(paste("CUDA unavailable:", info$reason))
  for (required in capability) {
    reason <- info$library_reasons[[required]]
    if (is.null(reason)) stop("unknown CUDA capability")
    if (nzchar(reason)) return(paste("CUDA", required, "unavailable:", reason))
  }
  NULL
}

skip_if_no_cuda <- function(capability = "driver") {
  reason <- cuda_unavailable(capability)
  if (!is.null(reason)) testthat::skip(reason)
  invisible(NULL)
}

expect_device_equal <- function(actual, expected, tolerance, quantity) {
  testthat::expect_identical(length(actual), length(expected), info = quantity)
  testthat::expect_identical(dim(actual), dim(expected), info = quantity)
  if (length(actual) != length(expected) || !identical(dim(actual), dim(expected)))
    return(invisible(NULL))
  values <- as.numeric(actual); reference <- as.numeric(expected)
  finite <- is.finite(reference)
  testthat::expect_identical(is.finite(values), finite, info = quantity)
  if (!identical(is.finite(values), finite)) return(invisible(NULL))
  testthat::expect_identical(values[!finite], reference[!finite], info = quantity)
  scale <- max(abs(reference[finite]), 1)
  error <- max(c(0, abs(values[finite] - reference[finite])))
  output <- Sys.getenv("TULPA_DEVICE_EQUIVALENCE", "")
  if (nzchar(output)) {
    row <- data.frame(quantity = quantity, scale = scale,
      absolute_error = error, relative_tolerance = tolerance)
    utils::write.table(row, output, sep = ",", append = file.exists(output),
      row.names = FALSE, col.names = !file.exists(output), quote = TRUE)
  }
  testthat::expect_true(error <= tolerance * scale,
    info = sprintf("%s: reference scale %.17g, relative tolerance %.3g",
                   quantity, scale, tolerance))
}
