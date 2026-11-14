#' CUDA device availability
#'
#' Report the CUDA driver, the selected NVIDIA device, and the runtime
#' libraries used by the engine. Installation requires a C++ compiler only.
#' Device operations load CUDA libraries when first requested.
#'
#' @return An object of class `tulpa_gpu_info` with availability, the reason
#'   for unavailability, device properties, and library paths and reasons.
#' @examples
#' gpu_info()
#' @export
gpu_info <- function() {
  structure(cpp_device_info(), class = "tulpa_gpu_info")
}

#' @rdname gpu_info
#' @param x A `gpu_info()` result.
#' @param ... Ignored.
#' @export
print.tulpa_gpu_info <- function(x, ...) {
  if (isTRUE(x$available)) {
    cat(sprintf("CUDA: %s, %.0f MiB, compute capability %s\n",
                x$name, x$memory_mb, x$compute_capability))
  } else cat("CUDA unavailable: ", x$reason, "\n", sep = "")
  invisible(x)
}

.device_choice <- function(control) cpp_device_validate(control$device %||% "auto")

.device_start <- function(control) cpp_device_begin(.device_choice(control))

.device_attach <- function(fit) {
  fit$device <- cpp_device_snapshot()
  fit
}

# A crossover is the first measured size whose CUDA median and every larger
# measured median beat CPU. Each operation and matrix order has its own gate.
.device_thresholds <- function(timings) {
  required <- c("operation", "variant", "size", "cpu_seconds", "cuda_seconds")
  if (!all(required %in% names(timings))) stop("invalid device timing columns")
  out <- data.frame(operation = character(), variant = integer(), threshold = numeric())
  if (!nrow(timings)) return(out)
  if (anyNA(timings[required]) || any(!is.finite(as.matrix(timings[required[-1L]]))))
    stop("device timings must be finite and complete")
  if (any(timings$size <= 0 | timings$cpu_seconds <= 0 | timings$cuda_seconds <= 0))
    stop("device timing sizes and durations must be positive")
  groups <- split(timings, interaction(timings$operation, timings$variant, drop = TRUE))
  for (g in groups) {
    sizes <- sort(unique(g$size))
    wins <- vapply(sizes, function(s) {
      rows <- g[g$size == s, , drop = FALSE]
      stats::median(rows$cuda_seconds) < stats::median(rows$cpu_seconds)
    }, logical(1))
    sustained <- rev(cumprod(rev(as.integer(wins)))) == 1L
    threshold <- if (any(sustained)) min(sizes[sustained]) else Inf
    out <- rbind(out, data.frame(operation = g$operation[1L],
                                variant = as.integer(g$variant[1L]), threshold = threshold))
  }
  rownames(out) <- NULL
  out
}

.device_configure <- function(libname, pkgname) {
  root <- system.file("device", package = pkgname, lib.loc = libname)
  timings <- utils::read.csv(file.path(root, "benchmarks.csv"), stringsAsFactors = FALSE)
  files <- list.files(root, pattern = "\\.cu$", full.names = TRUE)
  sources <- lapply(files, function(file) paste(readLines(file, warn = FALSE), collapse = "\n"))
  names(sources) <- tools::file_path_sans_ext(basename(files))
  headers <- c("device_math.h", "precision_coefficients.h", "precision_math.h",
               "count_math.h", "likelihood_math.h")
  include <- system.file("include", "tulpa", package = pkgname, lib.loc = libname)
  for (header in headers) sources[[paste0("tulpa/", header)]] <-
    paste(readLines(file.path(include, header), warn = FALSE), collapse = "\n")
  cpp_device_configure(.device_thresholds(timings), sources)
  invisible(NULL)
}

.print_device <- function(device) {
  stages <- device$stages
  if (!is.null(stages) && nrow(stages)) {
    cat("Device: ", paste(paste0(stages$stage, "=", stages$device), collapse = ", "), "\n", sep = "")
  }
  invisible(NULL)
}
