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

# A crossover is the first measured size whose fastest CUDA median and every
# larger measured median beat CPU. Batch preference has its own crossover.
.device_thresholds <- function(timings) {
  required <- c("operation", "variant", "size", "cpu_seconds", "cuda_seconds")
  if (!all(required %in% names(timings))) stop("invalid device timing columns")
  if ('scope' %in% names(timings))
    timings <- timings[is.na(timings$scope) | timings$scope != 'fit_auto', , drop=FALSE]
  out <- data.frame(operation = character(), variant = integer(), threshold = numeric(),
                    batch_threshold = numeric())
  if (!nrow(timings)) return(out)
  if (anyNA(timings[required]) || any(!is.finite(as.matrix(timings[required[-1L]]))))
    stop("device timings must be finite and complete")
  if (any(timings$size <= 0 | timings$cpu_seconds <= 0 | timings$cuda_seconds <= 0))
    stop("device timing sizes and durations must be positive")
  if (!"stream_seconds" %in% names(timings)) timings$stream_seconds <- Inf
  if (anyNA(timings$stream_seconds) || any(timings$stream_seconds <= 0))
    stop("device stream timings must be positive or infinite")
  if (!"calibration" %in% names(timings)) timings$calibration <- 0L
  if ('model' %in% names(timings)) timings$model <- as.character(timings$model)
  if (anyNA(timings$calibration) || any(!is.finite(timings$calibration)) ||
      any(timings$calibration < 0 | timings$calibration != floor(timings$calibration)))
    stop("device calibration identifiers must be non-negative integers")
  groups <- split(timings, interaction(timings$operation, timings$variant, drop = TRUE))
  for (g in groups) {
    g <- g[g$calibration == max(g$calibration), , drop = FALSE]
    sizes <- sort(unique(g$size))
    by_model <- 'model' %in% names(g) && any(!is.na(g$model) & nzchar(g$model))
    if (by_model && (anyNA(g$model) || any(!nzchar(g$model))))
      stop('device fit calibration must name every model')
    costs <- vapply(sizes, function(s) {
      rows <- g[g$size == s, , drop = FALSE]
      medians <- function(x) c(cpu=stats::median(x$cpu_seconds), batch=stats::median(x$cuda_seconds),
        stream=stats::median(x$stream_seconds))
      if (!by_model) return(medians(rows))
      if (!setequal(unique(rows$model), unique(g$model)))
        stop('device fit calibration must cover every model at every size')
      models <- vapply(split(rows, rows$model), medians, c(cpu=0,batch=0,stream=0))
      c(cpu=1, batch=max(models['batch',] / models['cpu',]),
        stream=max(models['stream',] / models['cpu',]))
    }, c(cpu=0,batch=0,stream=0))
    crossover <- function(wins) {
      sustained <- rev(cumprod(rev(as.integer(wins)))) == 1L
      if (any(sustained)) min(sizes[sustained]) else Inf
    }
    threshold <- crossover(pmin(costs['batch',],costs['stream',]) < costs['cpu',])
    batch_threshold <- if (all(!is.finite(costs['stream',]))) 0 else
      crossover(costs['batch',] < costs['stream',])
    out <- rbind(out, data.frame(operation = g$operation[1L],
                                variant = as.integer(g$variant[1L]), threshold = threshold,
                                batch_threshold = batch_threshold))
  }
  rownames(out) <- NULL
  out
}

.device_kernel_sources <- function(libname, pkgname) {
  root <- system.file("device", package = pkgname, lib.loc = libname)
  files <- list.files(root, pattern = "\\.cu$", full.names = TRUE)
  sources <- lapply(files, function(file) paste(readLines(file, warn = FALSE), collapse = "\n"))
  names(sources) <- tools::file_path_sans_ext(basename(files))
  headers <- c("device_math.h", "precision_coefficients.h", "precision_math.h",
               "count_math.h", "likelihood_math.h")
  include <- system.file("include", "tulpa", package = pkgname, lib.loc = libname)
  for (header in headers) sources[[paste0("tulpa/", header)]] <-
    paste(readLines(file.path(include, header), warn = FALSE), collapse = "\n")
  sources
}

.device_configure <- function(libname, pkgname) {
  root <- system.file("device", package = pkgname, lib.loc = libname)
  timings <- utils::read.csv(file.path(root, "benchmarks.csv"), stringsAsFactors = FALSE)
  sources <- .device_kernel_sources(libname, pkgname)
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
