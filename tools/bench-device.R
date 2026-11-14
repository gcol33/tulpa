args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "inst/device/benchmarks.csv"
library(tulpa)
info <- gpu_info()
if (!info$available) stop(info$reason)
source("tests/testthat/helper-device.R")
ladder <- c(50L, 100L, 250L, 500L, 1000L, 2500L, 5000L, 10000L,
            25000L, 50000L, 100000L)
rows <- list()
append_row <- function(operation, variant, size, cpu, cuda) {
  stopifnot(is.finite(cpu), cpu > 0, is.finite(cuda), cuda > 0)
  rows[[length(rows) + 1L]] <<- data.frame(operation = operation,
    variant = variant, size = size, cpu_seconds = cpu, cuda_seconds = cuda)
  utils::write.csv(do.call(rbind, rows), output, row.names = FALSE)
  cat(sprintf("%s variant=%d size=%d CPU=%.6fs CUDA=%.6fs\n",
              operation, variant, size, cpu, cuda))
  flush.console()
}
runners <- list(
  nngp_cholesky = function() {
    for (nn in c(10L, 15L, 30L)) for (batch in ladder) {
      fixture <- device_nngp_fixture(batch, nn)
      stopifnot(nrow(fixture$coords) == batch + 1L,
                ncol(fixture$nn_idx) == nn,
                sum(rowSums(fixture$nn_idx > 0) > 0) == batch)
      run <- function(choice, repetitions = 1L) do.call(tulpa:::cpp_device_nngp,
        c(fixture, list(choice = choice, repetitions = repetitions)))
      cpu <- run("cpu")
      cuda <- run("cuda")
      stopifnot(cuda$used_cuda, !cpu$used_cuda)
      for (q in c("mean", "variance", "alpha")) {
        scale <- max(1, abs(cpu[[q]]))
        stopifnot(max(abs(cpu[[q]] - cuda[[q]])) <= 1e-8 * scale)
      }
      repetitions <- max(1L, min(20L, as.integer(ceiling(0.1 / min(cpu$seconds, cuda$seconds)))))
      for (trial in seq_len(3L)) {
        order <- if (trial %% 2L) c("cpu", "cuda") else c("cuda", "cpu")
        measured <- lapply(order, function(choice) run(choice, repetitions)$seconds)
        names(measured) <- order
        append_row("nngp_cholesky", nn, batch, measured$cpu, measured$cuda)
      }
      rm(fixture, cpu, cuda)
      gc(FALSE)
    }
  },
  axpy = function() {
    for (size in as.integer(10^(2:7))) {
      x <- sin(seq_len(size) * 0.001)
      y <- cos(seq_len(size) * 0.001)
      run <- function(choice, repetitions = 1L) tulpa:::cpp_device_axpy(x, y, 0.13, choice, repetitions)
      cpu <- run("cpu"); cuda <- run("cuda")
      stopifnot(cuda$used_cuda, max(abs(cpu$value - cuda$value)) < 1e-13)
      repetitions <- max(1L, min(1000L, as.integer(ceiling(0.05 / min(cpu$seconds, cuda$seconds)))))
      for (trial in seq_len(3L)) {
        measured <- numeric(2L); names(measured) <- c("cpu", "cuda")
        order <- if (trial %% 2L) names(measured) else rev(names(measured))
        for (choice in order) {
          measured[[choice]] <- run(choice, repetitions)$seconds
        }
        append_row("axpy", 0L, size, measured[["cpu"]], measured[["cuda"]])
      }
    }
  })
registered <- tulpa:::cpp_device_operations()
stopifnot(setequal(names(runners), registered))
dir.create(dirname(output), showWarnings = FALSE, recursive = TRUE)
cat(sprintf("Device: %s; driver=%d; threads=%s\n", info$name,
            info$driver_version, Sys.getenv("OMP_NUM_THREADS", "default")))
for (operation in registered) runners[[operation]]()
print(tulpa:::.device_thresholds(do.call(rbind, rows)))
writeLines(c(capture.output(print(info)), capture.output(sessionInfo())),
           paste0(output, ".session.txt"))
