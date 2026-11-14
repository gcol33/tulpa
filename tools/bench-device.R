args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "inst/device/benchmarks.csv"
library(tulpa)
info <- gpu_info()
if (!info$available) stop(info$reason)
source("tests/testthat/helper-device.R")
ladder <- c(50L, 100L, 250L, 500L, 1000L, 2500L, 5000L, 10000L,
            25000L, 50000L, 100000L)
rows <- list()
if (file.exists(output)) {
  existing <- utils::read.csv(output)
  rows <- split(existing, seq_len(nrow(existing)))
}
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
  sparse_cholesky = function() {
    source('tests/testthat/helper-device-sparse.R')
    stages <- list()
    for (size in c(1000L, 3000L, 10000L, 30000L, 100000L)) {
      fixture <- device_sparse_fixture(size, 'spde')
      stopifnot(nrow(fixture$matrix) == size, nrow(fixture$b) == size)
      cat(sprintf('SPDE mesh: %d nodes, %d stored lower entries\n', size, length(fixture$i)))
      run <- function(choice) device_sparse_run(fixture, choice)
      cpu <- run('cpu'); cuda <- run('cuda')
      stopifnot(cpu$ok, cuda$ok, cuda$used_cuda, !cpu$used_cuda)
      for (q in c('solution','logdet','diagonal')) {
        scale <- max(1, abs(cpu[[q]]))
        cat(sprintf('%s scale=%.17g relative tolerance=1e-8\n', q, scale))
        stopifnot(max(abs(cpu[[q]] - cuda[[q]])) <= 1e-8 * scale)
      }
      for (trial in seq_len(3L)) {
        order <- if (trial %% 2L) c('cpu','cuda') else c('cuda','cpu')
        measured <- lapply(order, run); names(measured) <- order
        append_row('sparse_cholesky', 0L, size, measured$cpu$seconds, measured$cuda$seconds)
        for (choice in order) {
          r <- measured[[choice]]
          stages[[length(stages) + 1L]] <- data.frame(size = size, trial = trial,
            device = choice, analysis = r$analysis_seconds, factor = r$factor_seconds,
            solve = r$solve_seconds, selected_inverse = r$inverse_seconds, total = r$seconds)
        }
        utils::write.csv(do.call(rbind, stages), paste0(output, '.sparse-stages.csv'), row.names = FALSE)
      }
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
source("tools/bench-device-likelihood.R")
runners$likelihood <- function() device_likelihood_benchmark(append_row,
  detail_file = paste0(output, ".likelihood.csv"))
runners$likelihood_weights <- function() device_likelihood_benchmark(append_row, weights = TRUE,
  detail_file = paste0(output, ".likelihood-weights.csv"))
stopifnot(setequal(names(runners), registered))
dir.create(dirname(output), showWarnings = FALSE, recursive = TRUE)
cat(sprintf("Device: %s; driver=%d; threads=%s\n", info$name,
            info$driver_version, Sys.getenv("OMP_NUM_THREADS", "default")))
selected <- if (length(args) >= 2L) strsplit(args[[2L]], ',', fixed = TRUE)[[1L]] else registered
stopifnot(all(selected %in% registered))
for (operation in selected) runners[[operation]]()
print(tulpa:::.device_thresholds(do.call(rbind, rows)))
writeLines(c(capture.output(print(info)), capture.output(sessionInfo())),
           paste0(output, ".session.txt"))
