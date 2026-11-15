args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "inst/device/benchmarks.csv"
dir.create(dirname(output),showWarnings=FALSE,recursive=TRUE)
library(tulpa)
library_md5 <- unname(tools::md5sum(getLoadedDLLs()[['tulpa']][['path']]))
info <- gpu_info()
if (!info$available) stop(info$reason)
source("tests/testthat/helper-device.R")
source('tools/bench-device-fits.R')
device_wait_for_gpu(paste0(output,'.gpu.txt'))
ladder <- c(50L, 100L, 250L, 500L, 1000L, 2500L, 5000L, 10000L,
            25000L, 50000L, 100000L)
rows <- list()
batch_rows <- list()
calibration <- 1L
if (file.exists(output)) {
  existing <- utils::read.csv(output)
  if (!'stream_seconds' %in% names(existing)) existing$stream_seconds <- Inf
  if (!'library_md5' %in% names(existing)) existing$library_md5 <- NA_character_
  if (!'calibration' %in% names(existing)) existing$calibration <- 0L
  calibration <- max(existing$calibration) + 1L
  rows <- split(existing, seq_len(nrow(existing)))
}
append_row <- function(operation, variant, size, cpu, cuda, stream = Inf, extra = list()) {
  if (length(device_gpu_processes()$busy)) stop('GPU contention invalidated this benchmark sample')
  if (identical(extra$scope,'fit_auto')) {
    stopifnot(is.na(cpu),is.na(cuda),is.finite(extra$auto_seconds),extra$auto_seconds>0)
  } else stopifnot(is.finite(cpu), cpu > 0, is.finite(cuda), cuda > 0)
  row <- data.frame(operation = operation,
    variant = variant, size = size, cpu_seconds = cpu, cuda_seconds = cuda, stream_seconds = stream,
    library_md5 = library_md5, calibration = calibration)
  for (name in names(extra)) row[[name]] <- extra[[name]]
  columns <- unique(c(unlist(lapply(rows, names)), names(row)))
  complete <- function(x) {
    for (name in setdiff(columns, names(x))) x[[name]] <- NA
    x[columns]
  }
  rows <<- lapply(c(rows, list(row)), complete)
  utils::write.csv(do.call(rbind, rows), output, row.names = FALSE)
  cat(sprintf("%s variant=%d size=%d CPU=%.6fs CUDA=%.6fs\n",
              operation, variant, size, cpu, cuda))
  flush.console()
}
check_device_run <- function(result) {
  stopifnot(all(result$errors==''),!any(result$recovered))
  if(!is.null(result$positive))stopifnot(all(result$positive))
}
measure_batch <- function(operation, variant, size, run, quantities=c('solution','logdet')) {
  cpu <- run('cpu'); cuda <- run('cuda')
  stream_warm <- run('cuda',FALSE)
  for(z in list(cpu,cuda,stream_warm))check_device_run(z)
  for (q in quantities) {
    for(z in list(cuda,stream_warm))stopifnot(length(cpu[[q]])>0L,
      length(cpu[[q]])==length(z[[q]]),identical(dim(cpu[[q]]),dim(z[[q]])),
      all(is.finite(cpu[[q]])),all(is.finite(z[[q]])))
    if(q=='diagonal')for(z in list(cpu,cuda,stream_warm))stopifnot(all(z[[q]]>0))
    scale <- max(1,abs(cpu[[q]]))
    cat(sprintf('%s variant=%d size=%d %s scale=%.17g tolerance=1e-8\n',operation,variant,size,q,scale))
    stopifnot(max(c(0,abs(cpu[[q]]-cuda[[q]]))) <= 1e-8*scale)
    stopifnot(max(c(0,abs(cpu[[q]]-stream_warm[[q]]))) <= 1e-8*scale)
  }
  for (trial in seq_len(3L)) {
    order <- if(trial %% 2L) c('cpu','cuda') else c('cuda','cpu')
    measured <- lapply(order,run); names(measured) <- order
    streams <- run('cuda',FALSE)
    for(z in c(measured,list(streams)))check_device_run(z)
    append_row(operation,variant,size,measured$cpu$seconds,measured$cuda$seconds,streams$seconds)
    batch_rows[[length(batch_rows)+1L]] <<- data.frame(operation=operation,variant=variant,
      size=size,trial=trial,cpu=measured$cpu$seconds,cuda_streams=streams$seconds,
      cuda_batched=measured$cuda$seconds)
    utils::write.csv(do.call(rbind,batch_rows),paste0(output,'.batch-stages.csv'),row.names=FALSE)
  }
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
  sparse = function() {
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
        append_row('sparse_selected_inverse', 0L, size,
          measured$cpu$factor_seconds + measured$cpu$inverse_seconds, measured$cuda$inverse_seconds)
        for (choice in order) {
          r <- measured[[choice]]
          stages[[length(stages) + 1L]] <- data.frame(size = size, trial = trial,
            device = choice, analysis = r$analysis_seconds, factor = r$factor_seconds,
            solve = r$solve_seconds, selected_inverse = r$inverse_seconds, total = r$seconds)
        }
        utils::write.csv(do.call(rbind, stages), paste0(output, '.sparse-stages.csv'), row.names = FALSE)
      }
    }
    source('tools/bench-device-fits.R')
    device_fit_benchmark(append_row, output)
  },
  laplace_dense = function() {
    source('tests/testthat/helper-device-laplace-batch.R')
    for (width in c(1L,4L,8L)) for (size in c(8L,32L,74L,128L,202L,256L,512L,1024L)) {
      fixture <- device_laplace_fixture(size,128L)
      measure_batch('laplace_dense',width,size,function(choice,batched=TRUE)
        device_laplace_run(fixture,choice,width,batched))
      csc <- device_full_csc_fixture(fixture)
      measure_batch('laplace_full_csc',width,size,function(choice,batched=TRUE)
        do.call(tulpa:::cpp_device_sparse_batch,c(csc,list(choice=choice,workers=width,
          batched=batched,selected_inverse=FALSE))))
      rm(fixture,csc);gc(FALSE)
    }
  },
  laplace_dense_inverse = function() {
    source('tests/testthat/helper-device-laplace-batch.R')
    for(width in c(1L,4L,8L))for(size in c(31L,74L,128L,202L,252L,256L,512L)) {
      fixture <- device_full_csc_fixture(device_laplace_fixture(size,16L))
      fixture$b <- matrix(numeric(),size,0L)
      measure_batch('laplace_dense_inverse',width,size,function(choice,batched=TRUE)
        do.call(tulpa:::cpp_device_sparse_batch,c(fixture,list(choice=choice,workers=width,
          batched=batched,selected_inverse=TRUE))),c('logdet','diagonal'))
    }
  },
  laplace_scatter = function() {
    source('tests/testthat/helper-device-laplace-batch.R')
    stages <- list()
    for (width in c(1L,4L,8L)) for (n in c(120L,250L,1000L))
      for (p in c(8L,32L,74L,128L,202L,256L,512L)) {
        fixture <- device_scatter_fixture(n,p,128L)
        run <- function(choice,batched=TRUE)device_scatter_run(fixture,choice,width,batched)
        cpu <- run('cpu');cuda <- run('cuda')
        stream_warm <- run('cuda',FALSE)
        for(z in list(cpu,cuda,stream_warm))check_device_run(z)
        for(q in c('gradient','hessian')) {
          scale <- max(1,abs(cpu[[q]]));error <- max(abs(cpu[[q]]-cuda[[q]]))
          cat(sprintf('scatter rows=%d columns=%d width=%d %s scale=%.17g error=%.17g tolerance=1e-9\n',
                      n,p,width,q,scale,error))
          stopifnot(error<=1e-9*scale)
          stopifnot(max(abs(cpu[[q]]-stream_warm[[q]]))<=1e-9*scale)
        }
        for(trial in seq_len(3L)) {
          order <- if(trial%%2L)c('cpu','cuda')else c('cuda','cpu')
          measured <- lapply(order,run);names(measured)<-order
          streams <- run('cuda',FALSE)
          for(z in c(measured,list(streams)))check_device_run(z)
          append_row('laplace_scatter',width,as.double(n)*p*p,
                     measured$cpu$seconds,measured$cuda$seconds,streams$seconds)
          stages[[length(stages)+1L]] <- data.frame(rows=n,columns=p,width=width,trial=trial,
            cpu=measured$cpu$seconds,streams=streams$seconds,batch=measured$cuda$seconds,
            launches=sum(measured$cuda$stages$cuda_calls[measured$cuda$stages$stage=='laplace_scatter_launch']))
          utils::write.csv(do.call(rbind,stages),paste0(output,'.scatter-stages.csv'),row.names=FALSE)
        }
        rm(fixture,cpu,cuda);gc(FALSE)
      }
  },
  sparse_uniform = function() {
    source('tests/testthat/helper-device-sparse.R')
    for (width in c(4L,8L)) for(size in c(1000L,3000L,10000L,30000L,100000L)) {
      fixture <- device_sparse_fixture(size,'spde')
      values <- outer(fixture$values[,1L],1+seq_len(16L)/100)
      stopifnot(nrow(values)==length(fixture$i),ncol(values)==16L)
      measure_batch('sparse_uniform',width,size,function(choice,batched=TRUE)
        tulpa:::cpp_device_sparse_batch(fixture$p,fixture$i,values,fixture$b,
          choice,width,batched,FALSE))
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
source("tools/bench-device-dense.R")
basis_columns <- if (length(args) > 2L) as.integer(strsplit(args[[3L]], ",", fixed = TRUE)[[1L]]) else
  c(16L, 64L, 144L, 225L, 256L, 400L, 900L)
runners <- c(runners, device_dense_runners(append_row, basis_columns))
groups <- c(nngp_cholesky='nngp_cholesky',axpy='axpy',sparse_cholesky='sparse',
  sparse_selected_inverse='sparse',laplace_dense='laplace_dense',laplace_full_csc='laplace_dense',
  laplace_dense_inverse='laplace_dense_inverse',
  laplace_scatter='laplace_scatter',sparse_uniform='sparse_uniform')
extra <- setdiff(registered, names(groups))
groups <- c(groups, stats::setNames(extra, extra))
stopifnot(setequal(names(groups),registered),setequal(unname(groups),names(runners)))
dir.create(dirname(output), showWarnings = FALSE, recursive = TRUE)
cat(sprintf("Device: %s; driver=%d; threads=%s\n", info$name,
            info$driver_version, Sys.getenv("OMP_NUM_THREADS", "default")))
selected <- if (length(args) >= 2L) strsplit(args[[2L]], ',', fixed = TRUE)[[1L]] else registered
stopifnot(all(selected %in% registered))
for (group in unique(unname(groups[selected]))) runners[[group]]()
print(tulpa:::.device_thresholds(do.call(rbind, rows)))
writeLines(c(capture.output(print(info)), capture.output(sessionInfo())),
           paste0(output, ".session.txt"))
