device_gpu_processes <- function() {
  lines <- system2('nvidia-smi', stdout = TRUE, stderr = TRUE)
  if (!is.null(attr(lines, 'status'))) stop('nvidia-smi failed')
  compute <- lines[grepl('^[|].*\\sC\\s', lines, perl=TRUE)]
  pids <- as.integer(sub('^.*?([0-9]+)\\s+C\\s.*$', '\\1', compute, perl=TRUE))
  load <- device_process_gpu_load()
  load <- load[names(load) != as.character(Sys.getpid())]
  idle_compute <- !is.na(pids) & (pids == Sys.getpid() | !(as.character(pids) %in% names(load)[load > 1]))
  heavy <- names(load)[load > 10]
  cpu <- device_foreign_cpu_cores()
  cores_free <- parallel::detectCores() * 0.1
  list(lines=c(lines, sprintf('engine utilization pid %s: %.1f%%', names(load)[load > 1], load[load > 1]),
                sprintf('foreign CPU load: %.1f cores', cpu)),
       busy=c(compute[!idle_compute], sprintf('pid %s engine utilization %.1f%%', heavy, load[heavy]),
              if (cpu > cores_free) sprintf('foreign CPU load %.1f cores', cpu)))
}

device_foreign_cpu_cores <- function() {
  own <- function() sum(proc.time()[c('user.self', 'sys.self')], na.rm = TRUE)
  script <- "(Get-Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 3 -MaxSamples 1).CounterSamples[0].CookedValue"
  before <- own(); start <- proc.time()[['elapsed']]
  out <- system2('powershell', c('-NoProfile', '-Command', shQuote(script, type = 'cmd')), stdout = TRUE, stderr = TRUE)
  total <- suppressWarnings(as.numeric(tail(out[nzchar(trimws(out))], 1L)))
  if (!is.finite(total)) stop('processor counter unavailable')
  mine <- (own() - before) / max(proc.time()[['elapsed']] - start, 1)
  max(0, total / 100 * parallel::detectCores() - mine)
}

device_process_gpu_load <- function() {
  script <- paste0("$s = (Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -SampleInterval 1 -MaxSamples 3).CounterSamples; ",
    "$s | Where-Object { $_.InstanceName -match '^pid_(\\d+)_' } | ForEach-Object { '{0} {1}' -f ",
    "($_.InstanceName -replace '^pid_(\\d+)_.*$','$1'), $_.CookedValue }")
  out <- system2('powershell', c('-NoProfile', '-Command', shQuote(script, type = 'cmd')), stdout = TRUE, stderr = TRUE)
  parts <- strsplit(trimws(out[grepl('^[0-9]+ ', out)]), ' ')
  if (!length(parts)) stop('GPU engine counters unavailable')
  pid <- vapply(parts, `[`, '', 1L); value <- as.numeric(vapply(parts, `[`, '', 2L))
  load <- tapply(value, pid, max)
  stats::setNames(as.numeric(load), names(load))
}

device_wait_for_gpu <- function(log = NULL) {
  repeat {
    state <- device_gpu_processes()
    lines <- state$lines; busy <- state$busy
    if (!length(busy)) break
    cat('Waiting for GPU compute processes:', paste(busy, collapse = '\n'), '\n')
    flush.console()
    if (!is.null(log)) writeLines(c(format(Sys.time()), lines), log)
    Sys.sleep(30)
  }
  if (!is.null(log)) writeLines(c(format(Sys.time()), lines), log)
  invisible(lines)
}

device_fit_benchmark <- function(append_row, table_file, sizes = c(1000L, 3000L, 10000L, 30000L, 100000L),
                                 trials = 3L, models = c('spde', 'areal', 'temporal')) {
  source('tests/testthat/helper-device-sparse.R')
  profile <- Sys.getenv('TULPA_DEVICE_PROFILE', unset = NA_character_)
  on.exit(if (is.na(profile)) Sys.unsetenv('TULPA_DEVICE_PROFILE') else Sys.setenv(TULPA_DEVICE_PROFILE = profile))
  Sys.setenv(TULPA_DEVICE_PROFILE = 'true')
  timings <- read.csv(system.file('device', 'benchmarks.csv', package = 'tulpa'))
  settings <- tulpa:::.device_thresholds(timings)
  candidate <- settings
  candidate$threshold[] <- Inf
  candidate$batch_threshold[] <- Inf
  candidate$threshold[candidate$operation == 'sparse_cholesky' & candidate$variant == 0L] <- 1
  sources <- tulpa:::.device_kernel_sources(dirname(find.package('tulpa')), 'tulpa')
  on.exit(tulpa:::cpp_device_configure(settings, sources), add = TRUE)
  capture <- new.env(parent=emptyenv())
  capture$cells <- capture$newton <- capture$calls <- 0L
  entries <- grep('^cpp_nested_laplace',ls(asNamespace('tulpa')),value=TRUE)
  tracer <- substitute({
    value <- returnValue()
    sink <- SINK
    if (is.list(value) && !is.null(value$n_iter)) {
      sink$cells <- sink$cells + length(value$n_iter)
      sink$newton <- sink$newton + sum(value$n_iter)
      sink$calls <- sink$calls + 1L
    }
  },list(SINK=capture))
  for(entry in entries) trace(entry,where=asNamespace('tulpa'),print=FALSE,exit=tracer)
  on.exit(for(entry in entries) untrace(entry,where=asNamespace('tulpa')),add=TRUE)
  references <- list()
  done <- read.csv(table_file, stringsAsFactors = FALSE)
  measured_pair <- function(model, size) {
    if (!all(c('model', 'scope', 'nodes') %in% names(done))) return(FALSE)
    sum(done$calibration == calibration & done$operation == 'sparse_cholesky' & done$nodes %in% size &
      done$model %in% model & done$scope %in% 'fit') >= trials
  }
  for (model in models) for (size in as.integer(sizes)) {
    if (measured_pair(model, size)) {
      cat(sprintf('fit %s nodes=%d already measured in calibration %d
', model, size, calibration))
      next
    }
    input <- device_sparse_fit_inputs(size, model, 'cpu')
    run <- function(choice) {
      repeat {
        device_wait_for_gpu(paste0(table_file, '.gpu.txt'))
        tulpa:::cpp_device_configure(if (choice == 'factor_cuda') candidate else settings, sources)
        input$control$device <- if (choice == 'factor_cuda') 'auto' else choice
        capture$cells <- capture$newton <- capture$calls <- 0L
        start <- proc.time()[['elapsed']]
        fit <- device_sparse_fit(input, if(model=='spde') 'joint' else 'registry')
        seconds <- proc.time()[['elapsed']] - start
        contention <- device_gpu_processes()$busy
        if (!length(contention)) break
        cat(sprintf('discarded %s %s nodes=%d timing: GPU contention %s\n', model, choice, size,
          paste(contention, collapse = '; ')))
        flush.console()
      }
      expected_cells <- prod(vapply(input$prior,function(block)
        max(1L,lengths(block[endsWith(names(block),'_grid')])),0))
      stopifnot(all(fit$converged), !any(fit$device_cpu_recovered),
        ncol(fit$modes) == size + 2L, length(fit$log_marginal) == expected_cells,
        capture$cells >= length(fit$log_marginal),capture$newton >= sum(fit$n_iter))
      measured <- stats::setNames(rep(0, 5L), c('analysis', 'factor', 'solve', 'selected_inverse', 'inverse_factor'))
      timing <- fit$device$timings
      stopifnot(!is.null(timing), all(timing$stage %in% names(measured)))
      measured[timing$stage] <- timing$seconds
      stopifnot(sum(measured) <= seconds + .01)
      other <- max(0, seconds - sum(measured))
      stages <- fit$device$stages
      work <- stages$work[stages$stage == 'sparse_cholesky']
      work <- if (length(work)) max(work) else NA_real_
      stopifnot(choice == 'cuda' || (is.finite(work) && work > 0))
      row <- data.frame(model = model, size = size, device = choice, seconds = seconds, work = work,
        cells = capture$cells, newton_steps = capture$newton, native_calls = capture$calls,
        returned_cells = length(fit$log_marginal), returned_newton_steps = sum(fit$n_iter),
        analysis = measured[['analysis']], analysis_calls = sum(timing$calls[timing$stage == 'analysis']),
        factor = measured[['factor']] + measured[['inverse_factor']], solve = measured[['solve']],
        inverse_factor = measured[['inverse_factor']],
        selected_inverse = measured[['selected_inverse']], other = other,
        inverse_share = measured[['selected_inverse']] / seconds,
        factor_stage_calls = sum(timing$calls[timing$stage == 'factor']),
        inverse_stage_calls = sum(timing$calls[timing$stage == 'selected_inverse']),
        inverse_evaluations = sum(fit$device$stages$cpu_calls[fit$device$stages$stage=='sparse_selected_inverse']) +
          sum(fit$device$stages$cuda_calls[fit$device$stages$stage=='sparse_selected_inverse']),
        sparse_cuda = any(fit$device$stages$stage=='sparse_cholesky' & fit$device$stages$cuda_calls>0))
      list(fit = fit, row = row)
    }
    reference <- run('cpu')
    quantities <- c('log_marginal','modes','fitted_eta_var')
    references[[paste(model,size)]] <- c(reference$fit[quantities], list(work = reference$row$work))
    for (choice in c('auto', 'cuda', 'factor_cuda')) {
      warm <- run(choice)
      for (q in c('log_marginal', 'modes', 'fitted_eta_var')) {
        scale <- max(1, abs(reference$fit[[q]]))
        error <- max(abs(warm$fit[[q]] - reference$fit[[q]]))
        cat(sprintf('benchmark %s nodes=%d %s %s scale=%.17g error=%.17g tolerance=1e-7\n',
          model, size, choice, q, scale, error))
        stopifnot(error <= 1e-7 * scale)
      }
    }
    rm(reference, warm); gc(FALSE)
    for (trial in seq_len(trials)) {
      order <- c('cpu', 'auto', 'cuda', 'factor_cuda')
      order <- order[((seq_along(order) + trial - 2L) %% length(order)) + 1L]
      results <- lapply(order, run); names(results) <- order
      extra <- list(nodes = size, model = model, scope = 'fit', trial = trial,
        auto_seconds = results$auto$row$seconds, forced_cuda_seconds = results$cuda$row$seconds,
        auto_threshold = settings$threshold[settings$operation=='sparse_cholesky' & settings$variant==0L])
      for (choice in order) for (stage in setdiff(names(results[[choice]]$row),
        c('model', 'size', 'device', 'seconds', 'work'))) {
        extra[[paste(choice, stage, sep = '_')]] <- results[[choice]]$row[[stage]]
      }
      append_row('sparse_cholesky', 0L, results$cpu$row$work,
        results$cpu$row$seconds, results$factor_cuda$row$seconds,
        extra = extra)
      cat(sprintf('fit %s nodes=%d trial=%d cpu=%.6f auto=%.6f cuda=%.6f factor_cuda=%.6f\n',
        model, size, trial, results$cpu$row$seconds, results$auto$row$seconds,
        results$cuda$row$seconds, results$factor_cuda$row$seconds))
      flush.console()
      rm(results); gc(FALSE)
    }
  }
  calibrated <- tulpa:::.device_thresholds(read.csv(table_file,stringsAsFactors=FALSE))
  gate <- function(plan) plan$threshold[plan$operation=='sparse_cholesky' & plan$variant==0L]
  if (!identical(gate(calibrated),gate(settings))) {
    settings <- calibrated
    for(model in models) for(size in as.integer(sizes)) {
      input <- device_sparse_fit_inputs(size,model,'auto')
      reference <- references[[paste(model,size)]]
      if (is.null(reference)) {
        base <- run('cpu')
        reference <- c(base$fit[c('log_marginal','modes','fitted_eta_var')], list(work = base$row$work))
      }
      for(trial in seq_len(trials)) {
        result <- run('auto')
        for(q in c('log_marginal','modes','fitted_eta_var')) {
          scale <- max(1,abs(reference[[q]]))
          stopifnot(max(abs(result$fit[[q]]-reference[[q]])) <= 1e-7*scale)
        }
        extra <- list(nodes=size,model=model,scope='fit_auto',trial=trial,
          auto_seconds=result$row$seconds,auto_threshold=gate(settings))
        for(stage in setdiff(names(result$row),c('model','size','device','seconds','work')))
          extra[[paste0('auto_',stage)]] <- result$row[[stage]]
        append_row('sparse_cholesky',0L,reference$work,NA_real_,NA_real_,extra=extra)
      }
    }
  }
  invisible(NULL)
}
