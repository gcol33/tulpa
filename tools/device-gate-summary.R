device_gate_summary <- function(csv) {
  timings <- read.csv(csv,stringsAsFactors=FALSE)
  operations <- tulpa:::cpp_device_operations()
  missing <- setdiff(operations,unique(timings$operation))
  gates <- tulpa:::.device_thresholds(timings)
  fits <- if ('scope' %in% names(timings)) timings[!is.na(timings$scope) & timings$scope=='fit',,drop=FALSE] else timings[FALSE,]
  if(nrow(fits)) fits <- fits[fits$calibration==max(fits$calibration),,drop=FALSE]
  validation <- if('scope' %in% names(timings)) timings[!is.na(timings$scope) & timings$scope=='fit_auto',,drop=FALSE] else timings[FALSE,]
  if(nrow(fits) && nrow(validation)) {
    keys <- c('operation','variant','size','nodes','model','trial','calibration')
    validation <- validation[validation$calibration==max(fits$calibration),,drop=FALSE]
    auto_columns <- grep('^auto_',names(validation),value=TRUE)
    stopifnot(!anyDuplicated(validation[keys]))
    fits <- merge(fits,validation[c(keys,auto_columns)],by=keys,all.x=TRUE,
      suffixes=c('','.validated'),sort=FALSE)
    for(name in auto_columns) {
      replacement <- fits[[paste0(name,'.validated')]]
      use <- !is.na(replacement)
      fits[[name]][use] <- replacement[use]
    }
  }
  required <- expand.grid(model=c('spde','areal','temporal'),size=c(1000L,3000L,10000L,30000L,100000L))
  present <- if(nrow(fits)) paste(fits$model,fits$nodes) else character()
  complete <- all(paste(required$model,required$size) %in% present)
  lines <- c('## Fit timing ladder','',
    'Factor work is the CHOLMOD flop count of the fit\'s sparse factorization, the unit the CUDA threshold is read in. Seconds are medians of paired complete fits. Factor CUDA uses device factorization and CPU selected inversion; its complete-fit time calibrates the sparse factor operation.',
    'Factor includes log determinants and the host factor needed for CPU selected inversion; that host factor is also recorded separately as inverse_factor in the CSV. Selected inverse measures the inverse work alone. Symbolic analysis covers host and device analysis of each sparsity pattern. Other includes assembly, likelihood, line search, and R work. Worker count is one. Cells and Newton steps include every native outer-placement and grid call; returned grid counts are recorded separately in the CSV.',
    '', '| Model | Nodes | Factor work | Cells CPU/auto/CUDA/hybrid | Newton CPU/auto/CUDA/hybrid | CPU | Auto | Forced CUDA | Factor CUDA | Measured noise | Auto inverse share |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
  rows <- list(); pass <- !length(missing) && complete
  auto_safe_all <- TRUE
  sparse_gate <- gates$threshold[gates$operation=='sparse_cholesky' & gates$variant==0L]
  # Fit seconds are elapsed proc.time differences, which advance in 10 ms
  # steps, so one step is the smallest difference the timer can resolve.
  tick <- 0.01
  gate_matches <- nrow(fits)>0L && 'auto_threshold' %in% names(fits) && all(fits$auto_threshold == sparse_gate)
  if(!gate_matches) pass <- FALSE
  if(nrow(fits)) for(key in unique(paste(fits$model,fits$nodes))) {
    g <- fits[paste(fits$model,fits$nodes)==key,,drop=FALSE]
    med <- function(name) stats::median(g[[name]])
    spread <- function(x) (max(x)-min(x))/stats::median(x)
    noise <- max(spread(g$cpu_seconds),spread(g$auto_seconds),spread(g$cuda_seconds))
    safe <- med('auto_seconds') <= med('cpu_seconds')*(1+noise) + tick
    auto_safe_all <- auto_safe_all && safe
    if(g$size[1L]>=sparse_gate && !all(g$auto_sparse_cuda)) pass <- FALSE
    if(nrow(g)<3L || !safe) pass <- FALSE
    share <- med('auto_selected_inverse')/med('auto_seconds')
    cells <- paste(vapply(c('cpu','auto','cuda','factor_cuda'),function(x)med(paste0(x,'_cells')),0),collapse='/')
    newton <- paste(vapply(c('cpu','auto','cuda','factor_cuda'),function(x)med(paste0(x,'_newton_steps')),0),collapse='/')
    lines <- c(lines,sprintf('| %s | %d | %.3g | %s | %s | %.6f | %.6f | %.6f | %.6f | %.1f%% | %.1f%% |',
      g$model[1L],as.integer(g$nodes[1L]),g$size[1L],cells,newton,med('cpu_seconds'),
      med('auto_seconds'),med('forced_cuda_seconds'),med('cuda_seconds'),100*noise,100*share))
    rows[[length(rows)+1L]] <- data.frame(model=g$model[1L],nodes=g$nodes[1L],work=g$size[1L],
      cpu=med('cpu_seconds'),auto=med('auto_seconds'),cuda=med('forced_cuda_seconds'),
      factor_cuda=med('cuda_seconds'),noise=noise,auto_safe=safe,inverse_share=share)
  }
  lines <- c(lines,'',
    'Noise is the largest observed repetition range divided by its median across CPU, auto and factor-CUDA fits at that model and size. The gate checks auto against CPU with that measured range widened by the 10 ms resolution of the elapsed timer. These are numerical equivalence benchmarks; they do not establish parameter recovery or coverage.',
    '', '## Fit stages','', '| Model | Nodes | Device | Symbolic analysis | Factor and logdet | Solve | Selected inverse | Other |',
    '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |')
  if(nrow(fits)) for(key in unique(paste(fits$model,fits$nodes))) {
    g <- fits[paste(fits$model,fits$nodes)==key,,drop=FALSE]
    for(choice in c('cpu','auto','cuda','factor_cuda')) {
      values <- vapply(c('analysis','factor','solve','selected_inverse','other'),function(stage)
        stats::median(g[[paste(choice,stage,sep='_')]]),0)
      lines <- c(lines,sprintf('| %s | %d | %s | %.6f | %.6f | %.6f | %.6f | %.6f |',
        g$model[1L],as.integer(g$nodes[1L]),choice,values[1L],values[2L],values[3L],values[4L],values[5L]))
    }
  }
  lines <- c(lines,'', '## Device gates','', '| Operation | Variant | CUDA threshold | Batch threshold | Benchmark rows |',
    '| --- | ---: | ---: | ---: | ---: |')
  for(i in seq_len(nrow(gates))) {
    g <- gates[i,]
    n <- sum(timings$operation==g$operation & timings$variant==g$variant)
    lines <- c(lines,sprintf('| %s | %d | %s | %s | %d |',g$operation,g$variant,
      format(g$threshold,scientific=FALSE),format(g$batch_threshold,scientific=FALSE),n))
  }
  row_table <- if(length(rows)) do.call(rbind,rows) else data.frame()
  available <- nrow(row_table)>0L && any(row_table$factor_cuda < row_table$cpu*(1-row_table$noise) - tick)
  delivered <- nrow(row_table)>0L && any(row_table$work>=sparse_gate & row_table$auto < row_table$cpu*(1-row_table$noise) - tick)
  if(available && !delivered) pass <- FALSE
  lines <- c(lines,'',paste('Operations lacking benchmark rows:',if(length(missing)) paste(missing,collapse=', ') else 'none'),
    paste('Complete fit ladder:',complete),paste('Auto measured at the calibrated gate:',gate_matches),
    paste('Auto within measured noise at every size:',if(complete)auto_safe_all else NA),
    paste('Sparse fit speed-up exceeds measured noise:',available),paste('Auto delivers that speed-up:',delivered))
  source('tools/device-equivalence-map.R',local=TRUE)
  map <- device_equivalence_map()
  stopifnot(setequal(map$operation,operations))
  lines <- c(lines,'','## Device operation matrix','',
    '| Operation | Equivalence file and test name | Threshold coverage | Benchmark rows |',
    '| --- | --- | --- | ---: |')
  for (i in seq_len(nrow(map))) {
    row <- map[i,]
    if (!row$test %in% device_test_names(file.path('tests/testthat',row$file))) pass <- FALSE
    plan <- gates[gates$operation==row$operation,,drop=FALSE]
    coverage <- paste(if(any(is.finite(plan$threshold))) 'Both sides of each finite gate.' else '',
      if(any(!is.finite(plan$threshold))) 'Inf variants: CPU auto and forced CUDA; upper side absent.' else '')
    lines <- c(lines,sprintf('| %s | %s: %s | %s | %d |',row$operation,row$file,row$test,
      trimws(coverage),sum(timings$operation==row$operation)))
  }
  list(pass=pass,lines=lines,rows=row_table,gates=gates)
}
