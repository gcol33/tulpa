args <- commandArgs(trailingOnly = TRUE)
option <- function(name, default = NULL) {
  at <- match(paste0('--',name),args)
  if (is.na(at)) return(default)
  if (at == length(args)) stop('missing value for --',name)
  args[[at+1L]]
}
output <- normalizePath(option('output','gpu-release-gate'),mustWork=FALSE,winslash='/')
dir.create(output,recursive=TRUE,showWarnings=FALSE)
worker <- option('worker','')
Sys.setenv(NOT_CRAN='true',TULPA_SLOW_TESTS='true')
Sys.unsetenv('TULPA_FAST')
Sys.unsetenv('TULPA_DEVICE_PROFILE')
lib <- option('library')
if (!is.null(lib)) .libPaths(c(lib,.libPaths()))
cudss <- option('cudss',Sys.getenv('CUDSS_DIR'))
cuda <- option('cuda',Sys.getenv('CUDA_PATH'))
if (nzchar(cudss)) Sys.setenv(CUDSS_DIR=cudss)
if (nzchar(cuda)) Sys.setenv(CUDA_PATH=cuda)
Sys.setenv(PATH=paste(c(if(nzchar(cudss)) file.path(cudss,'bin'),
  if(nzchar(cuda)) c(file.path(cuda,'bin','x64'),file.path(cuda,'bin')),
  Sys.getenv('PATH')),collapse=.Platform$path.sep))
library(tulpa)
Sys.setenv(R_LIBS=paste(.libPaths(),collapse=.Platform$path.sep))
filter <- '^device-|^gpu-|^nested-laplace-joint-batch-equivalence$'
write_counts <- function(result, prefix) {
  saveRDS(result,file.path(output,paste0(prefix,'.rds')))
  tab <- as.data.frame(result)
  write.csv(tab[setdiff(names(tab),'result')],file.path(output,paste0(prefix,'.csv')),row.names=FALSE)
  counts <- colSums(tab[c('failed','error','warning','skipped','passed')])
  write.csv(as.data.frame(as.list(counts)),file.path(output,paste0(prefix,'-counts.csv')),row.names=FALSE)
  counts
}
if (nzchar(worker)) {
  if (!worker %in% c('tests','memcheck-tests')) stop('unknown worker')
  Sys.setenv(TULPA_DEVICE_EQUIVALENCE=file.path(output,paste0(worker,'-equivalence.csv')))
  result <- testthat::test_local(load_package='installed',filter=filter,
    reporter='summary',stop_on_failure=FALSE)
  counts <- write_counts(result,worker)
  print(counts)
  quit(status=if(counts[['failed']] || counts[['error']]) 1L else 0L)
}
info <- gpu_info()
if (!info$available || any(nzchar(info$library_reasons))) stop('all GPU runtime capabilities are required')
csv <- option('csv','inst/device/benchmarks.csv')
installed_csv <- system.file('device','benchmarks.csv',package='tulpa')
if (!identical(unname(tools::md5sum(csv)),unname(tools::md5sum(installed_csv))))
  stop('the installed private library must contain the selected benchmark table')
phase <- option('phase','all')
if (!phase %in% c('all','summary')) stop('phase must be all or summary')
inputs <- c(getLoadedDLLs()[['tulpa']][['path']],file.path(find.package('tulpa'),'R','tulpa.rdb'),csv,
  unlist(lapply(c('src','R','inst/include','inst/device','tests/testthat'),function(dir)
    list.files(dir,pattern='[.](R|h|cpp|cu)$',full.names=TRUE,recursive=TRUE))))
manifest <- tools::md5sum(inputs)
manifest_path <- file.path(output,'inputs.rds')
if (phase=='all') {
  saveRDS(manifest,manifest_path)
} else if (!file.exists(manifest_path) || !identical(readRDS(manifest_path),manifest)) {
  stop('summary inputs differ from the tested package, calibration or fixtures')
}
files <- list.files('tests/testthat',pattern='^test-.*\\.R$')
files <- files[grepl(filter,sub('^test-','',tools::file_path_sans_ext(files)))]
writeLines(files,file.path(output,'test-files.txt'))
writeLines(c(capture.output(info),capture.output(sessionInfo())),file.path(output,'session.txt'))
script <- normalizePath('tools/release-gate-gpu.R',winslash='/')
rscript <- file.path(R.home('bin'),'Rscript.exe')
run <- function(exe, argv, tag) {
  status <- system2(exe,argv,stdout=file.path(output,paste0(tag,'.stdout.log')),
    stderr=file.path(output,paste0(tag,'.stderr.log')))
  writeLines(as.character(status),file.path(output,paste0(tag,'.status')))
  status
}
test_status <- if(phase=='all') {
  run(rscript,c('--vanilla',shQuote(script),'--worker','tests','--output',shQuote(output)),'tests')
} else as.integer(readLines(file.path(output,'tests.status')))
sanitizer <- option('sanitizer')
if (is.null(sanitizer)) {
  found <- list.files('C:/GillesC/tulpa_fix/r22_sparse22_work/sanitizer',
    pattern='^compute-sanitizer[.]exe$',recursive=TRUE,full.names=TRUE)
  if (length(found)!=1L) stop('pass --sanitizer with the CUDA 13.4 executable')
  sanitizer <- found[[1L]]
}
version <- system2(sanitizer,'--version',stdout=TRUE,stderr=TRUE)
writeLines(version,file.path(output,'sanitizer-version.txt'))
if (!any(grepl('2026[.]3[.]',version))) stop('the CUDA 13.4 Compute Sanitizer is required')
sanitizer_status <- if(phase=='all') {
  run(sanitizer,c('--tool','memcheck','--target-processes','all',
  '--error-exitcode','99','--leak-check','full',shQuote(rscript),'--vanilla',shQuote(script),
  '--worker','memcheck-tests','--output',shQuote(output)),'memcheck')
} else as.integer(readLines(file.path(output,'memcheck.status')))
logs <- unlist(lapply(file.path(output,c('memcheck.stdout.log','memcheck.stderr.log')),readLines,warn=FALSE))
error_lines <- logs[grepl('ERROR SUMMARY: [0-9]+ errors',logs)]
error_counts <- as.numeric(sub('.*ERROR SUMMARY: ([0-9]+) errors.*','\\1',error_lines))
sanitizer_errors <- if(length(error_counts)) sum(error_counts) else NA_real_
source('tools/device-gate-summary.R')
summary <- device_gate_summary(csv)
read_counts <- function(prefix) {
  path <- file.path(output,paste0(prefix,'-counts.csv'))
  if(file.exists(path)) read.csv(path) else data.frame(failed=NA,error=NA,warning=NA,skipped=NA,passed=NA)
}
normal <- read_counts('tests'); sanitized <- read_counts('memcheck-tests')
passed <- test_status==0L && sanitizer_status==0L && is.finite(sanitizer_errors) && sanitizer_errors==0 && summary$pass
lines <- c(paste('# GPU release gate:',if(passed)'PASS' else 'FAIL'),'',
  paste('Tree:',system2('git',c('rev-parse','HEAD'),stdout=TRUE)),
  paste('Device:',info$name),paste('Benchmark table:',normalizePath(csv,winslash='/')),'',
  '| Run | Passed assertions | Failed assertions | Errors | Warnings | Skips |',
  '| --- | ---: | ---: | ---: | ---: | ---: |',
  sprintf('| Equivalence | %d | %d | %d | %d | %d |',normal$passed,normal$failed,normal$error,normal$warning,normal$skipped),
  sprintf('| Memcheck equivalence | %d | %d | %d | %d | %d |',sanitized$passed,sanitized$failed,sanitized$error,sanitized$warning,sanitized$skipped),'',
  paste('Sanitizer error count:',sanitizer_errors),paste('Sanitizer:',paste(version,collapse=' ')),
  '',summary$lines)
writeLines(lines,file.path(output,'SUMMARY.md'))
cat(paste(lines,collapse='\n'),'\n')
quit(status=if(passed)0L else 1L)
