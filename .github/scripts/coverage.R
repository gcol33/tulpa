#!/usr/bin/env Rscript

# Line coverage of the test suite, measured one shard of test files per job and
# merged into the one report Codecov receives.
#
#   coverage.R shard          builds the package instrumented, runs the files in
#                             TULPA_TEST_FILES through run-tests.R against that
#                             build, and writes coverage.rds next to the
#                             timings in TULPA_TEST_OUT.
#   coverage.R merge <dir>    sums every coverage.rds under <dir> and writes
#                             cobertura.xml.
#
# Environment:
#   TULPA_COV_OPT     optimisation level of the instrumented build, in place of
#                     covr's -O0; unset or empty is -Og. Eigen's expression
#                     templates are written for an optimiser that inlines them,
#                     and at -O0 every coefficient access is a call: on the
#                     same nine test files the instrumented run took 31.4 min
#                     at -O0, 10.4 at -Og, 11.9 at -O1 and 8.3 at -O2, against
#                     4.2 uninstrumented. -Og is the level GCC keeps the
#                     line mapping for, and it reported 21287 of 68692 lines
#                     covered where -O0 reported 21043 of 68403.
#   TULPA_COV_SHARDS  merge: how many shards were planned. A merge that finds
#                     fewer reports stops, so a lost shard cannot reach Codecov
#                     as the coverage of the whole suite.

suppressPackageStartupMessages(library(covr))

args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args)) args[[1]] else ""

shard <- function() {
  out_dir <- Sys.getenv("TULPA_TEST_OUT", "")
  if (!nzchar(out_dir)) stop("TULPA_TEST_OUT is not set", call. = FALSE)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out_dir <- normalizePath(out_dir, winslash = "/")
  Sys.setenv(TULPA_TEST_OUT = out_dir, TULPA_DEFER_FAILURE = "true")

  opt <- Sys.getenv("TULPA_COV_OPT", "")
  if (!nzchar(opt)) opt <- "-Og"
  flags <- getOption("covr.flags")
  flags <- sub("-O0", opt, flags, fixed = TRUE)
  options(covr.flags = flags)
  cat("instrumented build flags:", unique(flags), sep = "\n  ")
  cat("\n")

  # The commands run in a fresh R CMD BATCH process, which inherits this
  # process's environment and working directory; run-tests.R reads its file
  # list and the test directory from both.
  root <- normalizePath(".", winslash = "/")
  code <- sprintf("setwd(%s); source(file.path('.github', 'scripts', 'run-tests.R'))",
                  deparse(root))

  install_path <- file.path(
    normalizePath(Sys.getenv("RUNNER_TEMP", tempdir()), winslash = "/"), "package")

  cov <- covr::package_coverage(
    type = "none",
    code = code,
    quiet = FALSE,
    clean = FALSE,
    install_path = install_path)

  saveRDS(cov, file.path(out_dir, "coverage.rds"))
  cat(sprintf("\nshard line coverage: %.2f%%\n", covr::percent_coverage(cov)))
}

# A report is one entry per traced R expression and one per line of each gcov
# file, and gcov writes a header once for every translation unit including it,
# so the same line is several entries. covr sums entries that share a source
# file, a source span and a function before it tallies lines, and the merge
# applies that same sum across shards. The entries are not the same list in
# every shard: gcov leaves out a line reached only on an exception path until
# something executes it, so which lines a shard lists depends on what it ran.
entry_keys <- function(cov) {
  vapply(unclass(cov), function(e) {
    paste(attr(e$srcref, "srcfile")$filename,
          paste(as.integer(e$srcref), collapse = ":"),
          e$functions[1L], sep = "|")
  }, character(1), USE.NAMES = FALSE)
}

merge <- function(dir) {
  files <- sort(list.files(dir, pattern = "^coverage\\.rds$", recursive = TRUE,
                           full.names = TRUE))
  planned <- as.integer(Sys.getenv("TULPA_COV_SHARDS", NA))
  cat(sprintf("shard reports: %d of %s planned\n", length(files),
              if (is.na(planned)) "<unset>" else planned))
  if (is.na(planned) || length(files) != planned) {
    stop("coverage is incomplete: every planned shard must report", call. = FALSE)
  }

  covs <- lapply(files, readRDS)
  for (i in seq_along(covs)) {
    cat(sprintf("  %-40s %6.2f%%\n", basename(dirname(files[[i]])),
                covr::percent_coverage(covs[[i]])))
  }

  keys <- unlist(lapply(covs, entry_keys))
  values <- unlist(lapply(covs, function(cv) {
    vapply(unclass(cv), function(e) as.numeric(e$value), numeric(1),
           USE.NAMES = FALSE)
  }))
  entries <- do.call(c, lapply(covs, unclass))
  first <- !duplicated(keys)
  totals <- tapply(values, factor(keys, levels = unique(keys)), sum)

  merged <- entries[first]
  for (i in seq_along(merged)) {
    merged[[i]]$value <- totals[[i]]
  }
  attrs <- attributes(covs[[1]])
  attrs$names <- names(merged)
  attributes(merged) <- attrs

  cat(sprintf("merged line coverage: %.2f%%\n", covr::percent_coverage(merged)))
  covr::to_cobertura(merged)
}

switch(mode,
  shard = shard(),
  merge = merge(if (length(args) > 1) args[[2]] else "."),
  stop("usage: coverage.R shard | coverage.R merge <dir>", call. = FALSE))
