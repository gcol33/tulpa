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
#                     covr's -O0. Eigen's expression templates are written for
#                     an optimiser that inlines them; at -O0 every coefficient
#                     access is a call.
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

  opt <- Sys.getenv("TULPA_COV_OPT", "-O0")
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

# Every shard instruments the same build of the same commit, so every report
# lists the same lines in the same order and the merge is a sum of counts. The
# order is asserted rather than assumed: a report that lists different lines is
# a different build, and adding its counts by position would credit one line's
# hits to another.
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
  keys <- names(covs[[1]])
  for (i in seq_along(covs)[-1]) {
    if (!identical(names(covs[[i]]), keys)) {
      stop(files[[i]], " lists different lines from ", files[[1]], call. = FALSE)
    }
  }

  counts <- Reduce(`+`, lapply(covs, function(cv) {
    vapply(cv, function(e) as.numeric(e$value), numeric(1))
  }))
  merged <- covs[[1]]
  for (i in seq_along(merged)) {
    merged[[i]]$value <- counts[[i]]
  }

  for (i in seq_along(covs)) {
    cat(sprintf("  %-40s %6.2f%%\n", basename(dirname(files[[i]])),
                covr::percent_coverage(covs[[i]])))
  }
  cat(sprintf("merged line coverage: %.2f%%\n", covr::percent_coverage(merged)))
  covr::to_cobertura(merged)
}

switch(mode,
  shard = shard(),
  merge = merge(if (length(args) > 1) args[[2]] else "."),
  stop("usage: coverage.R shard | coverage.R merge <dir>", call. = FALSE))
