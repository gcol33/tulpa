#!/usr/bin/env Rscript

# Runs one CI tier against the INSTALLED package and reports the tier it ran,
# not the tier it was asked for. A recovery run reporting a few thousand
# assertions and several hundred skips looks the same as a run whose gates all
# skipped unless the tier and the counts are on the record.
#
# Files run one at a time, and the per-file row is written as soon as that file
# finishes. A job killed at the runner's cap therefore still publishes seconds
# for everything it got through, which is what the tier lists are sized from.
#
# Environment:
#   TULPA_TIER        2 (recovery / equivalence) or 3 (samplers / coverage).
#   TULPA_TEST_FILES  comma-separated file names, overriding the tier list.
#                     The scheduled tier-3 matrix runs one file per job.
#   TULPA_TEST_SCOPE  "curated" (default) runs the tier list. "all-gated" runs
#                     every file carrying a skip_on_cran() gate, which is the
#                     measurement run the curated list is chosen from.
#   TULPA_TEST_OUT    directory for the machine-readable results. Unset writes
#                     nothing.
#
# The gates themselves read NOT_CRAN and TULPA_SLOW_TESTS, which the workflow
# sets; this script only selects which files run.

suppressPackageStartupMessages({
  library(testthat)
  library(tulpa)
})

source(file.path(".github", "scripts", "tier-files.R"))

tier <- Sys.getenv("TULPA_TIER", "2")
explicit <- Sys.getenv("TULPA_TEST_FILES", "")
scope <- Sys.getenv("TULPA_TEST_SCOPE", "curated")
out_dir <- Sys.getenv("TULPA_TEST_OUT", "")

assigned <- if (nzchar(explicit)) {
  sort(trimws(strsplit(explicit, ",", fixed = TRUE)[[1]]))
} else if (identical(scope, "all-gated") && identical(tier, "2")) {
  sort(setdiff(tier_gated_files("cran"), tier_files(3)))
} else {
  tier_files(tier)
}

on_disk <- basename(list.files(TEST_DIR, pattern = "^test-.*\\.R$"))
absent <- setdiff(assigned, on_disk)
if (length(absent)) {
  cat("assigned files that are not in ", TEST_DIR, ": ",
      paste(absent, collapse = ", "), "\n", sep = "")
  quit(status = 1L)
}

tier_name <- switch(
  tier,
  "2" = "recovery (single-fit recovery and equivalence)",
  "3" = "full validation (samplers and multi-seed coverage)",
  paste0("tier ", tier)
)

cat("tier      :", tier_name, "\n")
cat("scope     :", if (nzchar(explicit)) "explicit" else scope, "\n")
cat("tulpa     :", as.character(utils::packageVersion("tulpa")), "\n")
cat("R         :", R.version.string, "\n")
cat("NOT_CRAN  :", Sys.getenv("NOT_CRAN", "<unset>"), "\n")
cat("slow tier :", Sys.getenv("TULPA_SLOW_TESTS", "<unset>"), "\n")
cat(sprintf("files     : %d\n", length(assigned)))

# A curated subset is a bound on coverage, and a bound nothing states reads as
# full coverage.
uncovered <- tier_uncovered()
if (length(uncovered)) {
  cat(sprintf("uncovered : %d gated file(s) in no tier list\n", length(uncovered)))
  cat("           ", paste(utils::head(uncovered, 5), collapse = ", "),
      if (length(uncovered) > 5) sprintf(" (+%d more)", length(uncovered) - 5) else "",
      "\n", sep = "")
}
cat("\n")

timings_path <- if (nzchar(out_dir)) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  file.path(out_dir, "timings.csv")
} else {
  NULL
}
if (!is.null(timings_path)) {
  cat("file,seconds,assertions,skipped,failed,errors\n", file = timings_path)
}

rows <- list()
failures <- list()
started <- Sys.time()

for (f in assigned) {
  t0 <- Sys.time()
  # load_package = "installed": a plain library(tulpa) against one already-built
  # shared object. The alternative recompiles the C++ backend into the source
  # tree for every file.
  res <- test_file(file.path(TEST_DIR, f), package = "tulpa",
                   reporter = "summary", stop_on_failure = FALSE,
                   load_package = "installed")
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  df <- as.data.frame(res)
  row <- data.frame(
    file = f,
    seconds = round(secs, 3),
    assertions = if (nrow(df)) sum(df$passed) else 0L,
    skipped = if (nrow(df)) sum(as.integer(df$skipped)) else 0L,
    failed = if (nrow(df)) sum(df$failed) else 0L,
    errors = if (nrow(df)) sum(as.integer(df$error)) else 0L,
    stringsAsFactors = FALSE)
  rows[[f]] <- row

  if (!is.null(timings_path)) {
    cat(paste(row$file, row$seconds, row$assertions, row$skipped,
              row$failed, row$errors, sep = ","), "\n",
        sep = "", file = timings_path, append = TRUE)
  }

  if (nrow(df) && (row$failed > 0 || row$errors > 0)) {
    bad <- df[df$failed > 0 | df$error, , drop = FALSE]
    for (i in seq_len(nrow(bad))) {
      failures[[length(failures) + 1L]] <- sprintf(
        "  %s :: %s  (failed %d%s)", f, bad$test[i], bad$failed[i],
        if (bad$error[i]) ", errored" else "")
    }
  }

  cat(sprintf("[%3d/%3d] %8.1fs  %-56s  %5d ok  %3d skipped%s\n",
              length(rows), length(assigned), secs, f,
              row$assertions, row$skipped,
              if (row$failed + row$errors > 0)
                sprintf("  %d FAILED", row$failed + row$errors) else ""))
}

elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
by_file <- do.call(rbind, rows)
by_file <- by_file[order(-by_file$seconds), , drop = FALSE]

failed <- sum(by_file$failed)
errors <- sum(by_file$errors)

if (nzchar(out_dir)) {
  utils::write.csv(data.frame(
    tier = tier,
    scope = if (nzchar(explicit)) "explicit" else scope,
    os = Sys.info()[["sysname"]],
    files = nrow(by_file),
    uncovered = length(uncovered),
    assertions = sum(by_file$assertions),
    skipped = sum(by_file$skipped),
    failed = failed,
    errors = errors,
    seconds = round(elapsed, 1),
    stringsAsFactors = FALSE),
    file.path(out_dir, "summary.csv"), row.names = FALSE)
}

cat(sprintf(
  "\n%s\nfiles %d | assertions %d | skipped %d | failed %d | errors %d | %.1f min\n",
  tier_name, nrow(by_file), sum(by_file$assertions), sum(by_file$skipped),
  failed, errors, elapsed / 60))

cat("\nSlowest files:\n")
for (i in seq_len(min(10L, nrow(by_file)))) {
  cat(sprintf("  %8.1fs  %s\n", by_file$seconds[i], by_file$file[i]))
}

if (length(failures)) {
  cat("\nFailing tests:\n")
  cat(paste(unlist(failures), collapse = "\n"), "\n")
  quit(status = 1L)
}

cat("\nTier green.\n")
