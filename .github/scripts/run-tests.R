#!/usr/bin/env Rscript

# Runs one CI tier against the INSTALLED package and reports the tier it ran,
# not the tier it was asked for. A recovery run reporting a few thousand
# assertions and several hundred skips looks the same as a run whose gates all
# skipped unless the tier and the counts are on the record.
#
# Environment:
#   TULPA_TIER        2 (recovery / equivalence) or 3 (samplers / coverage).
#   TULPA_TEST_FILES  comma-separated file names, overriding the tier list.
#                     The scheduled tier-3 matrix runs one file per job.
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
out_dir <- Sys.getenv("TULPA_TEST_OUT", "")

assigned <- if (nzchar(explicit)) {
  sort(trimws(strsplit(explicit, ",", fixed = TRUE)[[1]]))
} else {
  tier_files(tier)
}
filter <- tier_filter_regex(assigned)

# Confirm the pattern selects exactly the assigned files before spending the
# job on the assumption. A filter that quietly selects the wrong set is the one
# failure mode that still looks green.
selected <- tier_filter_selects(filter)
if (!setequal(selected, assigned)) {
  cat("filter does not select the assigned files\n")
  cat("  assigned but not selected:",
      paste(setdiff(assigned, selected), collapse = ", "), "\n")
  cat("  selected but not assigned:",
      paste(setdiff(selected, assigned), collapse = ", "), "\n")
  quit(status = 1L)
}

tier_name <- switch(
  tier,
  "2" = "recovery (single-fit recovery and equivalence)",
  "3" = "full validation (samplers and multi-seed coverage)",
  paste0("tier ", tier)
)

cat("tier      :", tier_name, "\n")
cat("tulpa     :", as.character(utils::packageVersion("tulpa")), "\n")
cat("R         :", R.version.string, "\n")
cat("NOT_CRAN  :", Sys.getenv("NOT_CRAN", "<unset>"), "\n")
cat("slow tier :", Sys.getenv("TULPA_SLOW_TESTS", "<unset>"), "\n")
cat(sprintf("files     : %d\n", length(assigned)))
cat("           ", paste(assigned, collapse = "\n            "), "\n\n")

started <- Sys.time()

# load_package = "installed": every worker does a plain library(tulpa) against
# one already-built shared object. The alternative has each worker recompile
# the C++ backend into the same src/ tree.
res <- test_dir(TEST_DIR, package = "tulpa", filter = filter,
                reporter = "summary", stop_on_failure = FALSE,
                load_package = "installed")

elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))

df <- as.data.frame(res)
failed <- sum(df$failed)
errors <- sum(as.integer(df$error))

by_file <- data.frame(file = character(0), seconds = numeric(0),
                      assertions = integer(0), skipped = integer(0),
                      failed = integer(0), errors = integer(0),
                      stringsAsFactors = FALSE)
ran <- character(0)
if (nrow(df)) {
  df$file <- basename(as.character(df$file))
  ran <- sort(unique(df$file))
  split_by <- factor(df$file, levels = ran)
  num <- function(col) as.numeric(tapply(col, split_by, sum))
  by_file <- data.frame(
    file = ran,
    seconds = round(num(df$real), 3),
    assertions = as.integer(num(df$passed)),
    skipped = as.integer(num(as.integer(df$skipped))),
    failed = as.integer(num(df$failed)),
    errors = as.integer(num(as.integer(df$error))),
    stringsAsFactors = FALSE)
  by_file <- by_file[order(-by_file$seconds), , drop = FALSE]
}

# The per-file seconds are what the tier list is sized against. Published every
# run so the subset is chosen from measured cost rather than from file size.
if (nzchar(out_dir)) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  utils::write.csv(by_file, file.path(out_dir, "timings.csv"), row.names = FALSE)
  utils::write.csv(data.frame(
    tier = tier,
    os = Sys.info()[["sysname"]],
    files_assigned = length(assigned),
    files_ran = length(ran),
    assertions = sum(df$passed),
    skipped = sum(df$skipped),
    failed = failed,
    errors = errors,
    seconds = round(elapsed, 1),
    stringsAsFactors = FALSE),
    file.path(out_dir, "summary.csv"), row.names = FALSE)
}

cat(sprintf(
  "\n%s\nassertions %d | skipped %d | failed %d | errors %d | %.1f min\n",
  tier_name, sum(df$passed), sum(df$skipped), failed, errors, elapsed / 60))

if (nrow(by_file)) {
  cat("\nSlowest files:\n")
  for (i in seq_len(min(10L, nrow(by_file)))) {
    cat(sprintf("  %8.1fs  %s\n", by_file$seconds[i], by_file$file[i]))
  }
}

if (failed > 0 || errors > 0) {
  bad <- df[df$failed > 0 | df$error, , drop = FALSE]
  cat("\nFailing tests:\n")
  for (i in seq_len(nrow(bad))) {
    cat(sprintf("  %s :: %s  (failed %d%s)\n",
                bad$file[i], bad$test[i], bad$failed[i],
                if (bad$error[i]) ", errored" else ""))
  }
}

# A file that was selected but produced no results at all would otherwise be
# counted as covered by a green job.
dropped <- setdiff(assigned, ran)
if (length(dropped)) {
  cat("\nAssigned but produced no results: ",
      paste(dropped, collapse = ", "), "\n", sep = "")
  quit(status = 1L)
}

if (failed > 0 || errors > 0) quit(status = 1L)

cat("Tier green.\n")
