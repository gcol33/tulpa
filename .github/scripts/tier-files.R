# The subset each CI tier runs.
#
# Tier 2 is the recovery / equivalence tier, run on every push, and it runs a
# named list (.github/tier-files.csv): the ungated suite at NOT_CRAN=true is
# weeks of fits, so what CI covers on a push is chosen from measured cost.
#
# Tier 3 is the sampler and multi-seed coverage tier, run on demand with one job
# per file. It runs every file carrying a skip_if_not_slow() gate, read off the
# sources. A gate is what the tier exists to open, and a list of gated files
# that nothing reads back stops matching the suite as tests are written.
#
# run-tests.R sources this to select the files it runs, and the slow-tests
# workflow calls tier_files_json() to build its job matrix, so the jobs and the
# files come from the same function.

TIER_FILES_CSV <- file.path(".github", "tier-files.csv")
TEST_DIR <- file.path("tests", "testthat")

test_files_on_disk <- function() {
  sort(basename(list.files(TEST_DIR, pattern = "^test-.*\\.R$")))
}

tier_table <- function() {
  if (!file.exists(TIER_FILES_CSV)) {
    stop("tier list not found: ", TIER_FILES_CSV, call. = FALSE)
  }
  tab <- utils::read.csv(TIER_FILES_CSV, stringsAsFactors = FALSE)
  missing_cols <- setdiff(c("file", "why"), names(tab))
  if (length(missing_cols)) {
    stop("tier list is missing column(s): ", paste(missing_cols, collapse = ", "),
         call. = FALSE)
  }
  tab$file <- trimws(tab$file)

  # A renamed or deleted test file must fail the job that would otherwise have
  # reported green while covering one file fewer.
  gone <- setdiff(tab$file, test_files_on_disk())
  if (length(gone)) {
    stop("tier list names ", length(gone), " file(s) that are not in ", TEST_DIR,
         ": ", paste(gone, collapse = ", "), call. = FALSE)
  }

  dup <- tab$file[duplicated(tab$file)]
  if (length(dup)) {
    stop("tier list repeats: ", paste(unique(dup), collapse = ", "), call. = FALSE)
  }
  tab
}

tier_files <- function(tier) {
  tier <- as.integer(tier)
  files <- switch(as.character(tier),
    "2" = sort(tier_table()$file),
    "1" = test_files_on_disk(),
    "3" = tier_gated_files("slow"),
    stop("no CI tier ", tier, call. = FALSE))
  if (!length(files)) {
    stop("tier ", tier, " holds no files", call. = FALSE)
  }
  files
}

tier_files_json <- function(tier) {
  paste0("[", paste0("\"", tier_files(tier), "\"", collapse = ","), "]")
}

# Which files carry a tier gate, read off the sources.
tier_gated_files <- function(gate = c("any", "cran", "slow")) {
  gate <- match.arg(gate)
  pattern <- switch(gate,
    any = "skip_on_cran\\(\\)|skip_if_not_slow\\(\\)",
    cran = "skip_on_cran\\(\\)",
    slow = "skip_if_not_slow\\(\\)")
  files <- test_files_on_disk()
  keep <- vapply(files, function(f) {
    any(grepl(pattern, readLines(file.path(TEST_DIR, f), warn = FALSE)))
  }, logical(1))
  files[keep]
}

# Gated files no tier runs. Tier 3 takes every slow-gated file, so these are the
# files gated on CRAN alone that the tier-2 list leaves out. A curated subset is
# a bound on coverage, and an unstated bound reads as full coverage; every job
# prints this count.
tier_uncovered <- function() {
  setdiff(tier_gated_files("any"), union(tier_files(2), tier_files(3)))
}

# Two suites run in shards, each shard a job of its own:
#
#   coverage  every test file -- the suite devtools::test() runs, tiers 1 and
#             2 -- against an instrumented build, several times slower than the
#             build the tiers above run.
#   tier1     every test file under the CRAN profile with the CRAN budget open
#             (TULPA_CRAN_BUDGET=off), on every platform R CMD check runs on.
#             R CMD check itself runs what CRAN runs, the smoke files.
#
# Files are packed onto shards by their measured seconds in that suite
# (`seconds_csv`, refreshed from the timings.csv every shard publishes). A file
# with no measurement is costed at the median of those with one, so a new test
# file lands on some shard without editing the table. With no file measured
# there is nothing to size the shard count from, and `unmeasured` is the count
# the measuring run spreads the suite over. `budget` is the seconds of tests one
# shard should carry once the table exists.

SHARD_SUITES <- list(
  coverage = list(seconds_csv = file.path(".github", "coverage-seconds.csv"),
                  budget = 2400, unmeasured = 12L),
  tier1 = list(seconds_csv = file.path(".github", "tier1-seconds.csv"),
               budget = 300, unmeasured = 3L)
)

shard_suite <- function(suite) {
  spec <- SHARD_SUITES[[suite]]
  if (is.null(spec)) {
    stop("no shard suite '", suite, "'; one of ",
         paste(names(SHARD_SUITES), collapse = ", "), call. = FALSE)
  }
  spec
}

shard_cost <- function(suite, files) {
  csv <- shard_suite(suite)$seconds_csv
  known <- if (file.exists(csv)) {
    tab <- utils::read.csv(csv, stringsAsFactors = FALSE)
    stats::setNames(tab$seconds, trimws(tab$file))
  } else {
    numeric()
  }
  fill <- if (length(known)) stats::median(known) else 1
  cost <- unname(known[files])
  cost[is.na(cost)] <- fill
  stats::setNames(cost, files)
}

# Longest-processing-time packing: each file, longest first, onto the shard
# carrying the fewest seconds so far. `n` unset sizes the shard count from the
# suite's budget.
plan_shards <- function(suite, n = NULL, files = test_files_on_disk()) {
  spec <- shard_suite(suite)
  cost <- shard_cost(suite, files)
  if (is.null(n) || is.na(n)) {
    n <- if (file.exists(spec$seconds_csv)) {
      ceiling(sum(cost) / spec$budget)
    } else {
      spec$unmeasured
    }
  }
  n <- max(1L, min(as.integer(n), length(files)))
  load <- numeric(n)
  shards <- vector("list", n)
  for (f in names(cost)[order(-cost, names(cost))]) {
    s <- which.min(load)
    shards[[s]] <- c(shards[[s]], f)
    load[s] <- load[s] + cost[[f]]
  }
  data.frame(
    shard = sprintf("%02d", seq_len(n)),
    files = vapply(shards, function(x) paste(sort(x), collapse = ","), ""),
    seconds = round(load),
    stringsAsFactors = FALSE)
}

shards_json <- function(plan) {
  paste0("[", paste0(
    "{\"shard\":\"", plan$shard, "\",\"files\":\"", plan$files,
    "\",\"seconds\":", plan$seconds, "}", collapse = ","), "]")
}

# The plan step both shard workflows run: prints the plan and writes `shards`
# (the job matrix) and `count` to GITHUB_OUTPUT. SHARDS and FILES are the
# optional dispatch inputs overriding the count and the file list.
write_shard_plan <- function(suite) {
  n <- if (nzchar(Sys.getenv("SHARDS"))) as.integer(Sys.getenv("SHARDS")) else NULL
  files <- trimws(strsplit(Sys.getenv("FILES"), ",", fixed = TRUE)[[1]])
  if (!length(files)) files <- test_files_on_disk()
  plan <- plan_shards(suite, n = n, files = files)
  print(plan[, c("shard", "seconds")], row.names = FALSE)
  out <- Sys.getenv("GITHUB_OUTPUT")
  cat("shards=", shards_json(plan), "\n", sep = "", file = out, append = TRUE)
  cat("count=", nrow(plan), "\n", sep = "", file = out, append = TRUE)
  invisible(plan)
}
