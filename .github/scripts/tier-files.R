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
    "3" = tier_gated_files("slow"),
    stop("no CI tier ", tier, "; tier 1 is R CMD check", call. = FALSE))
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

# Coverage runs every test file -- the suite devtools::test() runs, tiers 1 and
# 2 -- against an instrumented build, which is several times slower than the
# build the tiers above run. It is split into shards, each a job of its own, and
# files are packed onto shards by their measured seconds under coverage
# (.github/coverage-seconds.csv, refreshed from the timings.csv every shard
# publishes). A file with no measurement is costed at the median of those with
# one, so a new test file lands on some shard without editing the table.

COVERAGE_SECONDS_CSV <- file.path(".github", "coverage-seconds.csv")

# With no file measured there is nothing to size the shard count from; this is
# the count the measuring run spreads the suite over.
COVERAGE_SHARDS_UNMEASURED <- 12L

coverage_cost <- function(files) {
  known <- if (file.exists(COVERAGE_SECONDS_CSV)) {
    tab <- utils::read.csv(COVERAGE_SECONDS_CSV, stringsAsFactors = FALSE)
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
# carrying the fewest seconds so far. `n` unset sizes the shard count from
# `budget`, the seconds of tests one shard should carry.
coverage_shards <- function(n = NULL, budget = 2400, files = test_files_on_disk()) {
  cost <- coverage_cost(files)
  if (is.null(n) || is.na(n)) {
    n <- if (file.exists(COVERAGE_SECONDS_CSV)) {
      ceiling(sum(cost) / budget)
    } else {
      COVERAGE_SHARDS_UNMEASURED
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

coverage_shards_json <- function(...) {
  plan <- coverage_shards(...)
  paste0("[", paste0(
    "{\"shard\":\"", plan$shard, "\",\"files\":\"", plan$files,
    "\",\"seconds\":", plan$seconds, "}", collapse = ","), "]")
}
