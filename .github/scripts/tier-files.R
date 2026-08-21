# The named subset each CI tier runs, read from .github/tier-files.csv.
#
# Tier 2 is the recovery / equivalence tier, run on every push. Tier 3 is the
# sampler and multi-seed coverage tier, run on a schedule with one job per
# file. Neither tier runs the whole directory: the ungated suite at
# NOT_CRAN=true is weeks of fits, so what CI covers is a list, and a list that
# nothing reads back is a list that silently stops matching the suite.
#
# run-tests.R sources this to select the files it runs, and the scheduled
# workflow calls tier_files_json() to build its job matrix, so the jobs and the
# files come from the same rows.

TIER_FILES_CSV <- file.path(".github", "tier-files.csv")
TEST_DIR <- file.path("tests", "testthat")

tier_table <- function() {
  if (!file.exists(TIER_FILES_CSV)) {
    stop("tier list not found: ", TIER_FILES_CSV, call. = FALSE)
  }
  tab <- utils::read.csv(TIER_FILES_CSV, stringsAsFactors = FALSE)
  missing_cols <- setdiff(c("file", "tier", "why"), names(tab))
  if (length(missing_cols)) {
    stop("tier list is missing column(s): ", paste(missing_cols, collapse = ", "),
         call. = FALSE)
  }
  tab$file <- trimws(tab$file)
  tab$tier <- as.integer(tab$tier)

  # A renamed or deleted test file must fail the job that would otherwise have
  # reported green while covering one file fewer.
  on_disk <- basename(list.files(TEST_DIR, pattern = "^test-.*\\.R$"))
  gone <- setdiff(tab$file, on_disk)
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
  tab <- tier_table()
  files <- sort(tab$file[tab$tier == tier])
  if (!length(files)) {
    stop("tier list holds no tier-", tier, " files", call. = FALSE)
  }
  files
}

tier_files_json <- function(tier) {
  paste0("[", paste0("\"", tier_files(tier), "\"", collapse = ","), "]")
}

# Which files carry a tier gate at all, read off the sources rather than off a
# list. A gate is what CI has to open for the block behind it to run, so this is
# the population the tier lists are drawn from, and it moves on its own as tests
# are written.
tier_gated_files <- function(gate = c("any", "cran", "slow")) {
  gate <- match.arg(gate)
  pattern <- switch(gate,
    any = "skip_on_cran\\(\\)|skip_if_not_slow\\(\\)",
    cran = "skip_on_cran\\(\\)",
    slow = "skip_if_not_slow\\(\\)")
  files <- sort(basename(list.files(TEST_DIR, pattern = "^test-.*\\.R$")))
  keep <- vapply(files, function(f) {
    any(grepl(pattern, readLines(file.path(TEST_DIR, f), warn = FALSE)))
  }, logical(1))
  files[keep]
}

# Gated files no tier list names. A curated subset is a bound on coverage, and
# an unstated bound reads as full coverage; every job that runs a subset prints
# this count.
tier_uncovered <- function() {
  tab <- tier_table()
  setdiff(tier_gated_files("any"), tab$file)
}
