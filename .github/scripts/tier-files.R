# The named subset each CI tier runs, read from .github/tier-files.csv.
#
# Tier 2 is the recovery / equivalence tier, run on every push. Tier 3 is the
# sampler and multi-seed coverage tier, run on a schedule with one job per
# file. Neither tier runs the whole directory: the ungated suite at
# NOT_CRAN=true is weeks of fits, so what CI covers is a list, and a list that
# nothing reads back is a list that silently stops matching the suite.
#
# run-tests.R sources this to build its filter, and the scheduled workflow
# calls tier_files_json() to build its job matrix, so the matrix and the filter
# come from the same rows.

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

# testthat matches `filter` against the file name stripped of its "test-"
# prefix and ".R" extension, so the pattern is built on the stem and anchored:
# an unanchored alternation selects every longer name sharing a prefix.
tier_stems <- function(files) sub("^test-", "", sub("\\.R$", "", files))

tier_filter_regex <- function(files) {
  paste0("^(", paste(gsub(".", "\\.", tier_stems(files), fixed = TRUE),
                     collapse = "|"), ")$")
}

# What the pattern actually selects out of the whole directory. run-tests.R
# checks this against the assigned list before spending the job on it.
tier_filter_selects <- function(filter) {
  on_disk <- basename(list.files(TEST_DIR, pattern = "^test-.*\\.R$"))
  sort(on_disk[grepl(filter, tier_stems(on_disk))])
}

tier_files_json <- function(tier) {
  paste0("[", paste0("\"", tier_files(tier), "\"", collapse = ","), "]")
}
