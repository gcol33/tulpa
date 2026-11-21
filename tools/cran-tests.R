# The source tarball carries the tests CRAN runs: the smoke files named by
# CRAN_SMOKE_FILES in tests/testthat/helper-tiers.R, with every helper and
# fixture. The full suite stays in the repository, where the tier-1 CI job and
# every local profile run it. This script writes the .Rbuildignore line that
# leaves out every other test file.
#
#   Rscript tools/cran-tests.R          # rewrite the line
#   Rscript tools/cran-tests.R --check  # check only; non-zero exit if stale
#
# Run it from the package root after changing CRAN_SMOKE_FILES.

args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args
tests <- file.path("tests", "testthat")
ignore_file <- ".Rbuildignore"
prefix <- "^tests/testthat/test-(?!"

exprs <- parse(file.path(tests, "helper-tiers.R"), keep.source = FALSE)
is_smoke <- vapply(exprs, function(e) {
  is.call(e) && identical(e[[1L]], as.name("<-")) &&
    identical(e[[2L]], as.name("CRAN_SMOKE_FILES"))
}, logical(1))
if (sum(is_smoke) != 1L) {
  stop("helper-tiers.R must assign CRAN_SMOKE_FILES exactly once")
}
smoke <- eval(exprs[is_smoke][[1L]][[3L]], baseenv())

absent <- smoke[!file.exists(file.path(tests, paste0("test-", smoke, ".R")))]
if (length(absent)) {
  stop("smoke files absent from tests/testthat: ", paste(absent, collapse = ", "))
}

escaped <- gsub("([][{}()+*^$|\\\\?.])", "\\\\\\1", sort(smoke))
line <- paste0(prefix, "(?:", paste(escaped, collapse = "|"), ")\\.R$)")

current <- readLines(ignore_file, warn = FALSE)
at <- which(startsWith(current, prefix))
if (length(at) > 1L) stop(ignore_file, " holds more than one smoke-file line")
updated <- if (length(at)) replace(current, at, line) else c(current, line)

files <- list.files(tests, "^test-.*\\.R$")
kept <- files[!grepl(line, file.path("tests/testthat", files), perl = TRUE,
                     ignore.case = TRUE)]
if (!setequal(kept, paste0("test-", smoke, ".R"))) {
  stop("the generated line does not keep exactly the smoke files")
}

if (identical(updated, current)) {
  cat("smoke-file line is current:", length(kept), "of", length(files),
      "test files ship\n")
} else if (check_only) {
  stop(ignore_file, " smoke-file line is stale; run Rscript tools/cran-tests.R")
} else {
  con <- file(ignore_file, "wb")
  writeLines(updated, con)
  close(con)
  cat("wrote the smoke-file line:", length(kept), "of", length(files),
      "test files ship\n")
}
