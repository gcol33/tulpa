# Shared loader for the dev_notes sweeps that score outer-grid read rules.
#
# Loads tulpa from a NAMED library (never the session's default one), sources
# the testthat helpers and the top-level definitions of the named test files
# into one environment whose parent is the tulpa namespace, and returns the
# identity every sweep stamps into its output: the library's BUILD_ID (the
# commit and overlaid file hashes the build script writes there), the installed
# package path and version, the R platform, and the prior arm and within-cell
# read the sweep states.

harness_load <- function(lib, repo, test_files = character(0)) {
  .libPaths(c(lib, .libPaths()))
  suppressPackageStartupMessages(library(tulpa, lib.loc = lib))
  env <- new.env(parent = asNamespace("tulpa"))
  tdir <- file.path(repo, "tests", "testthat")
  for (h in list.files(tdir, "^helper-.*[.]R$", full.names = TRUE)) {
    sys.source(h, envir = env)
  }
  for (f in test_files) {
    for (e in parse(file.path(tdir, f), keep.source = FALSE)) {
      if (is.call(e) && identical(e[[1L]], as.name("<-"))) eval(e, env)
    }
  }
  env
}

harness_identity <- function(lib, hyperprior, within_cell) {
  bid <- file.path(lib, "BUILD_ID")
  list(build       = if (file.exists(bid)) readLines(bid) else "BUILD_ID missing",
       tulpa_path  = find.package("tulpa"),
       tulpa_version = as.character(utils::packageVersion("tulpa")),
       platform    = R.version$platform,
       R           = R.version.string,
       hyperprior  = hyperprior,
       within_cell = within_cell,
       started     = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"))
}

harness_print_identity <- function(id) {
  cat("---- sweep identity ----\n")
  for (nm in names(id)) cat(sprintf("%-13s %s\n", nm, paste(id[[nm]], collapse = " | ")))
  cat("------------------------\n")
}

# The first BUILD_ID line, `head_sha=<sha>`, as a column value.
harness_build_tag <- function(id) sub("^head_sha=", "", id$build[1L])
