# Every Matrix CHOLMOD stub tulpa calls resolves its target on first call
# through R_GetCCallable(), which touches the R protect stack. The DLL-load
# resolver (src/cholmod_stub_resolve.cpp) calls each stub once on the main
# thread, so a stub it misses would first resolve on an OpenMP worker and race
# R_PPStackTop (gcol33/tulpa#918). This checks the resolver covers every stub
# the sources use.

test_that("the DLL-load resolver calls every M_cholmod_* stub used in src/", {
  src <- test_path("..", "..", "src")
  skip_if_not(dir.exists(src), "package sources not available")
  files <- list.files(src, pattern = "[.](cpp|h|hpp)$", full.names = TRUE)
  resolver <- file.path(src, "cholmod_stub_resolve.cpp")
  expect_true(file.exists(resolver))

  stubs_in <- function(paths) {
    txt <- unlist(lapply(paths, readLines, warn = FALSE))
    txt <- sub("//.*$", "", txt)
    sort(unique(unlist(regmatches(txt,
      gregexpr("M_cholmod_[A-Za-z0-9_]+", txt)))))
  }
  used    <- stubs_in(setdiff(files, resolver))
  covered <- stubs_in(resolver)

  expect_gt(length(used), 0L)
  expect_identical(setdiff(used, covered), character(0))
})
