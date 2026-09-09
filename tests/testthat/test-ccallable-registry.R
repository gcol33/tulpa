# Every C callable the exported headers advertise resolves (gcol33/tulpa#688).
#
# inst/include/tulpa/*.h is the interface a LinkingTo package compiles against,
# and each shim reaches the engine through R_GetCCallable. An unregistered name
# is a hard R error at the FIRST call, so the consumer's build is clean and the
# failure lands in its user's session. joint_nested_laplace_api.h advertised
# "tulpa_nested_laplace_joint_bym2" through four releases with no registration
# anywhere in src/.
#
# The names are scraped from the headers rather than listed here, so a header
# added without its registration fails in this suite instead of downstream.

test_that("every R_GetCCallable name in the exported headers is registered", {
  inc <- system.file("include", "tulpa", package = "tulpa")
  skip_if(!nzchar(inc) || !dir.exists(inc), "headers not installed")

  hdrs <- list.files(inc, pattern = "\\.h$", full.names = TRUE)
  expect_gt(length(hdrs), 0L)

  txt <- unlist(lapply(hdrs, readLines, warn = FALSE))
  # R_GetCCallable("tulpa", "name") -- the call is sometimes split over lines,
  # so match on the quoted pair rather than on the call itself.
  hits <- regmatches(txt, gregexpr('"tulpa"\\s*,\\s*"tulpa_[A-Za-z0-9_]+"', txt))
  names_found <- unique(sub('.*"(tulpa_[A-Za-z0-9_]+)".*', "\\1",
                            unlist(hits)))
  expect_gt(length(names_found), 20L)

  unresolved <- Filter(function(nm) {
    !isTRUE(tryCatch(tulpa:::cpp_test_ccallable_resolves(nm),
                     error = function(e) FALSE))
  }, names_found)

  expect_identical(unresolved, character(0))
})
