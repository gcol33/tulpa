test_that("ABI mismatches are rejected on every attempt", {
  skip_on_cran_budget()
  skip_if_not(.cpp_fixture_has_toolchain(), "no C++ toolchain")
  root <- .tulpa_source_root()
  skip_if(is.null(root), "tulpa source tree not found")
  env <- new.env(parent = globalenv())
  include <- normalizePath(file.path(root, "inst", "include"), winslash = "/")
  withr::with_envvar(c(PKG_CPPFLAGS = paste0('-I"', include, '"')),
    Rcpp::sourceCpp(test_path("cpp", "abi_guard.cpp"), env = env,
      cacheDir = file.path(tempdir(), "tulpa-abi-guard")))
  for (attempt in 1:2)
    expect_error(env$cpp_test_abi_guard_mismatch(), "ABI mismatch")
})
