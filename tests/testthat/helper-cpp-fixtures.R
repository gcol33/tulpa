# C++ test fixtures.
#
# The Rcpp entry points through which the suite drives the engine's internal
# C++ -- an autodiff primitive, a prior density, a sampler kernel on a planted
# target -- live in tests/testthat/cpp/<name>.cpp and are not part of the
# installed package. load_cpp_fixture(name) compiles one with Rcpp::sourceCpp()
# and links it against the engine's own objects: the package's src/ unity
# files, compiled by R CMD SHLIB with the package's own Makevars, from the same
# source tree the tested tulpa was installed from. A fixture therefore calls the
# engine's functions and instantiates its headers; it holds no copy of either.
#
# The engine objects are built once per source tree and cached, keyed by a
# hash of src/, inst/include/, the R version and the LinkingTo packages, under
# TULPA_FIXTURE_CACHE (default tools::R_user_dir("tulpa", "cache")). A
# fixture's own library is cached by sourceCpp next to them.
#
# A fixture links its own copy of the engine's objects, so process state the
# engine keeps (a registry, a verification flag) is the fixture's own. A
# fixture that has to reach the state of the loaded tulpa -- registering a
# cell coupling a front-door fit then looks up -- goes through the registered
# C callables, as a consumer package does.
#
# Compiling is not cheap, so the loader skips in fast mode and under the CRAN
# profile (skip_on_cran_budget(): TULPA_CRAN_BUDGET=off opens it, which is how
# the CI check matrix runs the fixture blocks on every platform), and also
# without a C++ toolchain and where the package sources are not on disk. A
# test file calls it inside the block or helper that needs it.

# Package source root: the source tree under devtools::test(), the unpacked
# tarball under R CMD check.
.tulpa_source_root <- function() {
  cands <- c(
    testthat::test_path("..", ".."),
    testthat::test_path("..", "..", "00_pkg_src", "tulpa")
  )
  for (d in cands) {
    desc <- file.path(d, "DESCRIPTION")
    if (file.exists(file.path(d, "src", "Makevars")) && file.exists(desc) &&
        identical(unname(read.dcf(desc, "Package")[1L, 1L]), "tulpa")) {
      return(normalizePath(d, winslash = "/"))
    }
  }
  NULL
}

.cpp_fixture_state <- new.env(parent = emptyenv())

.cpp_fixture_r <- function() {
  file.path(R.home("bin"), if (.Platform$OS.type == "windows") "R.exe" else "R")
}

.cpp_fixture_has_toolchain <- function() {
  if (is.null(.cpp_fixture_state$toolchain)) {
    cxx <- suppressWarnings(tryCatch(
      system2(.cpp_fixture_r(), c("CMD", "config", "CXX17"),
              stdout = TRUE, stderr = FALSE),
      error = function(e) character()
    ))
    prog <- if (length(cxx)) strsplit(trimws(cxx[1L]), "[[:space:]]+")[[1L]][1L] else ""
    .cpp_fixture_state$toolchain <-
      nzchar(prog) && nzchar(Sys.which(prog)) && nzchar(Sys.which("make"))
  }
  .cpp_fixture_state$toolchain
}

.cpp_fixture_linking_to <- c("Rcpp", "RcppEigen", "Matrix")

.cpp_fixture_include_flags <- function(root) {
  inc <- c(file.path(root, "src"), file.path(root, "inst", "include"),
           vapply(.cpp_fixture_linking_to,
                  function(p) system.file("include", package = p), ""))
  paste0("-I\"", inc, "\"", collapse = " ")
}

# Hash of everything the engine objects are compiled from.
.cpp_fixture_key <- function(root) {
  files <- c(
    list.files(file.path(root, "src"), pattern = "\\.(cpp|h|hpp|c)$|^Makevars",
               full.names = TRUE),
    list.files(file.path(root, "inst", "include"), recursive = TRUE,
               full.names = TRUE)
  )
  files <- sort(files)
  sums <- unname(tools::md5sum(files))
  rel <- substring(files, nchar(root) + 2L)
  vers <- vapply(.cpp_fixture_linking_to,
                 function(p) as.character(utils::packageVersion(p)), "")
  blob <- c(R.version.string, .Platform$r_arch, vers, paste(rel, sums))
  tf <- tempfile()
  on.exit(unlink(tf), add = TRUE)
  writeLines(blob, tf)
  unname(tools::md5sum(tf))
}

.cpp_fixture_cache_root <- function() {
  root <- Sys.getenv("TULPA_FIXTURE_CACHE", "")
  if (!nzchar(root)) root <- tools::R_user_dir("tulpa", "cache")
  file.path(root, "cpp-fixtures")
}

# Compile the engine's unity objects from `root` into the cache, once per key.
# The build runs in a scratch copy of src/ and inst/include/ (R CMD SHLIB
# writes its objects next to the sources) and is renamed into place when
# complete, so a concurrent session never reads a half-built directory.
.cpp_fixture_engine <- function(root) {
  if (!is.null(.cpp_fixture_state$engine)) return(.cpp_fixture_state$engine)
  key <- .cpp_fixture_key(root)
  cache <- .cpp_fixture_cache_root()
  dir <- file.path(cache, paste0("engine-", substr(key, 1L, 16L)))
  if (!file.exists(file.path(dir, "objects.txt"))) {
    dir.create(cache, recursive = TRUE, showWarnings = FALSE)
    tmp <- file.path(cache, paste0("build-", Sys.getpid(), "-",
                                   substr(key, 1L, 16L)))
    unlink(tmp, recursive = TRUE)
    dir.create(file.path(tmp, "inst"), recursive = TRUE)
    file.copy(file.path(root, "src"), tmp, recursive = TRUE)
    file.copy(file.path(root, "inst", "include"), file.path(tmp, "inst"),
              recursive = TRUE)
    src <- file.path(tmp, "src")
    unlink(list.files(src, pattern = "\\.(o|dll|so|dylib)$", full.names = TRUE))
    unity <- sort(list.files(src, pattern = "^unity_.*\\.cpp$"))
    log <- file.path(tmp, "build.log")
    flags <- paste(vapply(.cpp_fixture_linking_to,
                          function(p) paste0("-I\"", system.file("include", package = p), "\""),
                          ""), collapse = " ")
    owd <- setwd(src)
    status <- withr::with_envvar(
      c(CLINK_CPPFLAGS = flags),
      system2(.cpp_fixture_r(),
              c("CMD", "SHLIB", "-o", paste0("tulpa_engine", .Platform$dynlib.ext),
                unity),
              stdout = log, stderr = log)
    )
    setwd(owd)
    objs <- sub("\\.cpp$", ".o", unity)
    if (!identical(status, 0L) || !all(file.exists(file.path(src, objs)))) {
      stop("building the engine objects for the C++ test fixtures failed; ",
           "last lines of ", log, ":\n",
           paste(utils::tail(readLines(log, warn = FALSE), 30L), collapse = "\n"),
           call. = FALSE)
    }
    writeLines(objs, file.path(src, "objects.txt"))
    file.copy(file.path(src, "objects.txt"), file.path(tmp, "objects.txt"))
    if (!file.rename(tmp, dir)) {
      unlink(tmp, recursive = TRUE)
      if (!file.exists(file.path(dir, "objects.txt"))) {
        stop("could not move the fixture engine build into ", dir, call. = FALSE)
      }
    }
  }
  objs <- file.path(dir, "src", readLines(file.path(dir, "objects.txt")))
  .cpp_fixture_state$engine <- list(dir = dir, objects = objs)
  .cpp_fixture_state$engine
}

# Compile tests/testthat/cpp/<name>.cpp against the engine and assign its
# exported functions into `env` (the calling test file or block by default).
load_cpp_fixture <- function(name, env = parent.frame()) {
  skip_if_fast()
  skip_on_cran_budget()
  testthat::skip_if_not_installed("RcppEigen")
  testthat::skip_if_not(.cpp_fixture_has_toolchain(), "no C++ toolchain")
  root <- .tulpa_source_root()
  testthat::skip_if(is.null(root), "tulpa source tree not found")

  fns <- .cpp_fixture_state$loaded[[name]]
  if (is.null(fns)) {
    engine <- .cpp_fixture_engine(root)
    file <- normalizePath(testthat::test_path("cpp", paste0(name, ".cpp")),
                          winslash = "/", mustWork = TRUE)
    lib_flags <- paste(
      paste0("\"", engine$objects, "\"", collapse = " "),
      "$(SHLIB_OPENMP_CXXFLAGS) $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)"
    )
    fx <- new.env(parent = globalenv())
    withr::with_envvar(
      c(PKG_CPPFLAGS = .cpp_fixture_include_flags(file.path(engine$dir)),
        PKG_CXXFLAGS = "$(SHLIB_OPENMP_CXXFLAGS)",
        PKG_LIBS = lib_flags,
        USE_CXX17 = "yes"),
      Rcpp::sourceCpp(file, env = fx,
                      cacheDir = file.path(engine$dir, "fixtures"))
    )
    fns <- as.list(fx)
    if (is.null(.cpp_fixture_state$loaded)) .cpp_fixture_state$loaded <- list()
    .cpp_fixture_state$loaded[[name]] <- fns
  }
  list2env(fns, envir = env)
  invisible(names(fns))
}
