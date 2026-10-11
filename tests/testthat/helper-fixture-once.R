# A fit that several blocks of one test file read, built once per file.
#
# `fixture_once(build)` returns an accessor: the first call with a given set of
# arguments runs `build(...)` and stores the result, and later calls with the
# same arguments return the stored value. A build that errors stores nothing,
# so the next block that asks runs it again and reports the error under its own
# name. The store belongs to the accessor, which a test file creates at its top
# level, so a stored fit lives as long as that file's run and is never shared
# with another file.
fixture_once <- function(build) {
  store <- new.env(parent = emptyenv())
  function(...) {
    key <- paste(deparse(list(...)), collapse = "")
    if (!exists(key, envir = store, inherits = FALSE)) {
      assign(key, build(...), envir = store)
    }
    get(key, envir = store, inherits = FALSE)
  }
}
