# A test-file fixture built on first use and shared by every later block.
#
# `cached_fixture(build)` returns an accessor; the first call runs `build()` and
# every later call returns that same value. Several blocks that read different
# properties off one fit (an accessor, a routing field, an interval) then pay
# for the fit once. The build runs inside whichever test_that() block calls the
# accessor first, after that block's tier gate, so a fixture behind
# skip_on_cran() is never built where the gate is closed. A build that errors
# is not cached: the next caller retries and reports the same error in its own
# block.
cached_fixture <- function(build) {
  value <- NULL
  built <- FALSE
  function() {
    if (!built) {
      value <<- build()
      built <<- TRUE
    }
    value
  }
}

assign("cached_fixture", cached_fixture, envir = globalenv())
