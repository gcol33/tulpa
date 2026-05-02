# helper-internal.R
#
# Aliases for tulpa internals used by the test suite. Sourced by testthat
# ahead of every test file. Without these, `library(tulpa); test_dir(...)`
# cannot resolve bare names because the C++ wrappers and many R helpers
# are not listed in NAMESPACE. `devtools::test()` worked transparently
# because load_all() exposes everything; CI / R CMD check / a fresh
# library load all need explicit `:::` accessors.
#
# This file iterates the package's namespace and binds every unexported
# function to a same-named alias in this helper environment, so test code
# can call them without `tulpa:::` prefixes.

local({
  ns <- asNamespace("tulpa")
  exported <- getNamespaceExports("tulpa")
  candidates <- ls(envir = ns)
  internals <- setdiff(candidates, exported)
  for (nm in internals) {
    obj <- get(nm, envir = ns, inherits = FALSE)
    if (is.function(obj)) {
      assign(nm, obj, envir = globalenv())
    }
  }
  invisible(NULL)
})
