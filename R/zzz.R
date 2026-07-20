# ============================================================================
# Load-time S3 registration against generics owned by other packages.
#
# tulpa declares `fixef`, `ranef` and the `as_draws*` family as its own generics
# so they work with nothing else attached. That alone would MASK lme4::fixef,
# nlme::ranef and posterior::as_draws whenever both packages are attached, with
# the winner decided by attach order. Registering tulpa's methods on the other
# packages' generics as well removes the ambiguity: `lme4::fixef(fit)` and
# `posterior::as_draws(fit)` dispatch to the tulpa_fit methods, and tulpa's own
# generics stay available standalone.
#
# None of these packages are Imports, and none is loaded here. Registration runs
# only if the package is already loaded, and is otherwise deferred to its load
# hook -- so tulpa neither depends on them nor forces them into memory.
# ============================================================================

# Register `method` for `class` on a generic owned by another package, now if
# that package is loaded and on its next load otherwise. The method is looked up
# lazily inside tulpa's namespace so registration works during load, before the
# namespace is sealed.
.s3_register <- function(generic, class) {
  parts <- strsplit(generic, "::", fixed = TRUE)[[1L]]
  if (length(parts) != 2L) {
    stop("`generic` must be of the form 'package::generic'.", call. = FALSE)
  }
  pkg <- parts[[1L]]
  gen <- parts[[2L]]
  method_name <- paste0(gen, ".", class)
  # Our own namespace, taken from the calling frame rather than by name, so this
  # keeps working under load_all() and does not carry a hardcoded package name.
  self <- topenv(parent.frame())

  register <- function(...) {
    ns <- asNamespace(pkg)
    # The other package may not export this generic (nlme and lme4 do not carry
    # identical surfaces, and posterior's shape variants have come and gone).
    # A missing generic is a non-event, not an error.
    if (!exists(gen, envir = ns, inherits = FALSE)) return(invisible(NULL))
    method <- get0(method_name, envir = self, inherits = FALSE)
    if (is.null(method)) return(invisible(NULL))
    registerS3method(gen, class, method, envir = ns)
    invisible(NULL)
  }

  if (isNamespaceLoaded(pkg)) register()
  setHook(packageEvent(pkg, "onLoad"), register)
  invisible(NULL)
}


.onLoad <- function(libname, pkgname) {
  # lme4 and nlme both own `fixef` and `ranef`; whichever the user has, the
  # tulpa_fit methods answer. tulpa's ranef() returns a one-row-per-coefficient
  # data frame rather than lme4's list of per-term data frames -- an S3 method
  # answers for its own class, so the shape is tulpa's.
  for (pkg in c("lme4", "nlme")) {
    .s3_register(paste0(pkg, "::fixef"), "tulpa_fit")
    .s3_register(paste0(pkg, "::ranef"), "tulpa_fit")
  }

  for (gen in c("as_draws", "as_draws_array", "as_draws_matrix",
                "as_draws_df", "as_draws_rvars")) {
    .s3_register(paste0("posterior::", gen), "tulpa_fit")
  }

  invisible(NULL)
}
