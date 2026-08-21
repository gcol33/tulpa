# ============================================================================
# Load-time S3 registration against generics owned by other packages.
#
# tulpa declares `fixef`, `ranef`, `VarCorr`, the `as_draws*` family,
# `pp_check`, `bayes_R2` and `posterior_predict` as its own generics so they
# work with nothing else attached. That alone would MASK lme4::fixef,
# nlme::ranef, posterior::as_draws, bayesplot::pp_check and
# rstantools::bayes_R2 / posterior_predict whenever both packages are attached,
# with the winner decided by attach order. Registering tulpa's methods on the
# other packages' generics as well removes the ambiguity: `lme4::fixef(fit)`,
# `posterior::as_draws(fit)` and `bayesplot::pp_check(y, yrep, fun)` all keep
# working, and tulpa's own generics stay available standalone.
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


# The mirror of `.s3_register()`: take the OWNER's own `.default` method and
# register it on tulpa's copy of the generic.
#
# Registering tulpa's method on the owner's generic fixes one direction only.
# In the other, `library(tulpa)` after `library(bayesplot)` makes bare
# `pp_check` resolve to tulpa's generic, whose UseMethod searches tulpa's own
# method table -- where the owner's `.default` is not, because a registered S3
# method is not an exported object. Every `pp_check(y, yrep, fun)` call in the
# session then fails on a numeric. Borrowing the default method here is what
# makes the two generics interchangeable at the call site; it is registered
# rather than exported, so it never masks the owner's own default in return.
.s3_borrow_default <- function(generic) {
  parts <- strsplit(generic, "::", fixed = TRUE)[[1L]]
  if (length(parts) != 2L) {
    stop("`generic` must be of the form 'package::generic'.", call. = FALSE)
  }
  pkg <- parts[[1L]]
  gen <- parts[[2L]]
  self <- topenv(parent.frame())

  register <- function(...) {
    ns <- asNamespace(pkg)
    if (!exists(gen, envir = ns, inherits = FALSE)) return(invisible(NULL))
    tbl <- get0(".__S3MethodsTable__.", envir = ns, inherits = FALSE)
    method <- if (is.null(tbl)) NULL
              else get0(paste0(gen, ".default"), envir = tbl, inherits = FALSE)
    if (is.null(method)) {
      method <- get0(paste0(gen, ".default"), envir = ns, inherits = FALSE)
    }
    if (is.null(method)) return(invisible(NULL))
    registerS3method(gen, "default", method, envir = self)
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
    .s3_register(paste0(pkg, "::VarCorr"), "tulpa_fit")
  }

  for (gen in c("as_draws", "as_draws_array", "as_draws_matrix",
                "as_draws_df", "as_draws_rvars")) {
    .s3_register(paste0("posterior::", gen), "tulpa_fit")
  }

  # bayesplot owns `pp_check`; rstantools owns `bayes_R2` and
  # `posterior_predict`. Unregistered, `pp_check(y, yrep, fun)` fails for the
  # whole session once tulpa is attached last, since tulpa's generic reaches no
  # method for a numeric vector.
  for (gen in c("bayesplot::pp_check", "rstantools::bayes_R2",
                "rstantools::posterior_predict")) {
    .s3_register(gen, "tulpa_fit")
    .s3_borrow_default(gen)
  }

  invisible(NULL)
}
