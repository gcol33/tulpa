# Outer hyperparameter mode-find.
#
# `fit_spde_nested_ccd()` and `fit_st_nested()`'s auto-grid both locate an outer
# mode the same way: box-constrained L-BFGS-B on a marginal that arrives from a
# compiled kernel with no analytic gradient, so both depend on `optim()`'s
# central-difference step and both treat a failed call as "no usable mode".
# The invocation lives here once; the tuning each one uses is in
# `.NL_MODE_FIND` (R/settings.R).

# Resolve a consumer's mode-find tuning, applying a caller's `control$mode_find`
# overrides on top of the settings defaults.
#
# `control` is the already-validated front-door control list, or NULL. Names
# inside `mode_find` are checked against the consumer's own key set, so a
# misspelled sub-knob errors rather than silently fitting the default -- the
# same contract `tulpa_check_control()` gives the outer list.
#' @keywords internal
.nl_mode_find_tuning <- function(consumer, control = NULL) {
  keys <- names(.NL_MODE_FIND[[consumer]])
  if (is.null(keys)) {
    stop("Unknown mode-find consumer '", consumer, "'.", call. = FALSE)
  }
  tune <- stats::setNames(
    lapply(keys, function(k) .nl_mode_find(consumer, k)), keys)

  ov <- control[["mode_find"]]
  if (is.null(ov)) return(tune)
  if (!is.list(ov)) {
    stop("`control$mode_find` must be a list.", call. = FALSE)
  }
  if (length(ov) == 0L) return(tune)

  nm <- names(ov)
  if (is.null(nm) || any(!nzchar(nm))) {
    stop("every `control$mode_find` entry must be named.", call. = FALSE)
  }
  unknown <- setdiff(nm, keys)
  if (length(unknown)) {
    stop(sprintf(
      "Unknown control$mode_find knob(s): %s.\nAllowed: %s.",
      paste(sQuote(unknown, q = FALSE), collapse = ", "),
      paste(sort(keys), collapse = ", ")), call. = FALSE)
  }
  for (k in nm) {
    v <- ov[[k]]
    if (!is.numeric(v) || length(v) != 1L || !is.finite(v) || v <= 0) {
      stop(sprintf("`control$mode_find$%s` must be one positive finite number.",
                   k), call. = FALSE)
    }
    tune[[k]] <- v
  }
  tune
}

# Box-constrained L-BFGS-B with a numerical gradient. Returns the `optim()`
# result, or NULL if the call errored -- callers decide what an absent mode
# means for them.
#' @keywords internal
.nl_lbfgsb_mode_find <- function(par, fn, lower, upper, tuning,
                                 hessian = FALSE) {
  tryCatch(
    stats::optim(par = par, fn = fn,
                 method = "L-BFGS-B", lower = lower, upper = upper,
                 hessian = hessian,
                 control = list(factr = tuning$factr,
                                maxit = tuning$maxit,
                                ndeps = rep(tuning$ndeps, length(par)))),
    error = function(e) NULL
  )
}
