# Shared extractor for spatially- and temporally-varying coefficient
# posteriors. svc() and tvc() differ only in the fit slot they read, the field
# names inside it, the not-fitted message, and the posterior object they build;
# validation, the draws lookup, and term subsetting are identical. This is the
# single source of truth for both.
#
# build_result(info, draws, term_names) constructs the variant-specific
# posterior object from the (possibly subset) draws and names.
.extract_varying_coef <- function(object, terms, summary, probs,
                                  slot, info_class, not_fitted_msg,
                                  draws_field, names_field, build_result) {
  info <- object[[slot]]
  if (is.null(info) || !inherits(info, info_class)) {
    stop(not_fitted_msg, call. = FALSE)
  }

  term_names <- info[[names_field]]
  draws <- object$.internal[[draws_field]]
  if (is.null(draws)) {
    stop(toupper(slot), " draws not found in model output", call. = FALSE)
  }

  if (!is.null(terms)) {
    if (is.numeric(terms)) {
      term_idx <- terms
    } else if (is.character(terms)) {
      term_idx <- match(terms, term_names)
      if (any(is.na(term_idx))) {
        stop("Terms not found: ",
             paste(terms[is.na(term_idx)], collapse = ", "), call. = FALSE)
      }
    } else {
      stop("`terms` must be numeric or character", call. = FALSE)
    }
    draws <- draws[, , term_idx, drop = FALSE]
    term_names <- term_names[term_idx]
  }

  result <- build_result(info, draws, term_names)

  if (summary) {
    return(summary(result, probs = probs))
  }

  result
}
