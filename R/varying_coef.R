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

# Shared print for svc/tvc posteriors. The two methods differ only in the
# title, the per-axis count line, the meta line, and the visualization word.
.print_varying_coef <- function(x, kind, axis_label, axis_value,
                                meta_label, meta_value, viz) {
  title <- paste0(kind, " coefficient posterior")
  cat(title, "\n", sep = "")
  cat(strrep("=", nchar(title)), "\n\n", sep = "")
  cat("Terms:", paste(x$term_names, collapse = ", "), "\n")
  cat(paste0(axis_label, ":"), axis_value, "\n")
  cat("Posterior draws:", x$n_draws, "\n")
  cat(paste0(meta_label, ":"), meta_value, "\n")
  cat("\nUse summary() for posterior summaries\n")
  cat("Use plot() for", viz, "visualization\n")
  invisible(x)
}

# Shared summary for svc/tvc posteriors. `lead_cols(j)` returns the
# variant-specific identifier columns (obs + coords for svc, time index +
# levels for tvc); mean / sd / quantile columns are appended identically.
.summary_varying_coef <- function(object, probs, n_terms, lead_cols,
                                  summary_class) {
  draws <- object$draws

  results <- lapply(seq_len(n_terms), function(j) {
    term_draws <- draws[, , j]
    lead <- lead_cols(j)
    stats <- data.frame(
      mean = colMeans(term_draws),
      sd = apply(term_draws, 2, sd),
      t(apply(term_draws, 2, quantile, probs = probs))
    )
    summaries <- cbind(lead, stats)
    q_start <- ncol(lead) + 3L
    names(summaries)[q_start:ncol(summaries)] <- paste0("q", probs * 100)
    rownames(summaries) <- NULL
    summaries
  })

  result <- do.call(rbind, results)

  structure(
    result,
    n_draws = object$n_draws,
    term_names = object$term_names,
    class = c(summary_class, "data.frame")
  )
}
