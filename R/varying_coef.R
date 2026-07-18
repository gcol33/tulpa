# Shared extractor for spatially- and temporally-varying coefficient
# posteriors. svc() and tvc() differ only in the fit slot they read, the field
# names inside it, the not-fitted message, and the posterior object they build;
# validation, the draws lookup, and term subsetting are identical. This is the
# single source of truth for both.
#
# Resolve requested SVC/TVC term labels to column indices of the fitted design
# `coef_names`, given the model `data`. A label that names a design column
# directly (numeric covariate, or an already-expanded contrast column) matches
# as-is; a factor term is expanded through model.matrix so its treatment-contrast
# columns (e.g. `habitatB`, `habitatC`) resolve, instead of a bare-name match
# failing outright. `kind` ("SVC"/"TVC") only shapes the error message. Shared by
# validate_svc() and validate_tvc().
.resolve_varying_coef_columns <- function(term_labels, has_intercept, data,
                                          coef_names, kind) {
  idx <- integer(0)
  if (has_intercept) {
    ji <- match("(Intercept)", coef_names)
    if (!is.na(ji)) idx <- c(idx, ji)
  }
  for (lbl in term_labels) {
    j <- match(lbl, coef_names)
    if (!is.na(j)) { idx <- c(idx, j); next }
    # Expand a factor / non-column term to its treatment-contrast columns the
    # same way the fitted design was built (intercept present, then dropped).
    cols <- tryCatch(
      colnames(stats::model.matrix(
        stats::as.formula(paste0("~ ", lbl)), data))[-1L],
      error = function(e) character(0))
    jj <- match(cols, coef_names)
    jj <- jj[!is.na(jj)]
    if (length(jj) == 0L) {
      stop(sprintf("%s terms not found in design matrix: %s", kind, lbl),
           call. = FALSE)
    }
    idx <- c(idx, jj)
  }
  unique(idx)
}

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
