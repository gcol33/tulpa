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

# The fit slot a validated field spec lands in. tulpa() attaches every spatial
# spec at `$spatial` and every temporal one at `$temporal`, so a varying-
# coefficient spec is found there; a model package assembling its own fit may
# instead attach it under the accessor's own name (`$svc` / `$tvc`). Both are
# looked up and whichever carries `info_class` wins, so one accessor serves
# both assemblies.
.varying_coef_info <- function(object, slots, info_class) {
  for (s in slots) {
    info <- object[[s]]
    if (!is.null(info) && inherits(info, info_class)) return(info)
  }
  NULL
}

# The `<prefix>[k]` columns of a draw matrix, ordered by k rather than by the
# order they happen to sit in. The one place a latent block is read back out of
# `fit$draws` by the name the sampler gave it. NULL when there are none.
.draws_by_prefix <- function(draws, prefix) {
  if (is.null(draws)) return(NULL)
  cn <- colnames(draws)
  if (is.null(cn)) return(NULL)
  pat <- paste0("^", prefix, "\\[([0-9]+)\\]$")
  hit <- grep(pat, cn)
  if (!length(hit)) return(NULL)
  k <- as.integer(sub(pat, "\\1", cn[hit]))
  as.matrix(draws[, hit[order(k)], drop = FALSE])
}

# The varying-coefficient fields at every stored draw of a ModelData sampler
# fit, as the likelihood read them: `cpp_varying_coef_draws()` rebuilds the
# model the sampler ran on and reconstructs each draw's fields through the
# engine's own log-posterior state (the HSGP basis expansion, the non-centered
# NNGP transform, the centring that removes each block's level). Reading the
# draw columns directly would reshape whatever a backend stored there -- basis
# coefficients, a whitened z, an uncentred field -- as though it were the
# field.
#
# With no `svc_coords` / `tvc_query` the fields are returned at the training
# design: `$svc` is S x (n_svc * n_units), term-major, over the fitted
# locations (NNGP) or rows (HSGP), and `$tvc` is S x (n_groups * n_tvc *
# n_times), (group, term)-major over time. `svc_coords` (n_new x 2, on the
# fitted coordinate scale) and `tvc_query` (from `.tvc_query()`) ask for the
# fields at new coordinates / times instead, S x (n_terms * n_new). `rows`
# selects stored draws. NULL when the fit carries no sampler model.
#' @keywords internal
.varying_coef_field_draws <- function(object, svc_coords = NULL,
                                      tvc_query = NULL, rows = NULL) {
  if (!is.list(object$model_inputs) || !is.matrix(object$draws) ||
      nrow(object$draws) == 0L) {
    return(NULL)
  }
  D <- object$draws
  if (!is.null(rows)) D <- D[rows, , drop = FALSE]
  nn <- as.integer(object$model_inputs$svc_spec$nn %||% 15L)
  do.call(cpp_varying_coef_draws,
          c(list(draws = unname(as.matrix(D))), .sampler_model_args(object),
            list(svc_coords_new = svc_coords, tvc_new = tvc_query,
                 svc_nn = nn)))
}

# The engine's SVC fields as an [n_draws, n_units, n_terms] array: units run
# fastest and terms next in `$svc`, which is the array's own fill order.
#' @keywords internal
.svc_engine_draws <- function(object, info) {
  m <- .varying_coef_field_draws(object)$svc
  if (is.null(m)) return(NULL)
  n_terms <- length(info$svc_names)
  array(m, dim = c(nrow(m), ncol(m) / n_terms, n_terms))
}

# The engine's TVC fields as an [n_draws, n_groups * n_times, n_terms] array,
# a unit being one (group, time) pair, group-major. `$tvc` runs time fastest
# within a (group, term) block, group-major, so the columns are permuted into
# the array's (unit, term) order; with one group that is the identity.
#' @keywords internal
.tvc_engine_draws <- function(object, info) {
  m <- .varying_coef_field_draws(object)$tvc
  if (is.null(m)) return(NULL)
  n_terms <- length(info$tvc_names)
  n_t <- as.integer(info$n_times)
  n_g <- as.integer(info$n_groups %||% 1L)
  g <- rep(seq_len(n_g) - 1L, each = n_t)
  t <- rep(seq_len(n_t), times = n_g)
  cols <- unlist(lapply(seq_len(n_terms) - 1L,
                        function(j) (g * n_terms + j) * n_t + t))
  array(m[, cols, drop = FALSE], dim = c(nrow(m), n_g * n_t, n_terms))
}

# `engine_draws(object, info)` returns the [n_draws, n_units, n_terms] field
# array from `.varying_coef_field_draws()`, or NULL. A model package that
# assembles its own fit supplies the array itself at `.internal[[draws_field]]`.
.extract_varying_coef <- function(object, terms, summary, probs,
                                  slot, info_class, not_fitted_msg,
                                  draws_field, names_field, build_result,
                                  field_slot = NULL, engine_draws = NULL) {
  info <- .varying_coef_info(object, c(slot, field_slot), info_class)
  if (is.null(info)) {
    stop(not_fitted_msg, call. = FALSE)
  }

  term_names <- info[[names_field]]
  draws <- object$.internal[[draws_field]]
  if (is.null(draws) && !is.null(engine_draws)) {
    draws <- engine_draws(object, info)
  }
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

# Shared print for the classed tables the posterior accessors and diagnostics
# return (svc / tvc / temporal / spatiotemporal summaries, the empirical
# variogram): one line naming the table and its draw count, then the rows.
.print_effect_table <- function(x, header, ...) {
  nd <- attr(x, "n_draws")
  if (!is.null(nd)) header <- paste0(header, ", ", nd, " posterior draws")
  cat(header, "\n", sep = "")
  print.data.frame(x, ...)
  invisible(x)
}
