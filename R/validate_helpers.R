#' Internal validation helpers
#'
#' Small shared helpers used by `validate_*()` functions across the
#' spatial / temporal / SVC / TVC specs. Centralised here to keep the
#' per-spec validators thin and prevent drift between near-identical
#' column-existence checks and coordinate preparation blocks.
#'
#' @name validate_helpers
#' @keywords internal
NULL

#' Assert that named columns exist in `data`.
#'
#' @param vars Character vector of column names that must be present.
#' @param data Data frame.
#' @param role Short label used in the error message (e.g. "Coordinate",
#'   "Temporal", "SVC covariate").
#' @return Invisibly `TRUE` on success; throws otherwise.
#' @keywords internal
#' @noRd
assert_columns_exist <- function(vars, data, role = "Required") {
  missing_cols <- setdiff(vars, names(data))
  if (length(missing_cols) > 0) {
    stop(sprintf("%s column(s) not found in data: %s",
                 role, paste(missing_cols, collapse = ", ")),
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Extract a coordinate matrix, check for missing values, optionally scale.
#'
#' Wraps the coord-validation pattern shared by `validate_hsgp()` and
#' `validate_hsgp_multiscale()`.
#'
#' @param coord_vars Character vector of coordinate column names.
#' @param data Data frame.
#' @param scale_coords If `TRUE`, applies `scale()` to the extracted matrix.
#' @return A numeric matrix `[n_obs x length(coord_vars)]`.
#' @keywords internal
#' @noRd
prepare_coords <- function(coord_vars, data, scale_coords = FALSE) {
  assert_columns_exist(coord_vars, data, role = "Coordinate")
  coords <- as.matrix(data[, coord_vars, drop = FALSE])
  if (any(is.na(coords))) {
    stop("Coordinate columns contain missing values", call. = FALSE)
  }
  if (isTRUE(scale_coords)) {
    coords <- scale(coords)
  }
  coords
}

#' Coerce a variable argument given as a formula or a string to a bare name.
#'
#' The field constructors all accept `~ x` or `"x"` for their variable
#' arguments. The block was copied verbatim into each one, so a constructor
#' could silently accept what its siblings reject.
#'
#' @param x A one-sided formula (`~ time`) or a length-1 character vector.
#' @param arg Name of the argument, used in the error messages.
#' @param example Example formula shown in the error (e.g. "~ time").
#' @return The bare variable name as a length-1 character vector, or `NULL`
#'   when `x` is `NULL`.
#' @keywords internal
#' @noRd
.coerce_var_arg <- function(x, arg, example = NULL) {
  if (is.null(x)) return(NULL)
  if (inherits(x, "formula")) {
    v <- all.vars(x)
    if (length(v) != 1) {
      stop("`", arg, "` formula must specify exactly 1 variable", call. = FALSE)
    }
    return(v)
  }
  if (!is.character(x) || length(x) != 1) {
    stop("`", arg, "` must be a formula", if (!is.null(example))
           paste0(" (", example, ")"), " or single character string",
         call. = FALSE)
  }
  x
}

#' Warn that a latent field was declared non-shared across processes.
#'
#' Every field constructor carried its own copy of this warning. Two had
#' drifted to a one-sentence form, and `temporal_ar1()` / `spatial()` had none
#' at all, so identical `shared = FALSE` input warned or stayed silent
#' depending only on which prior was used.
#'
#' @param label Field name for the first sentence (e.g. "temporal effects").
#' @param effects Field name for the advice sentence; defaults to `label`.
#' @keywords internal
#' @noRd
.warn_nonshared <- function(label, effects = label) {
  warning(
    "Non-shared ", label,
    " (shared = FALSE) means effects are not shared across processes.\n",
    "Consider whether ", effects, " should be shared between\n",
    "processes if shared confounding structure is expected.",
    call. = FALSE
  )
}

#' Validate an adjacency argument passed to a field constructor.
#'
#' `spatial_car()`, `spatial_bym2()` and `spatial()` each carried their own
#' inline block checking only matrix-ness, squareness and exact dense symmetry.
#' That is strictly weaker than [.validate_adjacency()], which `adjacency()` and
#' `check_adjacency()` already use: a raw matrix with self-loops, non-binary
#' weights or isolated nodes was reported by `check_adjacency()` and accepted
#' silently by the constructors, which then built an improper field. The inline
#' copies also coerced sparse graphs to dense to test symmetry (O(n^2) memory)
#' and demanded exact symmetry, rejecting float-rounded matrices that
#' `check_adjacency()` accepts.
#'
#' A `tulpa_adjacency` object is unwrapped and trusted -- it was validated at
#' construction.
#'
#' @param x The adjacency argument: a matrix, a `Matrix`, or a
#'   `tulpa_adjacency`.
#' @param arg Name of the argument, used in the messages.
#' @return The bare adjacency matrix.
#' @keywords internal
#' @noRd
.validate_adjacency_arg <- function(x, arg = "adjacency") {
  if (inherits(x, "tulpa_adjacency")) return(x$adjacency)
  if (!is.matrix(x) && !inherits(x, "Matrix")) {
    stop("`", arg, "` must be a matrix (dense or sparse).", call. = FALSE)
  }
  report <- .validate_adjacency(x)
  if (!report$square) {
    stop("`", arg, "` must be square (got ", report$nrow, " x ",
         report$ncol, ").", call. = FALSE)
  }
  if (!report$symmetric) {
    stop("`", arg, "` must be symmetric (max |W - t(W)| = ",
         signif(report$asymmetry, 3), ").", call. = FALSE)
  }
  # Structural issues do not make the graph unusable, so they warn rather than
  # error -- matching check_adjacency() so the same graph reports the same way
  # whichever door it enters through.
  if (!report$zero_diag) {
    warning("`", arg, "` has non-zero diagonal entries (self-loops): ",
            report$n_self, " node(s); zeroing the diagonal. The graph is used ",
            "as an off-diagonal adjacency; a self-loop would corrupt the ",
            "ICAR/CAR precision.", call. = FALSE)
    diag(x) <- 0
  }
  if (!report$binary) {
    warning("`", arg, "` has entries other than 0/1 (weighted graph).",
            call. = FALSE)
  }
  if (report$n_isolated > 0L) {
    warning(report$n_isolated, " isolated node(s) with no neighbours; an ",
            "ICAR/CAR field is improper on disconnected nodes.",
            call. = FALSE)
  }
  x
}
