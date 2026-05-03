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
