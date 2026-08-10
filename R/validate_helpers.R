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

#' Parse a coordinate specification into column names.
#'
#' The single body behind `spatial_gp()`, `spatial_multiscale()` and
#' `spatial_svc()`, which carried three verbatim copies of it.
#'
#' `allow_nd` is the arity policy, and it differs by what the spec ends up in
#' rather than by taste. The nested-Laplace NNGP/GP kernels read every
#' coordinate column (gcol33/tulpa#389) and the neighbour construction now
#' matches them, so a 1-D domain -- a transect, a depth profile -- and a 3-D one
#' are real models there. The HSGP basis and every SAMPLER spec store
#' coordinates at a fixed 2-D stride and cannot represent anything else.
#'
#' @param coords A formula (`~ lon + lat`) or a character vector of column names.
#' @param what What to name in the error, e.g. `"gp()"`.
#' @param allow_nd If `TRUE`, any dimension `>= 1`; if `FALSE`, exactly 2.
#' @keywords internal
#' @noRd
.parse_coord_spec <- function(coords, what, allow_nd = FALSE) {
  if (inherits(coords, "formula")) {
    coord_vars <- all.vars(coords)
  } else if (is.character(coords) && length(coords) >= 1L) {
    coord_vars <- coords
  } else {
    stop(what, ": `coords` must be a formula (e.g. ~ lon + lat) or a character ",
         "vector of coordinate column names.", call. = FALSE)
  }
  if (!length(coord_vars)) {
    stop(what, " requires at least one coordinate variable.", call. = FALSE)
  }
  if (!allow_nd && length(coord_vars) != 2L) {
    stop(what, " requires exactly 2 coordinate variables (x, y); got ",
         length(coord_vars), ".", call. = FALSE)
  }
  coord_vars
}

#' Euclidean distance from every row of a coordinate matrix to one point.
#'
#' Over every column the matrix carries, so the coordinate dimension is the
#' caller's. The neighbour SELECTION this serves and the neighbour COVARIANCE
#' the kernels build from it read the same metric (gcol33/tulpa#389).
#'
#' @param mat Coordinate matrix `[n x d]`.
#' @param pt Length-`d` coordinate.
#' @keywords internal
#' @noRd
.coord_dist_to <- function(mat, pt) {
  # `rep(pt, each = nrow(mat))` lays the point out column-major, which is the
  # layout `mat` already has, so this is one vectorised subtraction rather than
  # a sweep.
  sqrt(rowSums((mat - rep(as.numeric(pt), each = nrow(mat)))^2))
}

#' Strip a coordinate matrix's attributes without changing its shape.
#'
#' A coordinate matrix reaches C++ as a plain numeric matrix, so `scale()`'s
#' centre/scale attributes and any dimnames have to come off first. Its ARITY is
#' data, not something this step decides: the nested-Laplace NNGP/GP kernels and
#' the GP field predictor read whatever coordinate dimension they are given
#' (gcol33/tulpa#389).
#' @keywords internal
#' @noRd
.coords_plain <- function(x) {
  x <- as.matrix(x)
  matrix(as.numeric(x), nrow(x), ncol(x))
}

#' Strip a coordinate matrix's attributes and require exactly two columns.
#'
#' For the paths whose downstream storage is 2-D by layout -- every sampler
#' spec, whose `GPData::coords` and siblings are flat buffers at stride 2, and
#' the HSGP 2-D basis. Those sites used to be handed
#' `matrix(as.numeric(x), n, 2)`, which does not check the arity, it IMPOSES it:
#' an `n x 1` matrix is recycled so that column 2 equals column 1 and every
#' location lands on the diagonal, and an `n x 3` matrix is truncated to its
#' first two columns. Both are a different geometry accepted in silence, which
#' is the same class of defect as the out-of-bounds read in gcol33/tulpa#389.
#'
#' @param x Coordinate matrix.
#' @param what What to name in the error, e.g. `"gp()"`.
#' @keywords internal
#' @noRd
.coords_2col <- function(x, what) {
  x <- as.matrix(x)
  if (ncol(x) != 2L) {
    stop(what, " requires a coordinate matrix with exactly 2 columns (x, y); ",
         "got ", ncol(x), ". This path stores coordinates at a fixed 2-D ",
         "stride. The nested-Laplace NNGP/GP kernels accept any number of ",
         "coordinate columns.", call. = FALSE)
  }
  matrix(as.numeric(x), nrow(x), 2L)
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
