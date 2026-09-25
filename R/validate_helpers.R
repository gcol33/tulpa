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
#' coordinate column and the neighbour construction now
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
#' the kernels build from it read the same metric.
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
#' the GP field predictor read whatever coordinate dimension they are given.
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
#' is the same class of defect as an out-of-bounds coordinate-column read.
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

# Coordinates reaching a spatial field: every value finite, and -- where they
# are about to be standardised -- no column constant, since scale() divides by
# its SD. Each case used to reach a kernel as "NA/NaN/Inf in foreign function
# call (arg 1)" (gcol33/tulpa#909). `where` names the caller or the columns.
#' @keywords internal
#' @noRd
.check_coords_finite <- function(coords, where, scale = FALSE) {
  coords <- as.matrix(coords)
  bad <- !is.finite(suppressWarnings(as.numeric(coords)))
  if (any(bad)) {
    rows <- unique(((which(bad) - 1L) %% nrow(coords)) + 1L)
    stop(where, ": the coordinates contain ", sum(bad), " missing or ",
         "non-finite value(s) (NA / NaN / Inf), first at row ", rows[1L],
         ". Remove or impute them before fitting.", call. = FALSE)
  }
  if (isTRUE(scale) && nrow(coords) > 0L) {
    sds <- apply(coords, 2L, stats::sd)
    const <- which(!(sds > 0))
    if (length(const)) {
      nm <- colnames(coords)[const] %||% paste0("column ", const)
      stop(where, ": coordinate ", paste0("`", nm, "`", collapse = ", "),
           " is constant, so it cannot be standardised (scale_coords = TRUE ",
           "divides by its SD). Drop it from the coordinates, or pass ",
           "scale_coords = FALSE.", call. = FALSE)
    }
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
  .check_coords_finite(coords, "Coordinate columns", scale = scale_coords)
  if (isTRUE(scale_coords)) {
    coords <- .scale_coords_isotropic(coords)
  }
  coords
}

#' Standardize spatial coordinates by one common factor.
#'
#' Each column is centred on its own mean and every column is divided by the
#' SAME scale, the root mean of the column variances. `scale()` divided lon and
#' lat by their own spreads, which stretches one axis against the other and
#' turns an isotropic kernel anisotropic in the data's geometry
#' (gcol33/tulpa#907); one factor keeps every distance proportional to the
#' user's, so a fitted range converts back to data units by that one factor
#' (`.coord_scale()`). The attributes are the ones `scale()` sets --
#' `scaled:center` per column and `scaled:scale` (the common factor, repeated
#' per column) -- so a reader re-applying the training standardization to new
#' coordinates is unchanged.
#'
#' @param coords Numeric coordinate matrix.
#' @return The standardized matrix, carrying `scaled:center` / `scaled:scale`.
#' @keywords internal
#' @noRd
.scale_coords_isotropic <- function(coords) {
  coords <- as.matrix(coords)
  storage.mode(coords) <- "double"
  ctr <- colMeans(coords)
  s <- if (nrow(coords) > 1L) sqrt(mean(apply(coords, 2L, stats::var))) else NA
  if (!is.finite(s) || s <= 0) s <- 1
  out <- sweep(coords, 2L, ctr, "-") / s
  attr(out, "scaled:center") <- ctr
  attr(out, "scaled:scale") <- rep(s, ncol(coords))
  out
}

#' The factor a spec's coordinates were divided by (1 when they were not), which
#' converts a distance the kernel measured -- a range, a lengthscale -- back to
#' the user's coordinate units.
#' @param spec A validated spatial spec carrying `coords_matrix`.
#' @keywords internal
#' @noRd
.coord_scale <- function(spec) {
  s <- attr(spec$coords_matrix, "scaled:scale")
  if (!isTRUE(spec$scale_coords) || !length(s) || !is.finite(s[1L]) || s[1L] <= 0)
    return(1)
  s[1L]
}

#' A standardized coordinate matrix mapped back to the user's units, via the
#' attributes `.scale_coords_isotropic()` (or `scale()`) set; returned as is when
#' it carries none.
#' @keywords internal
#' @noRd
.unscale_coords <- function(coords) {
  if (is.null(coords)) return(NULL)
  ctr <- attr(coords, "scaled:center")
  scl <- attr(coords, "scaled:scale")
  if (is.null(ctr) || is.null(scl)) return(coords)
  out <- sweep(sweep(unclass(as.matrix(coords)), 2L, scl, "*"), 2L, ctr, "+")
  attr(out, "scaled:center") <- NULL
  attr(out, "scaled:scale") <- NULL
  out
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
#' That is strictly weaker than `.validate_adjacency()`, which `adjacency()` and
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
  # The gate is idempotent: a spec built by a spatial_*() constructor carries a
  # graph this has already passed, and the front door re-runs it on whatever
  # `spatial$adjacency` holds so a bare list reaches the same check
  # (gcol33/tulpa#670). Re-reporting the structural warnings on the second pass
  # would say the same thing twice about one graph.
  if (isTRUE(attr(x, "tulpa_adjacency_checked"))) return(x)
  if (!is.matrix(x) && !inherits(x, "Matrix")) {
    stop("`", arg, "` must be a matrix (dense or sparse).", call. = FALSE)
  }
  report <- .validate_adjacency(x)
  if (!report$square) {
    stop("`", arg, "` must be square (got ", report$nrow, " x ",
         report$ncol, ").", call. = FALSE)
  }
  if (!report$finite) {
    stop("`", arg, "` has ", report$n_nonfinite, " missing or non-finite ",
         "entr", if (report$n_nonfinite == 1L) "y" else "ies",
         " (NA / NaN / Inf); an adjacency must be 0 / 1 (or non-negative ",
         "weights) everywhere.", call. = FALSE)
  }
  if (!report$nonneg) {
    stop("`", arg, "` has ", report$n_negative, " negative entr",
         if (report$n_negative == 1L) "y" else "ies", "; an adjacency weight ",
         "must be non-negative (D - W is not a precision otherwise).",
         call. = FALSE)
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
  attr(x, "tulpa_adjacency_checked") <- TRUE
  x
}


# Scalar-argument checks shared by the constructors and post-fit methods, each
# naming the argument the caller set. `x` is refused unless it is ONE finite
# number, so an NA, a character, a vector or a NULL reaches the user as that
# message rather than as "missing value where TRUE/FALSE needed" from the first
# comparison it meets.
#' @keywords internal
.arg_repr <- function(x) {
  if (is.null(x)) return("NULL")
  s <- paste(deparse(x, width.cutoff = 60L, nlines = 1L), collapse = "")
  if (nchar(s) > 60L) paste0(substr(s, 1L, 57L), "...") else s
}

#' @keywords internal
.is_scalar_num <- function(x) {
  is.numeric(x) && length(x) == 1L && is.finite(x)
}

# One number in the open unit interval: a probability, an interval level, a
# window fraction.
#' @keywords internal
.check_unit_interval <- function(x, arg) {
  if (!.is_scalar_num(x) || x <= 0 || x >= 1) {
    stop("`", arg, "` must be a single number in (0, 1); got ", .arg_repr(x),
         ".", call. = FALSE)
  }
  invisible(x)
}

# One finite number (a location); `positive = TRUE` also demands > 0 (a scale).
#' @keywords internal
.check_scalar <- function(x, arg, positive = FALSE) {
  if (!.is_scalar_num(x) || (positive && x <= 0)) {
    stop("`", arg, "` must be a single ", if (positive) "positive" else "finite",
         " number; got ", .arg_repr(x), ".", call. = FALSE)
  }
  invisible(x)
}

# One whole number >= `min` (a count, a grid size). A fractional value is
# refused rather than truncated, which is what as.integer() would do with it.
#' @keywords internal
.check_count <- function(x, arg, min = 1L) {
  if (!.is_scalar_num(x) || x != round(x) || x < min) {
    stop("`", arg, "` must be a single whole number >= ", min, "; got ",
         .arg_repr(x), ".", call. = FALSE)
  }
  invisible(as.integer(x))
}

# Penalized-complexity anchors: P(sigma > U) = alpha calibrates the exponential
# rate lambda = -log(alpha) / U, which exists only for U > 0 and alpha in
# (0, 1). At alpha = 1 the rate is 0 and log(rate) is -Inf, so the prior is -Inf
# at every value; above 1 it is negative and the density grows without bound.
# Neither produces a message on its own -- it reaches a gradient as a number --
# so every front door that takes a pair checks it here, where the message can
# name the argument the user set. C++ carries the same predicate
# (pc_anchors_valid, src/pc_prior.h) for specs assembled without this door.
#' @keywords internal
.check_pc_anchors <- function(U, alpha, arg_U, arg_alpha, where,
                              tail = paste0("P(sigma > ", arg_U, ")")) {
  if (!is.numeric(U) || length(U) != 1L || !is.finite(U) || U <= 0) {
    stop(where, ": `", arg_U, "` must be a single positive number; got ",
         format(U), ".", call. = FALSE)
  }
  if (!is.numeric(alpha) || length(alpha) != 1L || !is.finite(alpha) ||
      alpha <= 0 || alpha >= 1) {
    stop(where, ": `", arg_alpha, "` is the tail probability ",
         tail, " and must lie in (0, 1); got ", format(alpha), ".",
         call. = FALSE)
  }
  invisible(TRUE)
}


# The number of adaptive Gauss-Hermite nodes per group: a single whole number
# >= 1. `as.integer()` truncated 2.5 to 2 and ran a quadrature the caller did
# not ask for (gcol33/tulpa#886). Returns it as an integer.
#' @keywords internal
.check_n_quad <- function(n_quad) {
  if (!is.numeric(n_quad) || length(n_quad) != 1L || !is.finite(n_quad) ||
      n_quad < 1 || n_quad != round(n_quad)) {
    stop("`n_quad` must be a single whole number >= 1; got ",
         paste(format(n_quad), collapse = ", "), ".", call. = FALSE)
  }
  as.integer(n_quad)
}


# A model with random effects only (`y ~ 0 + (1 | g)`) has an empty fixed
# design. The sampler and conditional-Laplace backends carry it; the fitters
# that summarize or integrate over a fixed-effect block do not, and failed on
# the empty block in their own words ("'names' attribute [1] must be the same
# length as the vector [0]", "subscript out of bounds", "0 x 0 matrix";
# gcol33/tulpa#881). They refuse it here, naming the backends that fit it.
#' @keywords internal
.require_fixed_effects <- function(X, where) {
  if (NCOL(X) >= 1L) return(invisible(TRUE))
  stop(where, "() needs at least one fixed-effect column, and the design has ",
       "none (a formula such as y ~ 0 + (1 | g)). Keep an intercept ",
       "(y ~ 1 + (1 | g)), or fit the random-effects-only model with mode = ",
       "'hmc', 'mala', 'imh_laplace', 'vi' or 'laplace'.", call. = FALSE)
}

# The LKJ(eta) density det(R)^(eta - 1) / c_d is a density only for eta > 0:
# its normaliser is a product of Beta functions B(eta + (d - k - 1) / 2, ...),
# which at d = 2 is B(eta, eta) and is not finite for eta <= 0. A non-positive
# shape therefore reached the outer integration as a NaN log-prior at every
# node and surfaced as "every integration node returned a non-finite
# log-marginal" (gcol33/tulpa#894).
#' @keywords internal
.check_lkj_eta <- function(eta, arg, where) {
  if (!is.numeric(eta) || length(eta) != 1L || !is.finite(eta) || eta <= 0) {
    stop(where, ": `", arg, "` is the LKJ shape and must be a single ",
         "positive number; got ", format(eta), ".", call. = FALSE)
  }
  invisible(TRUE)
}

# The same check for an anchor pair passed as one `c(U, alpha)` argument (the
# SPDE `prior_range` / `prior_sigma`), naming that argument and its elements.
# `param` and `lower` say which tail the pair calibrates: P(sigma > U) for a
# scale, P(range < U) for a range.
#' @keywords internal
.check_pc_anchor_pair <- function(pair, arg, where, param = "sigma",
                                  lower = FALSE) {
  if (!is.numeric(pair) || length(pair) != 2L) {
    stop(where, ": `", arg, "` must be a length-2 numeric c(U, alpha); got ",
         "length ", length(pair), ".", call. = FALSE)
  }
  .check_pc_anchors(pair[[1L]], pair[[2L]],
                    paste0(arg, "[1]"), paste0(arg, "[2]"), where,
                    tail = paste0("P(", param, if (lower) " < " else " > ",
                                  arg, "[1])"))
}
