# Prediction of spatially and temporally varying coefficients at new
# coordinates and times. The fields themselves are drawn in C++
# (`cpp_varying_coef_draws()`, through `.varying_coef_field_draws()`); this file
# places `newdata` against what the fit was built on and assembles each draw's
# contribution to the linear predictor.

# The new-coordinate argument of `.varying_coef_field_draws()` for an SVC spec:
# the coordinate columns of `newdata`, standardized by the transform the fit
# applied to its own coordinates.
#' @keywords internal
.svc_query_coords <- function(info, newdata) {
  cv <- info$coord_vars
  miss <- setdiff(cv, names(newdata))
  if (length(miss)) {
    stop("newdata is missing the coordinate column(s) the spatially varying ",
         "coefficient is a field over: ", paste(miss, collapse = ", "),
         ". Pass include_field = FALSE for the prediction at the average ",
         "location (the fixed effects alone).", call. = FALSE)
  }
  nc <- cbind(as.numeric(newdata[[cv[1L]]]), as.numeric(newdata[[cv[2L]]]))
  .check_coords_finite(structure(nc, dimnames = list(NULL, cv)),
                       "newdata coordinates")
  ctr <- attr(info$coords_matrix, "scaled:center")
  scl <- attr(info$coords_matrix, "scaled:scale")
  if (isTRUE(info$scale_coords) && !is.null(ctr) && !is.null(scl)) {
    nc <- sweep(sweep(nc, 2L, ctr, "-"), 2L, scl, "/")
  }
  .coords_plain(nc)
}

# The new times of a temporal path -- a time-varying coefficient's
# (`.varying_coef_field_draws()`) or an additive temporal field's
# (`.field_temporal_rows()`): for each row of `newdata`, the fitted level it
# sits on (0-based, -1 for none), its position, and its group (0-based).
#
# `gp` is a field over time VALUES, so any time is a position, on the scale the
# kernel measured. `rw1` / `rw2` / `ar1` read the levels as consecutive grid
# positions, one step apart: a time past either end is a number of steps only
# when the levels are equally spaced, and a time strictly between two levels is
# a position only for `rw1`, whose walk is a Brownian motion read at the grid
# (the bridge between two levels is its value there). `rw2` and `ar1` define the
# path at grid positions alone, so they are refused between levels and off the
# step lattice past them. `info$label` names the term in a refusal (a
# varying coefficient's `temporal_tvc()` by default) and `info$noun` what it
# defines.
#' @keywords internal
.temporal_query <- function(info, newdata) {
  tv <- info$time_var
  st <- info$structure
  label <- info$label %||% sprintf('temporal_tvc(structure = "%s")', st)
  noun <- info$noun %||% "coefficient"
  if (!tv %in% names(newdata)) {
    stop("newdata is missing the time variable '", tv, "' the ", noun,
         " evolves over. Pass include_field = FALSE for the prediction ",
         "without it.", call. = FALSE)
  }
  tn <- newdata[[tv]]
  if (anyNA(tn)) {
    stop("newdata has missing values in the time variable '", tv, "'.",
         call. = FALSE)
  }
  n <- length(tn)
  group <- rep(0L, n)
  if (!is.null(info$group_var)) {
    gv <- info$group_var
    if (!gv %in% names(newdata)) {
      stop("newdata is missing the group variable '", gv, "' of the ",
           "group-specific ", noun, ".", call. = FALSE)
    }
    g <- match(as.character(newdata[[gv]]), info$group_levels)
    if (anyNA(g)) {
      stop("newdata names group level(s) the fit has no field for: ",
           paste(unique(as.character(newdata[[gv]])[is.na(g)]), collapse = ", "),
           ". The ", noun, " is a path per group, and a group the data never ",
           "held has no fitted path to condition on.", call. = FALSE)
    }
    group <- g - 1L
  }

  lv <- info$level_values
  n_t <- info$n_times
  if (is.null(lv)) {
    ex <- match(as.character(tn), info$time_levels)
    if (anyNA(ex)) {
      stop("The time variable '", tv, "' is a factor, which orders its levels ",
           "and gives them no spacing, so only fitted levels can be predicted ",
           "at; newdata holds ", sum(is.na(ex)), " other value(s).",
           call. = FALSE)
    }
    return(list(exact = ex - 1L, pos = as.numeric(ex - 1L), group = group))
  }

  t <- if (is.factor(tn) || is.character(tn)) {
    suppressWarnings(as.numeric(as.character(tn)))
  } else as.numeric(tn)
  if (anyNA(t)) {
    stop("newdata has value(s) of the time variable '", tv, "' that are not ",
         "numbers: ", paste(utils::head(unique(as.character(tn[is.na(t)])), 5L),
                            collapse = ", "), ".", call. = FALSE)
  }
  tol <- 1e-8 * max(1, diff(range(lv)))
  ex <- vapply(t, function(x) {
    k <- which(abs(lv - x) <= tol)
    if (length(k)) k[1L] else NA_integer_
  }, integer(1))
  exact <- as.integer(ifelse(is.na(ex), -1L, ex - 1L))

  if (identical(st, "gp")) {
    pos <- (t - (info$time_center %||% 0)) / (info$time_scale %||% 1)
    return(list(exact = exact, pos = pos, group = group))
  }

  pos <- ifelse(is.na(ex), NA_real_, ex - 1)
  inside <- is.na(ex) & t > lv[1L] & t < lv[n_t]
  if (any(inside)) {
    if (!identical(st, "rw1")) {
      stop(sprintf(paste0(
        "%s defines the %s at its fitted grid positions only, and %d new ",
        "time(s) fall between two fitted levels. Predict at fitted levels, or ",
        "refit with a first-order random walk (a Brownian walk, which has a ",
        "value between levels) or a Gaussian process over time values."),
        label, noun, sum(inside)), call. = FALSE)
    }
    k <- findInterval(t[inside], lv)
    pos[inside] <- (k - 1) + (t[inside] - lv[k]) / (lv[k + 1L] - lv[k])
  }
  outside <- is.na(ex) & !inside
  if (any(outside)) {
    dd <- diff(lv)
    if (n_t < 2L || max(abs(dd - dd[1L])) > 1e-8 * abs(dd[1L])) {
      stop(sprintf(paste0(
        "%d new time(s) lie beyond the fitted levels of '%s', whose spacing ",
        "is not uniform, so they are no whole number of grid steps from them: ",
        "%s reads each gap between levels as one step. Refit with a Gaussian ",
        "process over time values to predict there."), sum(outside), tv,
        label), call. = FALSE)
    }
    to <- t[outside]
    steps <- ifelse(to > lv[n_t], (n_t - 1) + (to - lv[n_t]) / dd[1L],
                    (to - lv[1L]) / dd[1L])
    if (!identical(st, "rw1")) {
      off <- abs(steps - round(steps)) > 1e-8
      if (any(off)) {
        stop(sprintf(paste0(
          "%s steps on the grid of the fitted levels (spacing %g), and %d new ",
          "time(s) beyond them fall off it. Predict at whole steps, or refit ",
          "with a first-order random walk or a Gaussian process over time ",
          "values."), label, dd[1L], sum(off)), call. = FALSE)
      }
      steps <- round(steps)
    }
    pos[outside] <- steps
  }
  list(exact = exact, pos = as.numeric(pos), group = group)
}

# The S x n contribution sum_j x_ij w_j(s_i, t_i) of a fit's spatially and
# temporally varying coefficient terms at `newdata` (the training rows when
# NULL), one row per stored draw in `rows` (all of them when NULL). At the
# training rows each field is the one the likelihood read at that draw; at new
# coordinates / times it is drawn from its conditional given that draw. NULL
# when the fit carries no varying coefficient.
#' @keywords internal
.varying_coef_eta_contrib <- function(object, newdata = NULL, rows = NULL) {
  sv <- .varying_coef_info(object, c("svc", "spatial"), "tulpa_svc")
  tv <- .varying_coef_info(object, c("tvc", "temporal"), "tulpa_tvc")
  if (is.null(sv) && is.null(tv)) return(NULL)

  new <- !is.null(newdata)
  vc <- .varying_coef_field_draws(
    object,
    svc_coords = if (!is.null(sv) && new) .svc_query_coords(sv, newdata),
    tvc_query  = if (!is.null(tv) && new) .temporal_query(tv, newdata),
    rows = rows)
  if (is.null(vc)) {
    stop("This fit carries a varying coefficient but not the sampler model ",
         "($model_inputs) and draws its field is read from.", call. = FALSE)
  }
  X <- if (new) .tulpa_fixed_design(object, newdata) else object$model_inputs$X
  term_cols <- function(idx, nm) {
    if (!new) return(as.matrix(X)[, idx, drop = FALSE])
    miss <- setdiff(nm, colnames(X))
    if (length(miss)) {
      stop("newdata cannot reproduce the varying-coefficient column(s): ",
           paste(miss, collapse = ", "), call. = FALSE)
    }
    X[, nm, drop = FALSE]
  }
  n <- nrow(X)
  out <- matrix(0, nrow(vc$svc %||% vc$tvc), n)

  if (!is.null(sv)) {
    Xs <- term_cols(sv$svc_indices, sv$svc_names)
    n_terms <- length(sv$svc_names)
    n_units <- ncol(vc$svc) / n_terms
    unit <- if (!new && n_units != n) sv$obs_to_loc else seq_len(n)
    for (j in seq_len(n_terms)) {
      W <- vc$svc[, (j - 1L) * n_units + unit, drop = FALSE]
      out <- out + sweep(W, 2L, Xs[, j], "*")
    }
  }
  if (!is.null(tv)) {
    Xt <- term_cols(tv$tvc_indices, tv$tvc_names)
    n_terms <- length(tv$tvc_names)
    for (j in seq_len(n_terms)) {
      cols <- if (new) (j - 1L) * n + seq_len(n) else
        ((tv$group_index - 1L) * n_terms + (j - 1L)) * tv$n_times + tv$time_index
      out <- out + sweep(vc$tvc[, cols, drop = FALSE], 2L, Xt[, j], "*")
    }
  }
  out
}

# Linear-predictor draws (S x n) of a varying-coefficient fit at `newdata` for
# predict(): the fixed part and the offset at each stored draw plus that draw's
# varying-coefficient contribution, with the conditional draws at new
# coordinates / times taken from a pinned, RNG-neutral seed. NULL for a fit with
# no varying coefficient.
#' @keywords internal
.predict_varying_coef_eta <- function(object, newdata, lp) {
  if (is.null(.varying_coef_info(object, c("svc", "spatial"), "tulpa_svc")) &&
      is.null(.varying_coef_info(object, c("tvc", "temporal"), "tulpa_tvc"))) {
    return(NULL)
  }
  .preserve_seed_in_frame()
  set.seed(.PREDICT_VARYING_COEF$seed)
  contrib <- .varying_coef_eta_contrib(object, newdata)
  beta <- .fixed_coef_draws(object)$beta
  E <- beta[, lp$count_names, drop = FALSE] %*% t(lp$X)
  sweep(E, 2L, rep_len(lp$offset, ncol(E)), "+") + contrib
}
