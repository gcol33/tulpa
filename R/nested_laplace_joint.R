#' Joint multi-likelihood nested Laplace approximation
#'
#' @description
#' Outer-grid nested Laplace driver for *joint* models — multiple response
#' arms sharing one latent prior block, parameterized as a per-arm field
#' amplitude (sigma) on a unit-precision latent.
#'
#' Supported priors (Phase 1c):
#'  * `"bym2"`       — outer grid over `(sigma_spatial, rho [, sigma_pos])`.
#'                     Latent: `phi (n_s) | theta (n_s)` with unit-precision
#'                     ICAR + iid components.
#'  * `"icar"`       — outer grid over `(sigma_spatial [, sigma_pos])`.
#'                     Latent: `phi (n_s)` with unit-precision ICAR.
#'  * `"car_proper"` — outer grid over `(sigma_spatial, rho_car [, sigma_pos])`.
#'                     Latent: `phi (n_s)` with `Q = D - rho_car * W`.
#'
#' Other backends (NNGP, HSGP, RW1/2, AR1) follow the same interface and
#' land under Phase 3.
#'
#' @section Per-arm sigma parameterization (gcol33/tulpa#18):
#' Each arm's linear predictor reads
#' \deqn{\eta_{arm} = X_{arm} \beta_{arm} + \sigma_{arm} \cdot z_s,}
#' where \eqn{z_s} is a unit-precision latent (ICAR(tau=1) for ICAR/BYM2,
#' or the BYM2 mix; CAR_proper uses the structure of
#' \eqn{D - \rho_{car} W}). Non-copy arms see \eqn{\sigma_{arm}} from the
#' outer-grid `sigma_grid` axis (\eqn{\sigma_{occ}}). The copy arm sees
#' \eqn{\sigma_{arm}} from `copy$sigma_pos_grid` (\eqn{\sigma_{pos}}). The
#' Cartesian product is over `(sigma_occ, [rho/rho_car,] sigma_pos[, phi])`.
#' Anchoring each arm's field amplitude in its own likelihood breaks the
#' alpha-sigma identifiability ridge at small `n_pos`. The classical
#' copy-scale is recovered post-hoc as
#' \eqn{\alpha = \sigma_{pos} / \sigma_{occ}} on the joint posterior.
#'
#' @param responses A named list of arm specs (length >= 1). Each arm:
#'   * `y`           — numeric `[N_arm]` response.
#'   * `n_trials`    — integer `[N_arm]` (use `rep(1L, N_arm)` for non-binomial).
#'   * `X`           — numeric matrix `[N_arm x p_arm]` fixed-effects design.
#'   * `spatial_idx` — integer `[N_arm]`, 1-based map obs -> spatial unit.
#'   * `re_idx`      — optional numeric `[N_arm]` 1-based RE group index;
#'                     defaults to `rep(0, N_arm)` (no RE).
#'   * `n_re_groups` — optional integer (default `0L`).
#'   * `sigma_re`    — optional numeric (default `1`); ignored when
#'                     `n_re_groups == 0`.
#'   * `family`      — one of `"binomial"`, `"gaussian"`, `"poisson"`,
#'                     `"neg_binomial_2"`, `"beta"`, ... (see
#'                     [tulpa_nested_laplace()] for the full list).
#'   * `phi`         — numeric dispersion (gaussian SD, negbin size, beta
#'                     precision); default `1`.
#'
#' @param prior A list describing the shared latent prior block. Required
#'   field `type`. Backend-specific fields:
#'   * **bym2**: `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`,
#'     `n_neighbors`, `scale_factor` (default `1`); optional `sigma_grid`
#'     (donor-arm field amplitude, default 5 log-spaced values in
#'     `[0.1, 3]`), `rho_grid` (default `c(0.2, 0.5, 0.8, 0.95)`).
#'   * **icar**: `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`,
#'     `n_neighbors`; optional `sigma_grid` (default 5 log-spaced values
#'     in `[0.1, 3]`).
#'   * **car_proper**: same as icar plus `rho_car_grid`
#'     (default `c(0.5, 0.8, 0.95, 0.99)`).
#'
#' @param copy Optional list controlling the copy arm:
#'   * `arm`            — name (or 1-based index) of the copy arm.
#'   * `sigma_pos_grid` — numeric grid for that arm's field amplitude
#'                        \eqn{\sigma_{pos}}; default mirrors the donor
#'                        `sigma_grid` axis.
#'   When `NULL` (default), no copy scaling is applied — all arms share
#'   the donor `sigma_grid` axis.
#'
#' @param max_iter,tol Inner Newton iteration budget and tolerance.
#' @param n_threads OpenMP threads.
#' @param x_init Optional warm-start for the first grid point's inner solve.
#' @param verbose Currently a no-op; reserved.
#' @param store_Q If `TRUE`, also return the per-grid joint precision Q
#'   (lower triangle, CSC) as `Q_csc_p_per_grid`, `Q_csc_i_per_grid`,
#'   `Q_csc_x_per_grid`, `Q_csc_n`. Lets callers compute INLA-style
#'   total-variance posterior moments (`Var-of-means + Mean-of-Var`) on
#'   inner latent coordinates such as fixed-effect betas. Default `FALSE`
#'   to keep the result lightweight.
#' @param adaptive_grid Logical (default `FALSE`). When `TRUE`, a mode-
#'   tracked 1D refinement pass is triggered on any hyperparameter axis
#'   whose marginal posterior weight on the boundary point(s) exceeds
#'   `adaptive_grid_edge_thresh`. New points are appended on that axis
#'   (one densification between the boundary and its neighbour, two
#'   outward extensions beyond the boundary on a log-spaced axis) and
#'   the kernel is evaluated at each new point paired with the modal
#'   other-axis values from the boundary cell — *not* the full
#'   cartesian product. Each slice cell carries a calibration term
#'   `log S_b` so it contributes to the joint softmax on the
#'   *marginal* scale (where `S_b` is the integrated relative weight
#'   of the other axes at the boundary). Cost per refinement is
#'   `O(n_new_points)` kernel solves, not
#'   `O(n_new_points * prod(other_axis_sizes))`. Fixes posterior CI
#'   under-coverage when truth sits near or at the user's grid edge.
#'   Opt-in for now; defaults to `FALSE` to preserve legacy fixed-grid
#'   behaviour for existing callers.
#' @param adaptive_grid_edge_thresh Numeric (default `0.02`). Refinement
#'   triggers when the per-axis edge score on the boundary of the
#'   refinable axis (currently `sigma_pos` only) exceeds this value. The
#'   score is
#'   `max(marginal_weight_at_boundary,
#'        exp(max_log_marginal_at_boundary - max_log_marginal_overall))`,
#'   i.e. the larger of the integrated weight on the boundary level and
#'   the relative integrand density at the boundary level. Catches both
#'   boundary pile-up (truth lies *at* a grid endpoint, weight is heavy
#'   there) and integrand truncation (integrand still has appreciable
#'   density at the boundary, but the cell width is so narrow it gets
#'   little integrated weight). The default `0.02` corresponds to ~4 log
#'   units of decay before refinement stops. Lower the threshold to
#'   refine more aggressively at the cost of extra kernel passes.
#' @param adaptive_grid_max_passes Integer (default `1L`). Maximum number
#'   of refinement passes. One pass typically suffices; two is rarely
#'   useful and inflates runtime.
#'
#' @return A list of class `c("tulpa_nested_laplace_joint",
#'   "tulpa_nested_laplace", "list")` with:
#'   * `theta_grid`, `theta_names` — outer-grid hyperparameter values
#'     (includes the derived `alpha` column when `copy` is set).
#'   * `log_marginal`, `weights` — per-grid-point log-marginal and integration
#'      weights (sum to 1).
#'   * `theta_mean`, `theta_sd` — posterior moments per hyperparameter,
#'      including the derived `alpha` moment computed from
#'      `sigma_pos / sigma_occ` over the joint grid.
#'   * `modes` — `[n_grid x n_x]` matrix of inner modes.
#'   * `n_iter` — inner Newton iterations per grid point.
#'   * `arm_layout` — list with per-arm `beta_start`, `re_start`,
#'      spatial offset(s) and `n_x` for decoding modes.
#'   * `prior`, `responses`, `copy` — echoed inputs.
#'   * `adaptive_grid_info` — when `adaptive_grid = TRUE`, a list with
#'      `triggered_axes` (character) and `n_points_added` (integer)
#'      describing the refinement passes. NULL otherwise.
#'
#' @seealso [tulpa_nested_laplace()] for the single-arm engine.
#' @export
tulpa_nested_laplace_joint <- function(responses,
                                       prior,
                                       copy = NULL,
                                       phi_grid = NULL,
                                       max_iter = 50L, tol = 1e-6,
                                       n_threads = 1L,
                                       x_init = NULL, verbose = FALSE,
                                       store_Q = FALSE,
                                       adaptive_grid = FALSE,
                                       adaptive_grid_edge_thresh = 0.02,
                                       adaptive_grid_max_passes = 1L) {
    if (!is.list(responses) || length(responses) < 1L) {
        stop("`responses` must be a non-empty list of arm specs.", call. = FALSE)
    }
    if (!is.list(prior) || is.null(prior$type)) {
        stop("`prior` must be a list with a `type` field.", call. = FALSE)
    }
    type <- tolower(prior$type)
    backend <- .joint_backends[[type]]
    if (is.null(backend)) {
        stop("tulpa_nested_laplace_joint(): unsupported prior$type '", type,
             "'. Supported: ",
             paste(shQuote(names(.joint_backends)), collapse = ", "),
             ".", call. = FALSE)
    }

    arms <- lapply(seq_along(responses), function(k) {
        a <- responses[[k]]
        .normalise_joint_arm(a, k)
    })

    cp <- .resolve_copy(copy, responses, prior, type)
    arm_names <- names(responses) %||% paste0("arm", seq_along(responses))
    phi_axes <- .normalise_phi_grid(phi_grid, arm_names)
    grids <- backend$build_grids(prior, cp$has_copy, cp$sigma_pos_grid, phi_axes)

    res <- backend$call_kernel(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, isTRUE(store_Q),
                                arm_names = arm_names)

    # Adaptive grid refinement. Detect heavy boundary mass on any axis
    # and append cartesian-product points covering an interior bisection
    # plus an outward extension on that axis. The merged grid is fed to
    # the same `backend$call_kernel`, and the C++ kernel is shape-agnostic
    # (it just consumes paired hyperparameter vectors of equal length),
    # so the legacy fixed-grid path and the refined path share a single
    # implementation — no primary-and-fallback branching.
    refine_info <- NULL
    if (isTRUE(adaptive_grid)) {
        refined <- .adaptive_refine_pass(
            grids       = grids,
            res         = res,
            backend     = backend,
            arms        = arms, prior = prior, cp = cp,
            max_iter = max_iter, tol = tol, n_threads = n_threads,
            x_init = x_init, store_Q = store_Q,
            edge_thresh = adaptive_grid_edge_thresh,
            max_passes  = adaptive_grid_max_passes,
            arm_names   = arm_names
        )
        grids       <- refined$grids
        res         <- refined$res
        refine_info <- refined$info
    }

    res$theta_grid  <- backend$theta_grid(grids, cp$has_copy)
    res$theta_names <- colnames(res$theta_grid)
    res$weights     <- .nl_normalise_weights(res$log_marginal)
    res             <- .nl_posterior_moments(res, paste0("joint_", type))
    res             <- .joint_recalibrate_axis_moments(res)
    res             <- .joint_attach_alpha_moments(res, cp$has_copy)
    res$arm_layout  <- backend$layout(arms, prior)
    res$prior       <- prior
    res$responses   <- responses
    res$copy        <- copy
    res$adaptive_grid_info <- refine_info
    class(res) <- c("tulpa_nested_laplace_joint", "tulpa_nested_laplace", "list")
    res
}


# --- backend dispatch table --------------------------------------------------
#
# Adding a new joint backend means appending one entry here. Each entry
# bundles the four backend-specific concerns: grid construction, kernel call,
# theta-grid materialisation for the result, and latent-layout metadata.

.joint_backends <- list(
    bym2 = list(
        build_grids = function(prior, has_copy, sigma_pos_axis, phi_axes = NULL) {
            sigma_axis <- prior$sigma_grid %||%
                exp(seq(log(0.1), log(3), length.out = 5))
            rho_axis <- prior$rho_grid %||% c(0.2, 0.5, 0.8, 0.95)
            .joint_cartesian(list(sigma = sigma_axis, rho = rho_axis),
                              has_copy, sigma_pos_axis, phi_axes)
        },
        call_kernel = function(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, store_Q = FALSE,
                                arm_names = NULL) {
            cpp_nested_laplace_joint_bym2(
                arms_list       = arms,
                copy_arm        = as.integer(cp$copy_arm_zero),
                n_spatial_units = as.integer(prior$n_spatial_units),
                adj_row_ptr     = as.integer(prior$adj_row_ptr),
                adj_col_idx     = as.integer(prior$adj_col_idx),
                n_neighbors     = as.integer(prior$n_neighbors),
                scale_factor    = as.numeric(prior$scale_factor %||% 1.0),
                sigma_occ_grid  = as.numeric(grids$sigma),
                rho_grid        = as.numeric(grids$rho),
                sigma_pos_grid  = as.numeric(grids$sigma_pos),
                max_iter   = as.integer(max_iter),
                tol        = as.numeric(tol),
                n_threads  = as.integer(n_threads),
                x_init_nullable = x_init,
                store_Q    = isTRUE(store_Q),
                phi_grid_per_arm = .joint_phi_grid_per_arm(grids, arm_names)
            )
        },
        theta_grid = function(grids, has_copy) {
            base <- if (has_copy) {
                cbind(sigma_occ = grids$sigma, rho = grids$rho,
                      sigma_pos = grids$sigma_pos)
            } else {
                cbind(sigma_occ = grids$sigma, rho = grids$rho)
            }
            .append_phi_columns(base, grids)
        },
        layout = function(arms, prior) {
            .joint_layout(arms, prior$n_spatial_units, n_spatial_blocks = 2L,
                          spatial_block_names = c("phi_start", "theta_start"))
        }
    ),

    icar = list(
        build_grids = function(prior, has_copy, sigma_pos_axis, phi_axes = NULL) {
            sigma_axis <- prior$sigma_grid %||%
                exp(seq(log(0.1), log(3), length.out = 5))
            .joint_cartesian(list(sigma = sigma_axis), has_copy,
                              sigma_pos_axis, phi_axes)
        },
        call_kernel = function(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, store_Q = FALSE,
                                arm_names = NULL) {
            cpp_nested_laplace_joint_icar(
                arms_list       = arms,
                copy_arm        = as.integer(cp$copy_arm_zero),
                n_spatial_units = as.integer(prior$n_spatial_units),
                adj_row_ptr     = as.integer(prior$adj_row_ptr),
                adj_col_idx     = as.integer(prior$adj_col_idx),
                n_neighbors     = as.integer(prior$n_neighbors),
                sigma_occ_grid  = as.numeric(grids$sigma),
                sigma_pos_grid  = as.numeric(grids$sigma_pos),
                max_iter   = as.integer(max_iter),
                tol        = as.numeric(tol),
                n_threads  = as.integer(n_threads),
                x_init_nullable = x_init,
                store_Q    = isTRUE(store_Q),
                phi_grid_per_arm = .joint_phi_grid_per_arm(grids, arm_names)
            )
        },
        theta_grid = function(grids, has_copy) {
            base <- if (has_copy) {
                cbind(sigma_occ = grids$sigma, sigma_pos = grids$sigma_pos)
            } else {
                cbind(sigma_occ = grids$sigma)
            }
            .append_phi_columns(base, grids)
        },
        layout = function(arms, prior) {
            .joint_layout(arms, prior$n_spatial_units, n_spatial_blocks = 1L,
                          spatial_block_names = "phi_start")
        }
    ),

    car_proper = list(
        build_grids = function(prior, has_copy, sigma_pos_axis, phi_axes = NULL) {
            sigma_axis   <- prior$sigma_grid %||%
                exp(seq(log(0.1), log(3), length.out = 5))
            rho_car_axis <- prior$rho_car_grid %||% c(0.5, 0.8, 0.95, 0.99)
            .joint_cartesian(list(sigma = sigma_axis, rho_car = rho_car_axis),
                              has_copy, sigma_pos_axis, phi_axes)
        },
        call_kernel = function(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, store_Q = FALSE,
                                arm_names = NULL) {
            cpp_nested_laplace_joint_car_proper(
                arms_list       = arms,
                copy_arm        = as.integer(cp$copy_arm_zero),
                n_spatial_units = as.integer(prior$n_spatial_units),
                adj_row_ptr     = as.integer(prior$adj_row_ptr),
                adj_col_idx     = as.integer(prior$adj_col_idx),
                n_neighbors     = as.integer(prior$n_neighbors),
                sigma_occ_grid  = as.numeric(grids$sigma),
                rho_car_grid    = as.numeric(grids$rho_car),
                sigma_pos_grid  = as.numeric(grids$sigma_pos),
                max_iter   = as.integer(max_iter),
                tol        = as.numeric(tol),
                n_threads  = as.integer(n_threads),
                x_init_nullable = x_init,
                store_Q    = isTRUE(store_Q),
                phi_grid_per_arm = .joint_phi_grid_per_arm(grids, arm_names)
            )
        },
        theta_grid = function(grids, has_copy) {
            base <- if (has_copy) {
                cbind(sigma_occ = grids$sigma, rho_car = grids$rho_car,
                      sigma_pos = grids$sigma_pos)
            } else {
                cbind(sigma_occ = grids$sigma, rho_car = grids$rho_car)
            }
            .append_phi_columns(base, grids)
        },
        layout = function(arms, prior) {
            .joint_layout(arms, prior$n_spatial_units, n_spatial_blocks = 1L,
                          spatial_block_names = "phi_start")
        }
    )
)


# --- helpers -----------------------------------------------------------------

# Cartesian product over a named list of spatial axes plus an optional
# sigma_pos axis (copy-arm field amplitude) and optional per-arm phi axes.
# `phi_axes` is a list keyed by arm name; entries are either NULL/empty (no
# axis for that arm) or numeric vectors that become a new outer-grid axis
# named `phi_<arm>`. Returns a named list of paired vectors of identical
# length, ready to feed the C++ kernel. Phi axes vary slowest (added last)
# so within-spatial-block warm-starts stay good.
.joint_cartesian <- function(axes, has_copy, sigma_pos_axis, phi_axes = NULL) {
    full <- if (has_copy) c(axes, list(sigma_pos = sigma_pos_axis)) else axes
    if (!is.null(phi_axes)) {
        active <- phi_axes[vapply(phi_axes, length, integer(1)) > 0L]
        if (length(active) > 0L) {
            names(active) <- paste0("phi_", names(active))
            full <- c(full, active)
        }
    }
    gr <- do.call(expand.grid,
                  c(full, list(KEEP.OUT.ATTRS = FALSE,
                                stringsAsFactors = FALSE)))
    out <- as.list(gr)
    if (!has_copy) out$sigma_pos <- numeric(0)
    out
}

# Append any `phi_<arm>` columns from `grids` onto a backend's spatial
# theta_grid matrix so downstream posterior-moment helpers see phi as a
# regular hyperparameter axis.
.append_phi_columns <- function(base, grids) {
    phi_cols <- grep("^phi_", names(grids), value = TRUE)
    if (length(phi_cols) == 0L) return(base)
    extra <- do.call(cbind, lapply(phi_cols, function(c) {
        out <- as.numeric(grids[[c]]); attr(out, "name") <- c; out
    }))
    colnames(extra) <- phi_cols
    cbind(base, extra)
}

# Build the `phi_grid_per_arm` argument for the C++ kernels from a
# Cartesian-product `grids` list and arm names. Returns a list of length
# `n_arms`: entry k is either `NULL` (no phi axis for that arm — kernel
# uses the parse-time scalar phi) or a NumericVector of length n_grid
# matching the flat outer-grid size. Phi columns in `grids` follow the
# `phi_<arm_name>` convention produced by `.joint_cartesian`.
.joint_phi_grid_per_arm <- function(grids, arm_names) {
    out <- vector("list", length(arm_names))
    any_active <- FALSE
    for (k in seq_along(arm_names)) {
        col <- paste0("phi_", arm_names[k])
        if (!is.null(grids[[col]])) {
            out[[k]] <- as.numeric(grids[[col]])
            any_active <- TRUE
        }
    }
    if (!any_active) NULL else out
}

# Normalise the user-facing `phi_grid` argument into a list keyed by arm
# name, with NULL entries for arms without a phi axis. Accepts either a
# named list (subset of arm names) or a positional list of length n_arms.
# Single-element entries are treated as no-axis (the parse-time scalar phi
# already serves as that arm's dispersion).
.normalise_phi_grid <- function(phi_grid, arm_names) {
    if (is.null(phi_grid)) return(NULL)
    if (!is.list(phi_grid)) {
        stop("`phi_grid` must be a list (named by arm or positional).",
             call. = FALSE)
    }
    out <- vector("list", length(arm_names))
    names(out) <- arm_names
    if (!is.null(names(phi_grid))) {
        unknown <- setdiff(names(phi_grid), arm_names)
        if (length(unknown) > 0L) {
            stop("`phi_grid` names not in `responses`: ",
                 paste(shQuote(unknown), collapse = ", "), ".", call. = FALSE)
        }
        for (nm in names(phi_grid)) {
            v <- phi_grid[[nm]]
            if (!is.null(v) && length(v) > 1L) out[[nm]] <- as.numeric(v)
        }
    } else {
        if (length(phi_grid) != length(arm_names)) {
            stop("positional `phi_grid` must have length n_arms (",
                 length(arm_names), ").", call. = FALSE)
        }
        for (k in seq_along(phi_grid)) {
            v <- phi_grid[[k]]
            if (!is.null(v) && length(v) > 1L) out[[k]] <- as.numeric(v)
        }
    }
    out
}

# Validate one arm spec and fill in defaults.
.normalise_joint_arm <- function(a, k) {
    if (!is.list(a)) {
        stop("Arm ", k, ": expected a list of arm spec fields.", call. = FALSE)
    }
    must_have <- c("y", "X", "spatial_idx", "family")
    missing <- setdiff(must_have, names(a))
    if (length(missing)) {
        stop("Arm ", k, ": missing fields ", paste(shQuote(missing), collapse = ", "),
             ".", call. = FALSE)
    }
    N <- length(a$y)
    a$y <- as.numeric(a$y)
    if (!is.matrix(a$X)) {
        stop("Arm ", k, ": `X` must be a numeric matrix.", call. = FALSE)
    }
    if (nrow(a$X) != N) {
        stop("Arm ", k, ": nrow(X) (", nrow(a$X), ") must equal length(y) (",
             N, ").", call. = FALSE)
    }
    if (length(a$spatial_idx) != N) {
        stop("Arm ", k, ": length(spatial_idx) (", length(a$spatial_idx),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$spatial_idx <- as.integer(a$spatial_idx)
    a$n_trials <- if (is.null(a$n_trials)) rep(1L, N) else as.integer(a$n_trials)
    if (length(a$n_trials) != N) {
        stop("Arm ", k, ": length(n_trials) (", length(a$n_trials),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$re_idx <- if (is.null(a$re_idx)) rep(0, N) else as.numeric(a$re_idx)
    if (length(a$re_idx) != N) {
        stop("Arm ", k, ": length(re_idx) (", length(a$re_idx),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$n_re_groups <- as.integer(a$n_re_groups %||% 0L)
    a$sigma_re    <- as.numeric(a$sigma_re %||% 1.0)
    a$family      <- as.character(a$family)
    a$phi         <- as.numeric(a$phi %||% 1.0)
    a
}

# Decide which arm (if any) is the copy arm and what the sigma_pos grid is.
# Accepts either the new `copy$sigma_pos_grid` or a legacy `copy$alpha_grid`
# (translated to sigma_pos by multiplying by the median of the donor sigma
# grid; emits a clear deprecation message).
.resolve_copy <- function(copy, responses, prior, type) {
    if (is.null(copy)) {
        return(list(has_copy = FALSE, copy_arm_zero = -1L,
                    sigma_pos_grid = numeric(0)))
    }
    arm_id <- copy$arm
    if (is.null(arm_id)) {
        stop("`copy$arm` must be a name or 1-based index.", call. = FALSE)
    }
    if (is.character(arm_id)) {
        nm <- names(responses)
        if (is.null(nm) || !arm_id %in% nm) {
            stop("`copy$arm` = '", arm_id, "' not found in names(responses).",
                 call. = FALSE)
        }
        arm_zero <- match(arm_id, nm) - 1L
    } else {
        arm_zero <- as.integer(arm_id) - 1L
        if (arm_zero < 0L || arm_zero >= length(responses)) {
            stop("`copy$arm` index out of range.", call. = FALSE)
        }
    }
    if (!is.null(copy$sigma_pos_grid)) {
        if (!is.null(copy$alpha_grid)) {
            stop("`copy$alpha_grid` is deprecated and conflicts with ",
                 "`copy$sigma_pos_grid`. Pass only one (prefer ",
                 "`sigma_pos_grid`; see gcol33/tulpa#18).", call. = FALSE)
        }
        sigma_pos <- as.numeric(copy$sigma_pos_grid)
    } else if (!is.null(copy$alpha_grid)) {
        # Legacy alpha-grid path. Anchor sigma_pos = alpha * median(sigma_occ)
        # so a flat alpha grid maps to a flat sigma_pos grid centred on the
        # donor field amplitude — close to the prior behaviour while staying
        # in the new parameterization.
        sigma_donor <- prior$sigma_grid
        if (is.null(sigma_donor)) {
            sigma_donor <- exp(seq(log(0.1), log(3), length.out = 5))
        }
        anchor <- stats::median(as.numeric(sigma_donor))
        sigma_pos <- as.numeric(copy$alpha_grid) * anchor
        warning("`copy$alpha_grid` is deprecated. Translated to ",
                "`sigma_pos_grid = alpha_grid * median(prior$sigma_grid)` (= ",
                signif(anchor, 3), "). Pass `copy$sigma_pos_grid` directly ",
                "in new code (see gcol33/tulpa#18).", call. = FALSE)
    } else {
        # Default: mirror the donor-arm sigma grid.
        sigma_donor <- prior$sigma_grid
        if (is.null(sigma_donor)) {
            sigma_donor <- exp(seq(log(0.1), log(3), length.out = 5))
        }
        sigma_pos <- as.numeric(sigma_donor)
    }
    if (length(sigma_pos) == 0L) {
        stop("`copy$sigma_pos_grid` must have at least one positive value.",
             call. = FALSE)
    }
    if (any(sigma_pos < 0)) {
        stop("`copy$sigma_pos_grid` values must be non-negative.",
             call. = FALSE)
    }
    list(has_copy = TRUE, copy_arm_zero = arm_zero,
         sigma_pos_grid = sigma_pos)
}

# Recalibrate per-axis posterior moments after the joint pass. Slice
# cells from mode-tracked refinement on axis Y are pinned at modal
# (non-Y) values; including them in axis X's marginal (X != Y) collapses
# X to a point and shrinks Sd(X). Recompute mean/Sd for each column of
# `theta_grid` using only cells that vary that column — cartesian cells
# (`refining_axis == ""`) plus same-axis slice cells.
#
# Joint theta_mean / theta_sd come from `.nl_posterior_moments` and are
# left in place for axes with no foreign slice cells (the recompute is a
# no-op there). The original `theta_mean` / `theta_sd` are overwritten
# in place rather than augmented — downstream callers should read the
# axis marginal, not the cartesian-only joint moment.
.joint_recalibrate_axis_moments <- function(res) {
    refining <- res$refining_axis
    if (is.null(refining) || all(refining == "")) return(res)
    if (is.null(res$theta_grid) || !is.matrix(res$theta_grid)) return(res)
    cols  <- colnames(res$theta_grid)
    lm    <- res$log_marginal
    for (col in cols) {
        keep <- refining == "" | refining == col
        if (all(keep)) next
        lm_k <- lm[keep]
        m    <- max(lm_k)
        w    <- exp(lm_k - m)
        w    <- w / sum(w)
        vals <- res$theta_grid[keep, col]
        mu   <- sum(w * vals)
        sd_v <- sqrt(max(0, sum(w * vals^2) - mu^2))
        res$theta_mean[[col]] <- mu
        res$theta_sd[[col]]   <- sd_v
    }
    res
}

# Compute per-grid-point alpha = sigma_pos / sigma_occ and merge its
# posterior mean / sd into the result's theta_mean / theta_sd. Also append
# an `alpha` column to `theta_grid` so downstream code that inspects
# fit$theta_grid sees alpha alongside the sigma axes.
.joint_attach_alpha_moments <- function(res, has_copy) {
    if (!isTRUE(has_copy)) return(res)
    tg <- res$theta_grid
    if (is.null(tg) || !("sigma_occ" %in% colnames(tg)) ||
        !("sigma_pos" %in% colnames(tg))) {
        return(res)
    }
    so <- tg[, "sigma_occ"]
    sp <- tg[, "sigma_pos"]
    safe <- so > 0
    alpha_vec <- numeric(length(so))
    alpha_vec[safe]  <- sp[safe] / so[safe]
    alpha_vec[!safe] <- NA_real_
    # alpha varies on sigma_pos and sigma_occ; slice cells from refining
    # any other axis (e.g. phi_<arm>) fix sigma_pos and sigma_occ at
    # modal values and would collapse alpha to a point. Filter them out
    # before computing alpha posterior moments — same logic as the
    # per-axis recalibration step above.
    refining <- res$refining_axis %||% rep("", length(so))
    alpha_axes_ok <- refining == "" | refining == "sigma_pos" |
                      refining == "sigma_occ"
    # Recompute weights on the alpha-relevant subset directly from
    # log_marginal (rather than reusing res$weights, which is normalised
    # across all cells).
    use <- safe & alpha_axes_ok
    if (sum(use) == 0L) {
        alpha_mean <- NA_real_
        alpha_sd   <- NA_real_
    } else {
        lm_use <- res$log_marginal[use]
        m_use  <- max(lm_use)
        ws     <- exp(lm_use - m_use)
        ws     <- ws / sum(ws)
        a_use  <- alpha_vec[use]
        alpha_mean <- sum(ws * a_use)
        alpha_var  <- sum(ws * a_use^2) - alpha_mean^2
        alpha_sd   <- sqrt(max(0, alpha_var))
    }
    res$theta_grid <- cbind(tg, alpha = alpha_vec)
    res$theta_names <- colnames(res$theta_grid)
    res$theta_mean <- c(res$theta_mean, alpha = alpha_mean)
    res$theta_sd   <- c(res$theta_sd,   alpha = alpha_sd)
    res
}

# Compute per-arm latent offsets so callers can decode `modes` back into
# per-arm (beta, re) blocks plus the shared spatial block(s). For BYM2 the
# spatial block is two sub-blocks (phi, theta); for ICAR/CAR_proper it's
# just phi.
.joint_layout <- function(arms, n_spatial_units, n_spatial_blocks,
                          spatial_block_names) {
    n_arms <- length(arms)
    p_arm  <- vapply(arms, function(a) ncol(a$X),     integer(1))
    n_re   <- vapply(arms, function(a) a$n_re_groups, integer(1))

    beta_start <- integer(n_arms)
    cur <- 0L
    for (k in seq_len(n_arms)) {
        beta_start[k] <- cur
        cur <- cur + p_arm[k]
    }
    re_start <- integer(n_arms)
    for (k in seq_len(n_arms)) {
        re_start[k] <- cur
        cur <- cur + n_re[k]
    }
    spatial_starts <- integer(n_spatial_blocks)
    for (b in seq_len(n_spatial_blocks)) {
        spatial_starts[b] <- cur
        cur <- cur + n_spatial_units
    }
    n_x <- cur

    out <- list(
        n_arms     = n_arms,
        p          = p_arm,
        n_re       = n_re,
        beta_start = beta_start,
        re_start   = re_start,
        n_x        = n_x
    )
    for (b in seq_len(n_spatial_blocks)) {
        out[[spatial_block_names[b]]] <- spatial_starts[b]
    }
    out
}

# Back-compat alias: callers that pre-date the layout refactor still call
# `.joint_bym2_layout()`. Keep until the cover_hurdle wrapper migrates.
.joint_bym2_layout <- function(arms, n_spatial_units) {
    .joint_layout(arms, n_spatial_units, n_spatial_blocks = 2L,
                  spatial_block_names = c("phi_start", "theta_start"))
}


# --- adaptive grid refinement -----------------------------------------------
#
# Math ground.
#
# The joint marginal is approximated by
#   p(y) ~ sum_k w_k * exp(L_k),  L_k = log p(y, x_hat_k | theta_k),
# with normalised weights w_k. Posterior moments for any hyperparameter
# theta_j read
#   E[theta_j]   = sum_k w_k * theta_jk
#   Var[theta_j] = sum_k w_k * theta_jk^2 - E[theta_j]^2.
# When the user's grid for axis j is {t_1, ..., t_m} and the true posterior
# concentrates beyond the endpoints, the discretisation truncates the
# integrand. The resulting moments are biased toward the grid centre and
# their variance is bounded by the span of the grid points carrying weight.
#
# Refinement strategy: mode-tracked 1D slice (gcol33/tulpa#19).
#   1. Normalise log_marginal -> weights, project onto each refinable axis
#      by summing weight over the other axes (marginal weight per level).
#   2. For each refinable axis, check whether the lowest or highest level
#      carries marginal weight (or peak integrand density) above
#      `edge_thresh`. If so, propose:
#        a) one densification point at the (geometric, for log-spaced)
#           midpoint between the boundary and its inward neighbour,
#        b) two outward extension points past the boundary, mirroring the
#           spacing from the boundary to its neighbour.
#   3. Locate the boundary MAP cell: argmax of `log_marginal` over rows
#      whose refining-axis value equals the boundary level. Read off the
#      modal values of every *other* axis (sigma_occ, rho/rho_car/tau,
#      phi_<arm>, ...) from that single cell.
#   4. Evaluate the kernel at each new axis point paired with the modal
#      other-axis values — one Newton-Laplace solve per new point, *not*
#      one per `(new_point) x (other_cartesian)`. For an outer grid with
#      M other-axis cells this is an M-fold cost cut vs. the legacy
#      cartesian refinement (often M = 100-400 in production).
#   5. Calibrate the slice cells onto the *marginal* scale before merging:
#      add `log S_b = logsumexp_{k: axis == boundary} (L_k - L_b_modal)`
#      to each slice cell's log_marginal. Without this, slice cells sit
#      at the conditional-MAP density and are systematically under-weighted
#      by a factor of S_b in the joint softmax. With it, slice cells
#      contribute the integrated marginal mass the cartesian refinement
#      would have produced (under the assumption that the conditional
#      posterior in the other axes is locally stable across the refining
#      axis — true at the tail because the other axes are anchored by the
#      rest of the grid, not the refining axis).
#   6. Append the slice cells to the joint grid. Modes and per-grid Q come
#      from the real kernel evaluation, so downstream total-variance and
#      field-decoding code reads correct values at the new cells.
#
# Single helper handles every refinable axis (currently `sigma_pos`); the
# legacy cartesian path is gone — one code path, no fixed-vs-refined
# branching.

# Decide which axes are eligible for refinement.
#
# `sigma_pos` (copy arm field amplitude): at small effective sample size
#   the cover-arm likelihood barely identifies sigma_pos and the posterior
#   tail can extend past the user's grid endpoint. Boundary refinement
#   catches the truncation; interior refinement isn't relevant here.
#
# `phi_<arm>` (per-arm dispersion, e.g. beta concentration): the joint
#   engine integrates dispersion over an outer log-spaced grid; coarsening
#   that grid is the main lever on baseline wall time, but the posterior
#   peak can sit between grid levels and a discretisation bias creeps in.
#   Boundary + interior densification (mode-tracked) lets callers ship a
#   smaller default grid without regressing phi-recovery.
#
# Donor sigma_occ is the spatial prior amplitude and is treated as a
# deliberate prior choice; extending it requires explicit user opt-in
# (separate feature). Spatial mixing axes (rho, rho_car, tau) are
# similarly fixed for now — they have small grids and the integrand
# tends to span the user range.
.refinable_axes <- function(grids, cp_has_copy) {
    out <- character(0)
    if (cp_has_copy) {
        lev <- sort(unique(as.numeric(grids$sigma_pos)))
        if (length(lev) >= 2L) out <- c(out, "sigma_pos")
    }
    phi_cols <- grep("^phi_", names(grids), value = TRUE)
    for (col in phi_cols) {
        lev <- sort(unique(as.numeric(grids[[col]])))
        if (length(lev) >= 2L) out <- c(out, col)
    }
    out
}

# Per-axis edge diagnostics. Marginal weight gives the integrated posterior
# share on the boundary level — useful when the integrand has clearly
# decayed but the user's grid is just wide enough that the boundary
# *cell* still contributes. Integrand density at the boundary
# (`max_lm[boundary] - max_lm[overall]`, exponentiated) catches the
# complementary failure: when the boundary level is the heaviest *point-
# wise* mass but the user's grid is so narrow that very little is left
# inside it, so the boundary's *integrated* weight stays small while the
# truncated tail beyond the grid still carries posterior mass.
#
# We trigger refinement when EITHER criterion exceeds the threshold,
# which catches both failure modes with a single helper.
#
# Returns a named list keyed by axis with sorted unique `levels` and the
# scalar `min_score`, `max_score` to compare against `edge_thresh`.
.axis_edge_scores <- function(grids, log_marginal, axes) {
    if (length(log_marginal) == 0L) return(list())
    lm_max_total <- max(log_marginal)
    weights      <- .nl_normalise_weights(log_marginal)
    out <- list()
    for (a in axes) {
        v   <- as.numeric(grids[[a]])
        lev <- sort(unique(v))
        if (length(lev) < 2L) next
        # Integrated marginal weight at each boundary.
        w_lo <- sum(weights[v == lev[1L]])
        w_hi <- sum(weights[v == lev[length(lev)]])
        # Peak integrand density at each boundary (normalised by overall
        # peak): exp(max_lm@boundary - max_lm@overall).
        d_lo <- exp(max(log_marginal[v == lev[1L]])               - lm_max_total)
        d_hi <- exp(max(log_marginal[v == lev[length(lev)]])      - lm_max_total)
        out[[a]] <- list(
            levels    = lev,
            min_frac  = w_lo, max_frac = w_hi,
            min_dens  = d_lo, max_dens = d_hi,
            min_score = max(w_lo, d_lo),
            max_score = max(w_hi, d_hi)
        )
    }
    out
}

# Axis-aware spacing: sigma / sigma_pos / tau live on a log scale; rho and
# rho_car live on a linear scale. Per-arm phi axes (`phi_<arm>`) follow
# the GP-lengthscale convention — strictly positive, log-spaced.
.axis_is_log_scale <- function(axis_name) {
    axis_name %in% c("sigma", "sigma_occ", "sigma_pos", "tau", "sigma2",
                      "phi_gp", "lengthscale") ||
        startsWith(axis_name, "phi_")
}

# Natural domain clamp for bounded axes. NULL means unbounded (so just
# log-scale-positive for sigma/tau axes, which the log midpoint handles).
.axis_bounds <- function(axis_name) {
    if (axis_name == "sigma_pos") return(c(0, Inf))   # field amplitude >= 0
    if (axis_name == "sigma_occ") return(c(0, Inf))   # field amplitude >= 0
    if (axis_name == "rho")       return(c(0, 1))     # BYM2/AR1 mixing fraction
    if (axis_name == "rho_car")   return(c(-Inf, 1))  # proper-CAR (eigenvalue gated upstream)
    if (startsWith(axis_name, "phi_")) return(c(0, Inf))  # dispersion > 0
    NULL
}

# Propose new levels on one axis. Always adds one densification point
# between the boundary and its inward neighbour. Adds an outward extension
# only when the boundary is genuinely in the integrand's *tail*. When the
# boundary is the local mode, extension is suppressed because the data has
# not signalled that the true peak lies further out — densification alone
# safely tightens the integration in the heavy region without expanding
# the prior support beyond what the user requested.
.propose_axis_extension <- function(axis_name, lev, side,
                                     extend_ok = TRUE) {
    log_scale <- .axis_is_log_scale(axis_name)
    bounds    <- .axis_bounds(axis_name)
    mid <- if (log_scale) {
        function(a, b) exp(0.5 * (log(a) + log(b)))
    } else {
        function(a, b) 0.5 * (a + b)
    }
    if (side == "max") {
        edge      <- lev[length(lev)]
        neighbour <- lev[length(lev) - 1L]
        densify   <- mid(neighbour, edge)
        extend1   <- if (log_scale) edge * (edge / neighbour)
                     else            edge + (edge - neighbour)
        # Second extension point catches seeds where the integrand is
        # still appreciable one step past the boundary; without it we
        # under-cover when the true peak sits two steps beyond the user
        # grid.
        extend2   <- if (log_scale) extend1 * (edge / neighbour)
                     else            extend1 + (edge - neighbour)
    } else {
        edge      <- lev[1L]
        neighbour <- lev[2L]
        densify   <- mid(edge, neighbour)
        extend1   <- if (log_scale) edge * (edge / neighbour)
                     else            edge - (neighbour - edge)
        extend2   <- if (log_scale) extend1 * (edge / neighbour)
                     else            extend1 - (neighbour - edge)
    }
    pts <- if (isTRUE(extend_ok)) c(densify, extend1, extend2) else densify
    if (!is.null(bounds)) {
        pts <- pts[pts >= bounds[1L] & pts <= bounds[2L]]
    }
    # De-duplicate vs the existing levels at numerical tolerance.
    keep <- vapply(pts, function(p) {
        all(abs(lev - p) > 1e-8 * max(1, abs(p)))
    }, logical(1))
    pts[keep]
}

# Propose midpoint(s) on one axis to densify around an interior peak.
# `mode_idx` is the level index of the peak (1 < mode_idx < length(lev));
# `do_left` adds the midpoint between (mode_idx - 1, mode_idx),
# `do_right` adds the midpoint between (mode_idx, mode_idx + 1).
# Spacing follows the axis scale (geometric mean on log-spaced axes,
# arithmetic on linear axes). De-duplicated against the existing grid.
.propose_interior_densification <- function(axis_name, lev, mode_idx,
                                             do_left = FALSE, do_right = FALSE) {
    log_scale <- .axis_is_log_scale(axis_name)
    mid <- if (log_scale) {
        function(a, b) exp(0.5 * (log(a) + log(b)))
    } else {
        function(a, b) 0.5 * (a + b)
    }
    pts <- numeric(0)
    if (do_left  && mode_idx > 1L)              pts <- c(pts, mid(lev[mode_idx - 1L], lev[mode_idx]))
    if (do_right && mode_idx < length(lev))     pts <- c(pts, mid(lev[mode_idx], lev[mode_idx + 1L]))
    if (length(pts) == 0L) return(pts)
    bounds <- .axis_bounds(axis_name)
    if (!is.null(bounds)) {
        pts <- pts[pts >= bounds[1L] & pts <= bounds[2L]]
    }
    keep <- vapply(pts, function(p) {
        all(abs(lev - p) > 1e-8 * max(1, abs(p)))
    }, logical(1))
    pts[keep]
}

# Build mode-tracked slice triples for one (axis, anchor-level) pair plus
# the matching marginal-scale calibration. For each new axis point we
# produce *one* triple at (axis = p, others = modal_at_anchor), not a
# full cartesian — this is the M-fold cost cut.
#
# `anchor_lev` is the existing axis level whose modal cell we use as the
# anchor for the conditional posterior on the other axes:
#   * boundary refinement (peak at/near edge): anchor is the boundary
#     level (outer for max-side, inner for min-side).
#   * interior densification (peak between grid levels): anchor is the
#     peak level; new points sit on either side. The conditional posterior
#     is locally stable across one grid step in either direction.
#
# Returns NULL when no new points were given. Otherwise:
#   triples       : named list of equal-length vectors, kernel-input shaped
#                   (sigma_pos is present iff `cp_has_copy`).
#   calibration   : numeric, one entry per new point, added to that cell's
#                   raw log_marginal before it is merged into the joint
#                   grid. `log S_b` defined in the file-level math note.
#   warm_start_idx: index into the joint grid of the anchor MAP cell.
.new_mode_tracked_triples <- function(grids, log_marginal, axis_name,
                                       new_pts, anchor_lev, cp_has_copy) {
    if (length(new_pts) == 0L) return(NULL)
    v <- as.numeric(grids[[axis_name]])
    mask <- abs(v - anchor_lev) < 1e-12 * max(1, abs(anchor_lev))
    if (!any(mask)) return(NULL)
    anchor_lm   <- log_marginal[mask]
    k_map_local <- which.max(anchor_lm)
    idx_global  <- which(mask)[k_map_local]
    L_map       <- anchor_lm[k_map_local]
    # log S_a = log sum_{k @ anchor} exp(L_k - L_anchor_modal). >= 0.
    # Equals 0 when the conditional posterior collapses to a single cell,
    # log(M) at most (M = number of other-axis cells) when it is flat.
    calibration <- log(sum(exp(anchor_lm - L_map)))

    # Modal values of every *other* axis at the boundary MAP cell. Empty
    # axes (sigma_pos when has_copy = FALSE) stay empty.
    full_axes <- if (cp_has_copy) names(grids) else setdiff(names(grids), "sigma_pos")
    other_axes <- setdiff(full_axes, axis_name)
    triples <- vector("list", length(names(grids)))
    names(triples) <- names(grids)
    for (a in names(grids)) {
        if (a == axis_name) {
            triples[[a]] <- as.numeric(new_pts)
        } else if (a %in% other_axes && length(grids[[a]]) > 0L) {
            triples[[a]] <- rep(grids[[a]][idx_global], length(new_pts))
        } else {
            triples[[a]] <- numeric(0)
        }
    }
    list(triples       = triples,
         calibration   = rep(calibration, length(new_pts)),
         # Index into the original joint grid of the boundary MAP cell
         # whose modes are the best available warm-start for the slice
         # kernel call (first slice triple reuses it; later triples
         # chain prev_mode within the kernel).
         warm_start_idx = idx_global)
}

# Stitch the slice triples from multiple (axis, side) pairs into a single
# kernel-input shape, alongside one calibration vector. Drops any rows
# whose all-axis key already appears in `grids` (numerical tolerance via
# the same `%.10g` key format the legacy path used).
.concat_slice_triples <- function(triple_packs, grids, cp_has_copy) {
    if (length(triple_packs) == 0L) return(NULL)
    axes <- names(grids)
    combined <- vector("list", length(axes))
    names(combined) <- axes
    for (a in axes) {
        parts <- lapply(triple_packs, function(p) p$triples[[a]] %||% numeric(0))
        combined[[a]] <- as.numeric(unlist(parts, use.names = FALSE))
    }
    calib <- as.numeric(unlist(lapply(triple_packs, `[[`, "calibration"),
                                use.names = FALSE))
    n_new <- length(combined[[axes[1L]]])
    if (n_new == 0L) return(NULL)

    # De-duplicate against the existing grid on the active (non-empty)
    # axes. sigma_pos when has_copy = FALSE has length 0 and is ignored.
    active <- vapply(axes, function(a) length(grids[[a]]) > 0L, logical(1))
    active_axes <- axes[active]
    fmt <- function(lst) {
        if (length(lst[[active_axes[1L]]]) == 0L) return(character(0))
        cols <- lapply(active_axes, function(a) sprintf("%.10g", lst[[a]]))
        do.call(paste, c(cols, sep = ":"))
    }
    new_keys <- fmt(combined)
    old_keys <- fmt(grids)
    keep <- !new_keys %in% old_keys
    if (!any(keep)) return(NULL)
    for (a in axes) {
        if (length(combined[[a]]) > 0L) combined[[a]] <- combined[[a]][keep]
    }
    calib <- calib[keep]
    if (!cp_has_copy) combined$sigma_pos <- numeric(0)
    list(triples = combined, calibration = calib)
}

# Concatenate two kernel results row-wise, matching the run_nested_laplace_grid
# output layout (log_marginal, modes, n_iter, optional Q_csc_* lists).
# Also carries the per-cell `refining_axis` tag so per-axis moment
# recalibration downstream can drop slice cells that don't vary the axis
# whose marginal is being computed. Cells without an explicit tag (e.g.
# the initial cartesian pass) default to "" (= varies on every axis).
.concat_kernel_results <- function(a, b) {
    out <- a
    out$log_marginal <- c(a$log_marginal, b$log_marginal)
    out$n_iter       <- c(a$n_iter,       b$n_iter)
    out$n_grid       <- a$n_grid + b$n_grid
    if (!is.null(a$modes) && !is.null(b$modes)) {
        out$modes <- rbind(a$modes, b$modes)
    }
    tag_a <- a$refining_axis %||% rep("", length(a$log_marginal))
    tag_b <- b$refining_axis %||% rep("", length(b$log_marginal))
    out$refining_axis <- c(tag_a, tag_b)
    if (!is.null(a$Q_csc_p_per_grid) || !is.null(b$Q_csc_p_per_grid)) {
        out$Q_csc_p_per_grid <- c(a$Q_csc_p_per_grid %||% vector("list", a$n_grid),
                                  b$Q_csc_p_per_grid %||% vector("list", b$n_grid))
        out$Q_csc_i_per_grid <- c(a$Q_csc_i_per_grid %||% vector("list", a$n_grid),
                                  b$Q_csc_i_per_grid %||% vector("list", b$n_grid))
        out$Q_csc_x_per_grid <- c(a$Q_csc_x_per_grid %||% vector("list", a$n_grid),
                                  b$Q_csc_x_per_grid %||% vector("list", b$n_grid))
        out$Q_csc_n <- a$Q_csc_n %||% b$Q_csc_n
    }
    out
}

# Merge new axis triples into the kernel-input `grids` representation
# (paired vectors, sigma_pos empty when has_copy = FALSE).
.merge_grids <- function(grids, new_triples, cp_has_copy) {
    full_axes <- if (cp_has_copy) names(grids) else setdiff(names(grids), "sigma_pos")
    out <- grids
    for (a in full_axes) {
        out[[a]] <- c(grids[[a]], new_triples[[a]])
    }
    if (!cp_has_copy) out$sigma_pos <- numeric(0)
    out
}

# Detect refinement triggers on one axis and return the slice triple_pack
# (or NULL if nothing triggers). Pulled out so the outer pass can apply
# axes sequentially and let later axes see the updated grid/posterior.
.detect_axis_refinement <- function(grids, res, edge_info_a, axis_name,
                                     edge_thresh, cp_has_copy) {
    ei  <- edge_info_a
    lev <- ei$levels
    v   <- as.numeric(grids[[axis_name]])
    lm_max_at_lev <- vapply(lev, function(lv) {
        max(res$log_marginal[v == lv])
    }, numeric(1))
    mode_idx <- which.max(lm_max_at_lev)
    n_lev    <- length(lev)

    # Boundary triggers (peak-at-edge or tail-mass-at-edge).
    tr_min <- ei$min_score >= edge_thresh &&
              (mode_idx == 1L || ei$min_dens >= edge_thresh)
    tr_max <- ei$max_score >= edge_thresh &&
              (mode_idx == n_lev || ei$max_dens >= edge_thresh)

    # Interior densification trigger. When the posterior on the refining
    # axis is narrower than the grid spacing, weights collapse to a
    # single grid point and the posterior mean snaps to that grid value
    # — no continuous interpolation. Adjacent-level density rounds to
    # zero in that regime, so a density-based trigger cannot catch the
    # discretisation bias. Trigger on *grid coarseness* instead: when
    # the log-axis neighbour ratio exceeds `interior_log_step`, an
    # interior peak is suspected of sub-grid mismatch — densify between
    # peak and neighbour. Thresholds tuned to "always fire on a 7-point
    # log grid over [2, 300] (ratio ~2.4), never fire on a 25-point
    # grid (ratio ~1.25)".
    interior_log_step <- 0.55
    interior_lin_step <- 0.25
    wide_left <- wide_right <- FALSE
    if (mode_idx > 1L && mode_idx < n_lev) {
        if (.axis_is_log_scale(axis_name)) {
            wide_left  <- (log(lev[mode_idx])     - log(lev[mode_idx - 1L])) >= interior_log_step
            wide_right <- (log(lev[mode_idx + 1L]) - log(lev[mode_idx]))     >= interior_log_step
        } else {
            span <- diff(range(lev))
            if (span > 0) {
                wide_left  <- (lev[mode_idx]     - lev[mode_idx - 1L]) / span >= interior_lin_step
                wide_right <- (lev[mode_idx + 1L] - lev[mode_idx])     / span >= interior_lin_step
            }
        }
    }

    if (!tr_min && !tr_max && !wide_left && !wide_right) return(NULL)
    packs <- list()
    if (tr_max) {
        pts <- .propose_axis_extension(axis_name, lev, "max", extend_ok = TRUE)
        pk  <- .new_mode_tracked_triples(grids, res$log_marginal, axis_name,
                                          pts,
                                          anchor_lev   = lev[n_lev],
                                          cp_has_copy  = cp_has_copy)
        if (!is.null(pk)) packs[[length(packs) + 1L]] <- pk
    }
    if (tr_min) {
        pts <- .propose_axis_extension(axis_name, lev, "min", extend_ok = TRUE)
        pk  <- .new_mode_tracked_triples(grids, res$log_marginal, axis_name,
                                          pts,
                                          anchor_lev   = lev[1L],
                                          cp_has_copy  = cp_has_copy)
        if (!is.null(pk)) packs[[length(packs) + 1L]] <- pk
    }
    if (wide_left || wide_right) {
        pts <- .propose_interior_densification(axis_name, lev, mode_idx,
                                                wide_left, wide_right)
        pk  <- .new_mode_tracked_triples(grids, res$log_marginal, axis_name,
                                          pts,
                                          anchor_lev   = lev[mode_idx],
                                          cp_has_copy  = cp_has_copy)
        if (!is.null(pk)) packs[[length(packs) + 1L]] <- pk
    }
    if (length(packs) == 0L) return(NULL)
    packs
}

# Run the slice kernel for one axis's triple_packs and merge the result
# into (res, grids). Warm-start from the first pack's anchor MAP cell.
# Tags each new cell with `axis_name` so per-axis moment recalibration
# downstream can exclude cells that fix this axis at a non-varying value
# from OTHER axes' marginals.
.apply_axis_refinement <- function(grids, res, triple_packs, axis_name,
                                    backend, arms, prior, cp,
                                    max_iter, tol, n_threads, x_init,
                                    store_Q, arm_names) {
    merged <- .concat_slice_triples(triple_packs, grids, cp$has_copy)
    if (is.null(merged)) return(list(grids = grids, res = res, n_new = 0L))
    new_triples <- merged$triples
    calibration <- merged$calibration

    # Warm-start the slice kernel call from the first pack's anchor MAP
    # mode. The kernel chains prev_mode forward within a call, so this
    # skips a cold Newton on triple 1; subsequent triples benefit
    # transitively. Slice cells live close in theta-space to the
    # anchor, so the converged anchor mode is the best initial guess.
    slice_x_init <- x_init
    if (!is.null(res$modes) && is.matrix(res$modes)) {
        idx0 <- triple_packs[[1L]]$warm_start_idx
        if (length(idx0) == 1L && idx0 >= 1L && idx0 <= nrow(res$modes)) {
            slice_x_init <- as.numeric(res$modes[idx0, ])
        }
    }

    res_extra <- backend$call_kernel(arms, prior, cp, new_triples,
                                      max_iter, tol, n_threads,
                                      slice_x_init, isTRUE(store_Q),
                                      arm_names = arm_names)
    # Lift slice cells onto the marginal scale before merging. Same
    # softmax invariant the cartesian path used to give us for free.
    res_extra$log_marginal  <- res_extra$log_marginal + calibration
    res_extra$refining_axis <- rep(axis_name, length(res_extra$log_marginal))

    res   <- .concat_kernel_results(res, res_extra)
    grids <- .merge_grids(grids, new_triples, cp$has_copy)
    n_new <- length(new_triples[[names(grids)[which(vapply(grids, length,
                                                            integer(1)) > 0L)[1L]]]])
    list(grids = grids, res = res, n_new = n_new)
}

# Priority order for refining axes. `sigma_pos` (boundary truncation —
# distorts every derived quantity including alpha) is refined first; any
# `phi_<arm>` axes (interior densification — sharpens marginal recovery
# without shifting joint structure) are refined second so they read the
# post-`sigma_pos`-extension modal. Other axes fall in declaration order.
.axis_refinement_order <- function(axes) {
    priority <- function(a) {
        if (a == "sigma_pos") 1L
        else if (startsWith(a, "phi_")) 2L
        else 3L
    }
    axes[order(vapply(axes, priority, integer(1)), seq_along(axes))]
}

# Main entry: run up to `max_passes` of refinement. Each pass processes
# refinable axes *sequentially* (one kernel call per axis), so each axis
# sees the grid/posterior updates from any earlier axis in the same pass.
# Re-uses the backend kernel — one code path, no fixed-grid fallback.
# Stops early when no axis triggers.
.adaptive_refine_pass <- function(grids, res, backend, arms, prior, cp,
                                  max_iter, tol, n_threads, x_init, store_Q,
                                  edge_thresh, max_passes,
                                  arm_names = NULL) {
    info <- list(triggered_axes = character(0),
                 n_points_added = integer(0))
    if (max_passes < 1L) {
        return(list(grids = grids, res = res, info = NULL))
    }
    for (pass in seq_len(max_passes)) {
        axes <- .axis_refinement_order(.refinable_axes(grids, cp$has_copy))
        if (length(axes) == 0L) break
        any_triggered <- FALSE
        triggered_this_pass <- character(0)
        for (a in axes) {
            edge_info <- .axis_edge_scores(grids, res$log_marginal, a)
            ei <- edge_info[[a]]
            if (is.null(ei)) next
            packs <- .detect_axis_refinement(grids, res, ei, a,
                                              edge_thresh, cp$has_copy)
            if (is.null(packs)) next
            step <- .apply_axis_refinement(grids, res, packs, a, backend,
                                            arms, prior, cp, max_iter, tol,
                                            n_threads, x_init, store_Q,
                                            arm_names)
            if (step$n_new == 0L) next
            grids <- step$grids
            res   <- step$res
            triggered_this_pass <- c(triggered_this_pass, a)
            info$n_points_added <- c(info$n_points_added, step$n_new)
            any_triggered <- TRUE
        }
        if (!any_triggered) break
        info$triggered_axes <- c(info$triggered_axes,
                                  paste(triggered_this_pass, collapse = ","))
    }
    if (length(info$triggered_axes) == 0L) info <- NULL
    list(grids = grids, res = res, info = info)
}
