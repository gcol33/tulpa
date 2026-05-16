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
#' @param adaptive_grid Logical (default `TRUE`). When `TRUE`, a second
#'   kernel pass is triggered on any hyperparameter axis whose marginal
#'   posterior weight on the boundary point(s) exceeds
#'   `adaptive_grid_edge_thresh`. New points are appended on that axis
#'   (one densification between the boundary and its neighbour, one
#'   extension beyond the boundary on a log-spaced axis) and the cartesian
#'   product with the other axes is evaluated and concatenated. Fixes
#'   posterior CI under-coverage when truth sits near or at the user's
#'   grid edge. Pass `FALSE` to recover the legacy fixed-grid behaviour.
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
                                       adaptive_grid = TRUE,
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
    w <- res$weights
    so <- tg[, "sigma_occ"]
    sp <- tg[, "sigma_pos"]
    safe <- so > 0
    alpha_vec <- numeric(length(so))
    alpha_vec[safe]  <- sp[safe] / so[safe]
    alpha_vec[!safe] <- NA_real_
    # Posterior moments over the valid grid points only — division by 0 at
    # sigma_occ == 0 is a degenerate boundary that callers explicitly include.
    w_safe <- w[safe]
    a_safe <- alpha_vec[safe]
    if (length(a_safe) == 0L || sum(w_safe) <= 0) {
        alpha_mean <- NA_real_
        alpha_sd   <- NA_real_
    } else {
        ws <- w_safe / sum(w_safe)
        alpha_mean <- sum(ws * a_safe)
        alpha_var  <- sum(ws * a_safe^2) - alpha_mean^2
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
# Refinement strategy (two-pass).
#   1. Normalise log_marginal -> weights, project onto each axis by summing
#      weight over the other axes (marginal weight per axis level).
#   2. For each refinable axis, check whether the lowest or highest level
#      carries marginal weight above `edge_thresh`. If so, add:
#        a) one densification point at the geometric midpoint between the
#           boundary level and its neighbour (sigma_pos is log-spaced),
#        b) one extension point beyond the boundary, mirroring the spacing
#           from the boundary to its neighbour.
#      sigma_pos extends down only to ~0 (no-coupling anchor when callers
#      explicitly include it).
#   3. Cartesian-product the new axis levels with the other axes' original
#      levels, drop combinations already present in `grids`, and feed the
#      new triples through the same `backend$call_kernel`. Concatenate
#      log_marginal, modes, n_iter, and the optional per-grid Q lists.
#
# Single helper covers sigma_pos and the spatial axes via the same
# axis-by-axis logic — no per-axis duplication.

# Decide which axes are eligible for refinement. Currently restricted to
# the copy-arm sigma_pos axis, which carries the small-n_pos failure mode:
# at small effective sample size the cover-arm likelihood barely
# identifies sigma_pos and the posterior tail can extend past the user's
# grid endpoint. Donor sigma_occ is the spatial prior amplitude and is
# treated as a deliberate prior choice; extending it requires explicit
# user opt-in (separate feature).
.refinable_axes <- function(grids, cp_has_copy) {
    if (!cp_has_copy) return(character(0))
    lev <- sort(unique(as.numeric(grids$sigma_pos)))
    if (length(lev) < 2L) return(character(0))
    "sigma_pos"
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
# rho_car live on a linear scale. Returns the scale kind for axis name.
.axis_is_log_scale <- function(axis_name) {
    axis_name %in% c("sigma", "sigma_occ", "sigma_pos", "tau", "sigma2",
                      "phi_gp", "lengthscale")
}

# Natural domain clamp for bounded axes. NULL means unbounded (so just
# log-scale-positive for sigma/tau axes, which the log midpoint handles).
.axis_bounds <- function(axis_name) {
    if (axis_name == "sigma_pos") return(c(0, Inf))   # field amplitude >= 0
    if (axis_name == "sigma_occ") return(c(0, Inf))   # field amplitude >= 0
    if (axis_name == "rho")       return(c(0, 1))     # BYM2/AR1 mixing fraction
    if (axis_name == "rho_car")   return(c(-Inf, 1))  # proper-CAR (eigenvalue gated upstream)
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

# Build the new cartesian-product triples that need a kernel call. Existing
# combinations are dropped via a string-key set.
.new_cartesian_triples <- function(grids, new_per_axis, cp_has_copy) {
    full_axes <- if (cp_has_copy) names(grids) else setdiff(names(grids), "sigma_pos")
    axis_levels <- list()
    for (a in full_axes) {
        old <- sort(unique(as.numeric(grids[[a]])))
        new <- new_per_axis[[a]] %||% numeric(0)
        axis_levels[[a]] <- sort(unique(c(old, new)))
    }
    cart <- do.call(expand.grid,
                    c(axis_levels, list(KEEP.OUT.ATTRS = FALSE,
                                         stringsAsFactors = FALSE)))
    # Existing grid keys
    fmt <- function(df, axes) {
        apply(as.matrix(df[, axes, drop = FALSE]), 1L,
              function(r) paste(sprintf("%.10g", r), collapse = ":"))
    }
    cart_keys <- fmt(cart, full_axes)
    old_df    <- as.data.frame(grids[full_axes])
    old_keys  <- fmt(old_df, full_axes)
    new_rows  <- cart[!cart_keys %in% old_keys, , drop = FALSE]
    if (nrow(new_rows) == 0L) return(NULL)
    # Re-shape to the named-list format the kernel expects, preserving the
    # original sigma_pos-shape convention (empty when has_copy = FALSE).
    out <- as.list(new_rows)
    if (!cp_has_copy) out$sigma_pos <- numeric(0)
    out
}

# Concatenate two kernel results row-wise, matching the run_nested_laplace_grid
# output layout (log_marginal, modes, n_iter, optional Q_csc_* lists).
.concat_kernel_results <- function(a, b) {
    out <- a
    out$log_marginal <- c(a$log_marginal, b$log_marginal)
    out$n_iter       <- c(a$n_iter,       b$n_iter)
    out$n_grid       <- a$n_grid + b$n_grid
    if (!is.null(a$modes) && !is.null(b$modes)) {
        out$modes <- rbind(a$modes, b$modes)
    }
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

# Main entry: run up to `max_passes` of refinement. Each pass identifies the
# single axis with the largest exceedance over `edge_thresh` (so we add the
# minimal number of triples) and inflates the grid along that axis only.
# Re-using the same backend kernel keeps this a single code path — no
# fixed-grid fallback. Stops early when no axis crosses the threshold.
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
        axes    <- .refinable_axes(grids, cp$has_copy)
        if (length(axes) == 0L) break
        edge_info <- .axis_edge_scores(grids, res$log_marginal, axes)

        # Refinement targets two failure modes:
        #   1. Boundary IS the local mode for that axis. The integrand
        #      peak sits at the user-supplied grid edge; we have no
        #      evidence whether the true peak is at the boundary or
        #      further out. Extending outward is the only way to find
        #      out — without it we underestimate posterior variance by
        #      truncating the tail.
        #   2. Boundary is on the descent (a neighbour is heavier) but
        #      the integrand density at the boundary still exceeds
        #      `edge_thresh`. The tail beyond the boundary carries
        #      enough mass that the truncation matters; we extend to
        #      soak it up.
        # When the boundary is on the descent AND density at boundary is
        # below threshold, the integrand has already decayed by the time
        # we hit the edge — no refinement needed.
        new_per_axis <- list()
        triggered_this_pass <- character(0)
        for (a in names(edge_info)) {
            ei <- edge_info[[a]]
            lev <- ei$levels
            # Marginal max-log-marginal per level on this axis (max over
            # other axes' grid points). The "local mode" for the axis
            # lies at the level whose marginal max-log-marginal is the
            # largest. The boundary either IS that mode, or sits on the
            # descent from it.
            v <- as.numeric(grids[[a]])
            lm_max_at_lev <- vapply(lev, function(lv) {
                max(res$log_marginal[v == lv])
            }, numeric(1))
            mode_idx <- which.max(lm_max_at_lev)
            n_lev    <- length(lev)
            tr_min <- ei$min_score >= edge_thresh &&
                      (mode_idx == 1L || ei$min_dens >= edge_thresh)
            tr_max <- ei$max_score >= edge_thresh &&
                      (mode_idx == n_lev || ei$max_dens >= edge_thresh)
            if (!tr_min && !tr_max) next
            pts <- numeric(0)
            if (tr_max) {
                pts <- c(pts, .propose_axis_extension(a, lev, "max",
                                                      extend_ok = TRUE))
            }
            if (tr_min) {
                pts <- c(pts, .propose_axis_extension(a, lev, "min",
                                                      extend_ok = TRUE))
            }
            pts <- sort(unique(pts))
            if (length(pts) == 0L) next
            new_per_axis[[a]] <- pts
            triggered_this_pass <- c(triggered_this_pass, a)
        }
        if (length(new_per_axis) == 0L) break

        new_triples <- .new_cartesian_triples(grids, new_per_axis,
                                              cp$has_copy)
        if (is.null(new_triples)) break

        res_extra <- backend$call_kernel(arms, prior, cp, new_triples,
                                          max_iter, tol, n_threads,
                                          x_init, isTRUE(store_Q),
                                          arm_names = arm_names)
        res   <- .concat_kernel_results(res, res_extra)
        grids <- .merge_grids(grids, new_triples, cp$has_copy)

        # All paired vectors in `new_triples` have the same length (= number
        # of new grid points). Read it off the first axis.
        n_new <- length(new_triples[[names(grids)[1L]]])
        info$triggered_axes <- c(info$triggered_axes,
                                  paste(triggered_this_pass, collapse = ","))
        info$n_points_added <- c(info$n_points_added, n_new)
    }
    if (length(info$triggered_axes) == 0L) info <- NULL
    list(grids = grids, res = res, info = info)
}
