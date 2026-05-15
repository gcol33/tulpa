#' Joint multi-likelihood nested Laplace approximation
#'
#' @description
#' Outer-grid nested Laplace driver for *joint* models — multiple response
#' arms sharing one latent prior block, with optional INLA-style copy-scaling
#' (`f(..., copy=, fixed=FALSE)`) on a designated arm.
#'
#' Supported priors (Phase 1c):
#'  * `"bym2"`       — outer grid over `(sigma_spatial, rho [, alpha])`.
#'                     Latent: `phi (n_s) | theta (n_s)`.
#'  * `"icar"`       — outer grid over `(tau_spatial [, alpha])`.
#'                     Latent: `phi (n_s)`.
#'  * `"car_proper"` — outer grid over `(tau, rho_car [, alpha])`. Latent:
#'                     `phi (n_s)` with `Q = tau * (D - rho_car * W)`.
#'
#' Other backends (NNGP, HSGP, RW1/2, AR1) follow the same interface and
#' land under Phase 3.
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
#'     (default 5 log-spaced values), `rho_grid`
#'     (default `c(0.2, 0.5, 0.8, 0.95)`).
#'   * **icar**: `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`,
#'     `n_neighbors`; optional `tau_grid` (default 5 log-spaced values
#'     spanning roughly sigma in `[0.1, 3]`).
#'   * **car_proper**: same as icar plus `rho_car_grid`
#'     (default `c(0.5, 0.8, 0.95, 0.99)`).
#'
#' @param copy Optional list controlling the copy-scaling (alpha) arm:
#'   * `arm`         — name (or 1-based index) of the arm to copy-scale.
#'   * `alpha_grid`  — numeric grid for alpha; default
#'                     `c(0, 0.5, 1, 1.5, 2)`.
#'   When `NULL` (default), no copy-scaling is applied.
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
#'   grid edge — see gcol33/tulpaObs#7. Pass `FALSE` to recover the legacy
#'   fixed-grid behaviour.
#' @param adaptive_grid_edge_thresh Numeric (default `0.02`). Refinement
#'   triggers when the per-axis edge score on the boundary of the
#'   refinable axis (currently `alpha` only) exceeds this value. The
#'   score is
#'   `max(marginal_weight_at_boundary,
#'        exp(max_log_marginal_at_boundary - max_log_marginal_overall))`,
#'   i.e. the larger of the integrated weight on the boundary level and
#'   the relative integrand density at the boundary level. Catches both
#'   boundary pile-up (truth lies *at* a grid endpoint, weight is heavy
#'   there) and integrand truncation (integrand still has appreciable
#'   density at the boundary, but the cell width is so narrow it gets
#'   little integrated weight). The default `0.02` corresponds to ~4 log
#'   units of decay before refinement stops, which catches the D3
#'   alpha-at-boundary failure on a typical small-sample joint hurdle
#'   while leaving cleanly-decayed boundaries alone. Lower the threshold
#'   to refine more aggressively at the cost of extra kernel passes.
#' @param adaptive_grid_max_passes Integer (default `1L`). Maximum number
#'   of refinement passes. One pass typically suffices; two is rarely
#'   useful and inflates runtime.
#'
#' @return A list of class `c("tulpa_nested_laplace_joint",
#'   "tulpa_nested_laplace", "list")` with:
#'   * `theta_grid`, `theta_names` — outer-grid hyperparameter values.
#'   * `log_marginal`, `weights` — per-grid-point log-marginal and integration
#'      weights (sum to 1).
#'   * `theta_mean`, `theta_sd` — posterior moments per hyperparameter.
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

    cp <- .resolve_copy(copy, responses)
    grids <- backend$build_grids(prior, cp$has_copy, cp$alpha_grid)

    res <- backend$call_kernel(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, isTRUE(store_Q))

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
            max_passes  = adaptive_grid_max_passes
        )
        grids       <- refined$grids
        res         <- refined$res
        refine_info <- refined$info
    }

    res$theta_grid  <- backend$theta_grid(grids, cp$has_copy)
    res$theta_names <- colnames(res$theta_grid)
    res$weights     <- .nl_normalise_weights(res$log_marginal)
    res             <- .nl_posterior_moments(res, paste0("joint_", type))
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
        build_grids = function(prior, has_copy, alpha_axis) {
            sigma_axis <- prior$sigma_grid %||%
                exp(seq(log(0.1), log(3), length.out = 5))
            rho_axis <- prior$rho_grid %||% c(0.2, 0.5, 0.8, 0.95)
            .joint_cartesian(list(sigma = sigma_axis, rho = rho_axis),
                              has_copy, alpha_axis)
        },
        call_kernel = function(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, store_Q = FALSE) {
            cpp_nested_laplace_joint_bym2(
                arms_list       = arms,
                copy_arm        = as.integer(cp$copy_arm_zero),
                n_spatial_units = as.integer(prior$n_spatial_units),
                adj_row_ptr     = as.integer(prior$adj_row_ptr),
                adj_col_idx     = as.integer(prior$adj_col_idx),
                n_neighbors     = as.integer(prior$n_neighbors),
                scale_factor    = as.numeric(prior$scale_factor %||% 1.0),
                sigma_spatial_grid = as.numeric(grids$sigma),
                rho_grid           = as.numeric(grids$rho),
                alpha_grid         = as.numeric(grids$alpha),
                max_iter   = as.integer(max_iter),
                tol        = as.numeric(tol),
                n_threads  = as.integer(n_threads),
                x_init_nullable = x_init,
                store_Q    = isTRUE(store_Q)
            )
        },
        theta_grid = function(grids, has_copy) {
            if (has_copy) {
                cbind(sigma = grids$sigma, rho = grids$rho, alpha = grids$alpha)
            } else {
                cbind(sigma = grids$sigma, rho = grids$rho)
            }
        },
        layout = function(arms, prior) {
            .joint_layout(arms, prior$n_spatial_units, n_spatial_blocks = 2L,
                          spatial_block_names = c("phi_start", "theta_start"))
        }
    ),

    icar = list(
        build_grids = function(prior, has_copy, alpha_axis) {
            tau_axis <- prior$tau_grid %||%
                exp(seq(log(1 / 9), log(100), length.out = 5))  # sigma in [0.1, 3]
            .joint_cartesian(list(tau = tau_axis), has_copy, alpha_axis)
        },
        call_kernel = function(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, store_Q = FALSE) {
            cpp_nested_laplace_joint_icar(
                arms_list       = arms,
                copy_arm        = as.integer(cp$copy_arm_zero),
                n_spatial_units = as.integer(prior$n_spatial_units),
                adj_row_ptr     = as.integer(prior$adj_row_ptr),
                adj_col_idx     = as.integer(prior$adj_col_idx),
                n_neighbors     = as.integer(prior$n_neighbors),
                tau_grid        = as.numeric(grids$tau),
                alpha_grid      = as.numeric(grids$alpha),
                max_iter   = as.integer(max_iter),
                tol        = as.numeric(tol),
                n_threads  = as.integer(n_threads),
                x_init_nullable = x_init,
                store_Q    = isTRUE(store_Q)
            )
        },
        theta_grid = function(grids, has_copy) {
            if (has_copy) cbind(tau = grids$tau, alpha = grids$alpha)
            else          cbind(tau = grids$tau)
        },
        layout = function(arms, prior) {
            .joint_layout(arms, prior$n_spatial_units, n_spatial_blocks = 1L,
                          spatial_block_names = "phi_start")
        }
    ),

    car_proper = list(
        build_grids = function(prior, has_copy, alpha_axis) {
            tau_axis     <- prior$tau_grid %||%
                exp(seq(log(1 / 9), log(100), length.out = 5))
            rho_car_axis <- prior$rho_car_grid %||% c(0.5, 0.8, 0.95, 0.99)
            .joint_cartesian(list(tau = tau_axis, rho_car = rho_car_axis),
                              has_copy, alpha_axis)
        },
        call_kernel = function(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, store_Q = FALSE) {
            cpp_nested_laplace_joint_car_proper(
                arms_list       = arms,
                copy_arm        = as.integer(cp$copy_arm_zero),
                n_spatial_units = as.integer(prior$n_spatial_units),
                adj_row_ptr     = as.integer(prior$adj_row_ptr),
                adj_col_idx     = as.integer(prior$adj_col_idx),
                n_neighbors     = as.integer(prior$n_neighbors),
                tau_grid        = as.numeric(grids$tau),
                rho_car_grid    = as.numeric(grids$rho_car),
                alpha_grid      = as.numeric(grids$alpha),
                max_iter   = as.integer(max_iter),
                tol        = as.numeric(tol),
                n_threads  = as.integer(n_threads),
                x_init_nullable = x_init,
                store_Q    = isTRUE(store_Q)
            )
        },
        theta_grid = function(grids, has_copy) {
            if (has_copy) {
                cbind(tau = grids$tau, rho_car = grids$rho_car,
                      alpha = grids$alpha)
            } else {
                cbind(tau = grids$tau, rho_car = grids$rho_car)
            }
        },
        layout = function(arms, prior) {
            .joint_layout(arms, prior$n_spatial_units, n_spatial_blocks = 1L,
                          spatial_block_names = "phi_start")
        }
    )
)


# --- helpers -----------------------------------------------------------------

# Cartesian product over a named list of axes plus an optional alpha axis.
# Returns a named list of paired vectors of identical length, ready to feed
# the C++ kernel.
.joint_cartesian <- function(axes, has_copy, alpha_axis) {
    full <- if (has_copy) c(axes, list(alpha = alpha_axis)) else axes
    gr <- do.call(expand.grid,
                  c(full, list(KEEP.OUT.ATTRS = FALSE,
                                stringsAsFactors = FALSE)))
    out <- as.list(gr)
    if (!has_copy) out$alpha <- numeric(0)
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

# Decide which arm (if any) is copy-scaled and what the alpha grid is.
.resolve_copy <- function(copy, responses) {
    if (is.null(copy)) {
        return(list(has_copy = FALSE, copy_arm_zero = -1L,
                    alpha_grid = numeric(0)))
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
    alpha <- copy$alpha_grid %||% c(0.0, 0.5, 1.0, 1.5, 2.0)
    list(has_copy = TRUE, copy_arm_zero = arm_zero, alpha_grid = alpha)
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
# their variance is bounded by the span of the grid points carrying weight,
# producing the under-coverage seen in INLAabun D3 at alpha_true = 1.5.
#
# Refinement strategy (matches the user's chosen Strategy B, two-pass).
#   1. Normalise log_marginal -> weights, project onto each axis by summing
#      weight over the other axes (marginal weight per axis level).
#   2. For each axis, check whether the lowest or highest level carries
#      marginal weight above `edge_thresh`. If so, add:
#        a) one densification point at the midpoint between the boundary
#           level and its neighbour (geometric mid on log-spaced axes such
#           as sigma/tau, arithmetic mid on linear axes such as alpha/rho),
#        b) one extension point beyond the boundary, mirroring the spacing
#           from the boundary to its neighbour.
#      Bounded axes (rho, rho_car, alpha) are clipped to their natural
#      domain; alpha is forbidden to extend below zero (the no-coupling
#      anchor that callers explicitly include).
#   3. Cartesian-product the new axis levels with the other axes' original
#      levels, drop combinations already present in `grids`, and feed the
#      new triples through the same `backend$call_kernel`. Concatenate
#      log_marginal, modes, n_iter, and the optional per-grid Q lists.
#
# Single helper covers alpha and the spatial axes via the same axis-by-axis
# logic — no per-axis duplication.

# Decide which axes are eligible for refinement. Currently restricted to
# the copy-scaling alpha axis, where the D3 failure mode lives: a fixed
# alpha_grid that stops at the true alpha leaves the right tail
# truncated, biasing posterior moments. Spatial-prior axes (sigma, rho,
# tau, rho_car) are intentionally NOT refined here: their grids are
# user-set and reflect deliberate prior choices, and extending them
# pulls posterior latent variance into extreme-hyperparameter regions
# that the user did not request. Spatial-axis refinement is a separate
# feature scheduled when callers explicitly opt in.
.refinable_axes <- function(grids, cp_has_copy) {
    if (!cp_has_copy) return(character(0))
    lev <- sort(unique(as.numeric(grids$alpha)))
    if (length(lev) < 2L) return(character(0))
    "alpha"
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

# Axis-aware spacing: alpha and rho live on a linear scale, sigma / tau on
# a log scale. Returns a function `mid(a, b)` for that axis.
.axis_is_log_scale <- function(axis_name) {
    axis_name %in% c("sigma", "tau", "sigma2", "phi_gp", "lengthscale")
}

# Natural domain clamp for bounded axes. NULL means unbounded (so just
# log-scale-positive for sigma/tau, which the log midpoint handles).
.axis_bounds <- function(axis_name) {
    if (axis_name == "alpha")   return(c(0,  Inf))   # alpha >= 0 by convention
    if (axis_name == "rho")     return(c(0,  1))     # BYM2/AR1 mixing fraction
    if (axis_name == "rho_car") return(c(-Inf, 1))   # proper-CAR (eigenvalue gated upstream)
    NULL
}

# Propose new levels on one axis. Always adds one densification point
# between the boundary and its inward neighbour. Adds an outward extension
# only when the boundary is genuinely in the integrand's *tail* (i.e.
# the integrand has declined from the inward neighbour to the boundary
# by at least `decay_factor`). When the boundary is the local mode,
# extension is suppressed because the data has not signalled that the
# true peak lies further out — densification alone safely tightens the
# integration in the heavy region without expanding the prior support
# beyond what the user requested.
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
        # grid (the D3 alpha_true = 1.5 against alpha_grid stopping at
        # 1.5 case picks up ~10 extra percentage points from the second
        # extension).
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
# (sigma_k, rho_k, alpha_k) triples are dropped via a string-key set.
.new_cartesian_triples <- function(grids, new_per_axis, cp_has_copy) {
    full_axes <- if (cp_has_copy) names(grids) else setdiff(names(grids), "alpha")
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
    # original alpha-shape convention (empty when has_copy = FALSE).
    out <- as.list(new_rows)
    if (!cp_has_copy) out$alpha <- numeric(0)
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
# (paired vectors, alpha empty when has_copy = FALSE).
.merge_grids <- function(grids, new_triples, cp_has_copy) {
    full_axes <- if (cp_has_copy) names(grids) else setdiff(names(grids), "alpha")
    out <- grids
    for (a in full_axes) {
        out[[a]] <- c(grids[[a]], new_triples[[a]])
    }
    if (!cp_has_copy) out$alpha <- numeric(0)
    out
}

# Main entry: run up to `max_passes` of refinement. Each pass identifies the
# single axis with the largest exceedance over `edge_thresh` (so we add the
# minimal number of triples) and inflates the grid along that axis only.
# Re-using the same backend kernel keeps this a single code path — no
# fixed-grid fallback. Stops early when no axis crosses the threshold.
.adaptive_refine_pass <- function(grids, res, backend, arms, prior, cp,
                                  max_iter, tol, n_threads, x_init, store_Q,
                                  edge_thresh, max_passes) {
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
        #      truncating the tail. This is the D3 failure
        #      (alpha_true = 1.5 against the alpha_grid that stops at 1.5).
        #   2. Boundary is on the descent (a neighbour is heavier) but
        #      the integrand density at the boundary still exceeds
        #      `edge_thresh`. The tail beyond the boundary carries
        #      enough mass that the truncation matters; we extend to
        #      soak it up.
        # When the boundary is on the descent AND density at boundary is
        # below threshold, the integrand has already decayed by the time
        # we hit the edge — no refinement needed. This protects the
        # well-behaved interior-peak case (e.g. alpha = 0 truth, peak at
        # alpha = 0, boundary at alpha = 0 is the mode but with low
        # density on the high side, no extension on either side).
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
                                          x_init, isTRUE(store_Q))
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
