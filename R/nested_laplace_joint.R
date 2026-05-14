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
#'
#' @seealso [tulpa_nested_laplace()] for the single-arm engine.
#' @export
tulpa_nested_laplace_joint <- function(responses,
                                       prior,
                                       copy = NULL,
                                       max_iter = 50L, tol = 1e-6,
                                       n_threads = 1L,
                                       x_init = NULL, verbose = FALSE) {
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
                                n_threads, x_init)

    res$theta_grid  <- backend$theta_grid(grids, cp$has_copy)
    res$theta_names <- colnames(res$theta_grid)
    res$weights     <- .nl_normalise_weights(res$log_marginal)
    res             <- .nl_posterior_moments(res, paste0("joint_", type))
    res$arm_layout  <- backend$layout(arms, prior)
    res$prior       <- prior
    res$responses   <- responses
    res$copy        <- copy
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
                                n_threads, x_init) {
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
                store_Q    = FALSE
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
                                n_threads, x_init) {
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
                store_Q    = FALSE
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
                                n_threads, x_init) {
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
                store_Q    = FALSE
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
