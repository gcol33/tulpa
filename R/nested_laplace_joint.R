#' Joint multi-likelihood nested Laplace approximation
#'
#' @description
#' Outer-grid nested Laplace driver for *joint* models — multiple response
#' arms sharing one latent prior block, with optional INLA-style copy-scaling
#' (`f(..., copy=, fixed=FALSE)`) on a designated arm.
#'
#' Supported priors (Phase 1c, BYM2-only):
#'  * `"bym2"` — 3D outer grid over `(sigma_spatial, rho, alpha)` when a
#'    `copy` arm is set; 2D over `(sigma_spatial, rho)` otherwise.
#'
#' Other backends (ICAR, CAR_proper, NNGP, HSGP, RW1/2, AR1) follow the same
#' interface and land under Phase 3 / P1c-7.
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
#'   field `type`. For `type = "bym2"`:
#'   * `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`, `n_neighbors`
#'     (CSR adjacency, 0-based as in [tulpa_nested_laplace()]),
#'   * `scale_factor` (Riebler scaling constant; default `1`),
#'   * optional `sigma_grid`, `rho_grid` (defaults: 5 log-spaced sigmas
#'     and `c(0.2, 0.5, 0.8, 0.95)` for rho; Cartesian product is built
#'     internally).
#'
#' @param copy Optional list controlling the copy-scaling (alpha) arm:
#'   * `arm`         — name (or 1-based index) of the arm to copy-scale.
#'   * `alpha_grid`  — numeric grid for alpha; default
#'                     `c(0, 0.5, 1, 1.5, 2)`.
#'   When `NULL` (default), no copy-scaling is applied (all arms see the
#'   un-scaled shared latent — equivalent to alpha = 1 fixed).
#'
#' @param max_iter,tol Inner Newton iteration budget and tolerance.
#' @param n_threads OpenMP threads.
#' @param x_init Optional warm-start for the first grid point's inner solve.
#' @param verbose Print grid-point progress (currently a no-op; reserved).
#'
#' @return A list of class `c("tulpa_nested_laplace_joint",
#'   "tulpa_nested_laplace", "list")` with:
#'   * `theta_grid`, `theta_names` — outer-grid hyperparameter values.
#'   * `log_marginal`, `weights` — per-grid-point log-marginal and integration
#'      weights (sum to 1).
#'   * `theta_mean`, `theta_sd` — posterior moments per hyperparameter.
#'   * `modes` — `[n_grid x n_x]` matrix of inner modes.
#'   * `n_iter` — inner Newton iterations per grid point.
#'   * `arm_layout` — list with `beta_start`, `re_start`, `phi_start`,
#'      `theta_start`, `n_x` for decoding modes back into per-arm components.
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
    if (type != "bym2") {
        stop("tulpa_nested_laplace_joint() currently supports only ",
             "prior$type = 'bym2' (Phase 1c). Got '", type, "'.", call. = FALSE)
    }

    arms <- lapply(seq_along(responses), function(k) {
        a <- responses[[k]]
        .normalise_joint_arm(a, k)
    })

    n_arms <- length(arms)
    cp <- .resolve_copy(copy, responses)

    grids <- .default_bym2_joint_grids(prior, cp$has_copy, cp$alpha_grid)

    res <- cpp_nested_laplace_joint_bym2(
        arms_list = arms,
        copy_arm  = as.integer(cp$copy_arm_zero),
        n_spatial_units = as.integer(prior$n_spatial_units),
        adj_row_ptr = as.integer(prior$adj_row_ptr),
        adj_col_idx = as.integer(prior$adj_col_idx),
        n_neighbors = as.integer(prior$n_neighbors),
        scale_factor = as.numeric(prior$scale_factor %||% 1.0),
        sigma_spatial_grid = as.numeric(grids$sigma),
        rho_grid           = as.numeric(grids$rho),
        alpha_grid         = as.numeric(grids$alpha),
        max_iter = as.integer(max_iter),
        tol      = as.numeric(tol),
        n_threads = as.integer(n_threads),
        x_init_nullable = x_init,
        store_Q = FALSE
    )

    arm_layout <- .joint_bym2_layout(arms, prior$n_spatial_units)

    if (cp$has_copy) {
        res$theta_grid <- cbind(sigma = grids$sigma,
                                rho   = grids$rho,
                                alpha = grids$alpha)
        res$theta_names <- c("sigma", "rho", "alpha")
    } else {
        res$theta_grid <- cbind(sigma = grids$sigma,
                                rho   = grids$rho)
        res$theta_names <- c("sigma", "rho")
    }

    res$weights    <- .nl_normalise_weights(res$log_marginal)
    res            <- .nl_posterior_moments(res, "joint_bym2")
    res$arm_layout <- arm_layout
    res$prior      <- prior
    res$responses  <- responses
    res$copy       <- copy
    class(res) <- c("tulpa_nested_laplace_joint", "tulpa_nested_laplace", "list")
    res
}


# --- helpers -----------------------------------------------------------------

# Validate one arm spec and fill in defaults. Returns a list ready to hand
# to the C++ kernel (which Rcpp::as<>'s each named field).
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

# Build the paired (sigma, rho[, alpha]) outer grids. Each axis is the
# Cartesian product of the per-axis defaults; the C++ shim consumes paired
# vectors, so we expand here.
.default_bym2_joint_grids <- function(prior, has_copy, alpha_axis) {
    sigma_axis <- prior$sigma_grid %||% exp(seq(log(0.1), log(3), length.out = 5))
    rho_axis   <- prior$rho_grid   %||% c(0.2, 0.5, 0.8, 0.95)
    if (has_copy) {
        gr <- expand.grid(sigma = sigma_axis, rho = rho_axis, alpha = alpha_axis,
                          KEEP.OUT.ATTRS = FALSE)
        return(list(sigma = gr$sigma, rho = gr$rho, alpha = gr$alpha))
    }
    gr <- expand.grid(sigma = sigma_axis, rho = rho_axis,
                      KEEP.OUT.ATTRS = FALSE)
    list(sigma = gr$sigma, rho = gr$rho, alpha = numeric(0))
}

# Compute the per-arm latent offsets so callers can decode `modes` back into
# (beta_k, re_k, phi, theta) per arm.
.joint_bym2_layout <- function(arms, n_spatial_units) {
    n_arms <- length(arms)
    p_arm   <- vapply(arms, function(a) ncol(a$X),       integer(1))
    n_re    <- vapply(arms, function(a) a$n_re_groups,    integer(1))

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
    phi_start_   <- cur
    theta_start_ <- phi_start_ + n_spatial_units
    n_x          <- phi_start_ + 2L * n_spatial_units

    list(
        n_arms      = n_arms,
        p           = p_arm,
        n_re        = n_re,
        beta_start  = beta_start,
        re_start    = re_start,
        phi_start   = phi_start_,
        theta_start = theta_start_,
        n_x         = n_x
    )
}
