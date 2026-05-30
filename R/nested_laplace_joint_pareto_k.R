# Outer Pareto-k-hat for the joint nested-Laplace backend.
#
# Part of the joint nested-Laplace driver; the public entry point
# tulpa_nested_laplace_joint() lives in nested_laplace_joint.R.
#
# The re-cov / generic-single-axis / SPDE paths importance-sample a single
# positive-scale axis on the log scale (see R/psis.R, R/fit_spde_nested.R).
# The joint backend's hyperparameter space is heterogeneous -- positive
# scales (sigma, tau, phi_*, ...), a bounded BYM2 mixing weight (rho), an
# unbounded copy coefficient (alpha), and a CAR_proper correlation
# (rho_car) whose support is the adjacency's eigenvalue interval and is not
# safely guessable. Each axis therefore carries its own unconstraining
# transform + log-Jacobian; a fit carrying any axis whose support is not
# safely known DECLINES to the quadrature-ESS fallback rather than apply a
# guessed transform (never a wrong k-hat). This mirrors the generic path's
# decline on a bounded `rho_grid` (gcol33/tulpa#42).

# Positive-scale axes integrated on the log scale (theta = exp(u), Jacobian
# d theta / d u = theta, log-Jacobian u). Per-arm dispersion axes carry the
# `phi_` prefix and join this set by name.
.JOINT_POS_AXES <- c("sigma", "tau", "sigma2", "phi_gp", "ell", "lengthscale",
                     "range", "sigma_spatial", "sigma_occ", "sigma_pos",
                     "sigma_1", "sigma_2")

# Per-axis transform tag for one block. `type` is the (lower-case) block type,
# `axes` the bare axis names (block prefix already stripped). Returns one tag
# per axis ("log" / "logit01" / "identity") or NULL to DECLINE the whole fit
# when any axis has a support that cannot be safely transformed:
#   * `rho` is the BYM2 mixing weight in (0, 1) -> logit, but for CAR_proper /
#     AR1 / multi-output HSGP the same name denotes an eigenvalue- or
#     autocorrelation-bounded parameter whose support is not (0, 1); decline.
#   * `rho_car` is the proper-CAR correlation on the adjacency eigenvalue
#     interval; decline.
#   * `alpha` is the unbounded copy coefficient; identity.
.joint_pareto_block_tags <- function(type, axes) {
    tag_one <- function(a) {
        if (a == "alpha") return("identity")
        if (a == "rho") return(if (identical(type, "bym2")) "logit01" else NA_character_)
        if (a == "rho_car") return(NA_character_)
        if (a %in% .JOINT_POS_AXES || startsWith(a, "phi_")) return("log")
        NA_character_
    }
    tags <- vapply(axes, tag_one, character(1))
    if (anyNA(tags)) return(NULL)
    tags
}

# Resolve the per-column unconstraining transforms for a joint result.
# Walks the result's block layout (multi-block: `axis_offsets` + `blocks`;
# single-block: the lone `prior$type`) so the `rho` ambiguity above is
# resolved by the block that owns the axis, not by name alone. Trailing
# `phi_<arm>` dispersion columns (appended after the latent-block axes) are
# positive-scale (log). Returns a character vector of tags, one per
# `theta_grid` column, or NULL to decline.
.joint_pareto_axis_tags <- function(res) {
    tg <- res$theta_grid
    if (is.null(tg) || !is.matrix(tg) || ncol(tg) == 0L) return(NULL)
    cn <- colnames(tg)
    d  <- ncol(tg)
    tags <- rep(NA_character_, d)

    bare <- function(idx) sub("^b[0-9]+\\.", "", cn[idx])

    if (!is.null(res$axis_offsets) && !is.null(res$blocks)) {
        ao <- as.integer(res$axis_offsets)
        B  <- length(res$blocks)
        for (b in seq_len(B)) {
            if (ao[b + 1L] <= ao[b]) next                  # block carries no axis
            cols <- (ao[b] + 1L):ao[b + 1L]
            t_b <- .joint_pareto_block_tags(tolower(res$blocks[[b]]$type %||% ""),
                                            bare(cols))
            if (is.null(t_b)) return(NULL)
            tags[cols] <- t_b
        }
        # Columns past the last block axis are per-arm phi dispersion axes.
        if (ao[B + 1L] < d) {
            extra <- (ao[B + 1L] + 1L):d
            if (!all(startsWith(cn[extra], "phi_"))) return(NULL)
            tags[extra] <- "log"
        }
    } else {
        # Single-block: every non-phi column belongs to the lone prior block;
        # phi_<arm> columns are positive-scale dispersion axes.
        type <- tolower(res$prior$type %||% "")
        is_phi <- startsWith(cn, "phi_")
        if (any(!is_phi)) {
            t_blk <- .joint_pareto_block_tags(type, bare(which(!is_phi)))
            if (is.null(t_blk)) return(NULL)
            tags[!is_phi] <- t_blk
        }
        tags[is_phi] <- "log"
    }
    if (anyNA(tags)) return(NULL)
    tags
}

# Forward (constrained -> unconstrained) transform for one axis.
.joint_pareto_fwd <- function(tag, theta) {
    switch(tag,
        log      = log(theta),
        logit01  = stats::qlogis(theta),
        identity = theta)
}

# Inverse (unconstrained -> constrained) transform plus log|d theta / d u|.
.joint_pareto_inv <- function(tag, u) {
    switch(tag,
        log = list(theta = exp(u), log_jac = u),
        logit01 = {
            p <- stats::plogis(u)
            # log p + log(1 - p), stable form via plogis(., log.p = TRUE).
            list(theta = p,
                 log_jac = stats::plogis(u, log.p = TRUE) +
                           stats::plogis(-u, log.p = TRUE))
        },
        identity = list(theta = u, log_jac = rep(0, length(u))))
}

# Shared outer Pareto-k-hat driver for a joint nested-Laplace result.
#
# Fits a Gaussian proposal to the joint hyperparameter posterior in the
# per-axis unconstrained coordinate `u` (weighted mean + covariance over the
# integration grid, the analogue of `.nested_grid_pareto_k` generalised to
# mixed support), draws `n_samples` from it, re-evaluates the inner joint
# marginal at the back-transformed grid via `refit_log_marginal` (a closure
# that round-trips an `S x d` user-facing theta matrix through the SAME joint
# kernel the integrator used and returns the per-cell log-marginal WITH the
# baked hyperprior), adds the summed per-axis log-Jacobian, and PSIS-smooths.
# The proposal's normalizing constant is common to every draw and drops under
# PSIS, so only the quadratic enters (handled inside `.nested_is_pareto_k`).
# Runs with the RNG restored so the fit's draws are bit-for-bit unchanged.
#
# Returns list(pareto_k, is_ess): both NA when the fit declines (an axis with
# unguessable support, a degenerate proposal covariance, or too few finite
# importance draws), in which case the diagnostic layer reports quad-ESS.
.joint_pareto_k <- function(res, refit_log_marginal, n_samples = 200L) {
    tags <- .joint_pareto_axis_tags(res)
    if (is.null(tags)) return(list(pareto_k = NA_real_, is_ess = NA_real_))

    tg <- res$theta_grid
    w  <- res$weights
    if (is.null(w) || length(w) != nrow(tg) || !is.finite(sum(w)) || sum(w) <= 0) {
        return(list(pareto_k = NA_real_, is_ess = NA_real_))
    }
    cn <- colnames(tg)
    d  <- ncol(tg)

    u_grid <- matrix(0, nrow(tg), d)
    for (j in seq_len(d)) {
        u_grid[, j] <- .joint_pareto_fwd(tags[j], as.numeric(tg[, j]))
    }
    if (any(!is.finite(u_grid))) return(list(pareto_k = NA_real_, is_ess = NA_real_))

    u_hat <- as.numeric(crossprod(w, u_grid))
    cen   <- sweep(u_grid, 2L, u_hat)
    Su    <- crossprod(cen * w, cen)
    Su    <- (Su + t(Su)) / 2
    L <- tryCatch(t(chol(Su)), error = function(e) NULL)
    if (is.null(L)) return(list(pareto_k = NA_real_, is_ess = NA_real_))

    lt <- function(U) {
        theta_mat <- matrix(0, nrow(U), ncol(U))
        log_jac   <- numeric(nrow(U))
        for (j in seq_len(ncol(U))) {
            iv <- .joint_pareto_inv(tags[j], U[, j])
            theta_mat[, j] <- iv$theta
            log_jac <- log_jac + iv$log_jac
        }
        colnames(theta_mat) <- cn
        lm <- refit_log_marginal(theta_mat)
        if (length(lm) != nrow(U)) return(rep(-Inf, nrow(U)))
        lm + log_jac
    }

    has_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    old_seed <- if (has_seed) get(".Random.seed", envir = .GlobalEnv) else NULL
    kd <- tryCatch(.nested_is_pareto_k(u_hat, L, lt, n_samples),
                   error = function(e) NULL)
    if (!is.null(old_seed)) assign(".Random.seed", old_seed, envir = .GlobalEnv)
    if (is.null(kd)) return(list(pareto_k = NA_real_, is_ess = NA_real_))
    list(pareto_k = kd$pareto_k, is_ess = kd$is_ess)
}

# Single-block joint wrapper. Reuses the driver's already-built generic
# `kernel_fn` (the closure refinement passes drive, which round-trips a
# user-facing cell matrix through `backend$call_kernel`) and `hp_fn` (the
# baked-hyperprior closure) as the re-evaluation path, so no kernel-call
# machinery is duplicated. Attaches `pareto_k` / `pareto_k_is_ess` /
# `pareto_k_scope`; with `diagnose_k = FALSE` the fields are present but NA.
.joint_attach_pareto_k_single <- function(res, kernel_fn, hp_fn,
                                          diagnose_k = TRUE, k_samples = 200L) {
    res$pareto_k        <- NA_real_
    res$pareto_k_is_ess <- NA_real_
    res$pareto_k_scope  <- "outer (hyperparameter) Gaussian proposal"
    if (!isTRUE(diagnose_k)) return(res)

    refit <- function(theta_mat) {
        r  <- kernel_fn(theta_mat)
        lm <- r$log_marginal
        if (!is.null(hp_fn)) {
            hp <- hp_fn(theta_mat)
            if (!is.null(hp) && length(hp) == length(lm)) lm <- lm + hp
        }
        lm
    }
    kd <- .joint_pareto_k(res, refit, k_samples)
    res$pareto_k        <- kd$pareto_k
    res$pareto_k_is_ess <- kd$is_ess
    res
}
