# =============================================================================
# cila.R -- corrected integrated Laplace as an inner-layer DEBIAS.
#
# The engine's composition is nested approximation plus debias. Two inner
# debiases now ship, and they differ in what they select rather than in what
# they approximate. The subspace debias (R/subspace_debias.R) picks the
# coordinates the inner-layer bands flagged and runs exact Metropolis on them,
# carrying the rest at their Gaussian conditional. This one selects nothing: it
# draws the WHOLE inner Gaussian, reweights every draw by the exact joint
# density it came from, and reports the weighted particles. Effort M is the only
# dial, and increasing it drives both the cell marginal and the latent posterior
# toward the exact ones (Lai, Margossian & Sheldon, arXiv:2605.20345,
# Proposition 1 and Proposition 5).
#
# WHAT THE C++ SIDE HANDS BACK AND WHY THE POOLING IS ONE LINE. Cell k returns
# its M unnormalized log ratios `lw_ki` and the leading fixed-effect prefix of
# each draw. The cell's corrected mass is proportional to q_k * mean_i w_ki
# (eq 4) and the within-cell recovery weight to w_ki (Proposition 5), where q_k
# is the outer grid's own quadrature-times-hyperprior weight. The product is
# q_k * w_ki, so pooling the whole grid is ONE softmax over `log_q_k + lw_ki`
# and the corrected hyperparameter marginal falls out of the same numbers.
# `log_q_k` is read off the fit as `log(weights_k) - log_marginal_k`, which is
# the grid weight with the Laplace marginal divided back out.
#
# ONE GRID WEIGHTING PER FIT. The pooled masses ARE the fit's grid posterior --
# `fit$draws` is built from them -- so a corrected fit carries them as
# `fit$weights` and `fit$log_marginal`, and `fit$weights_source` says which read
# produced them. The pre-correction pair is kept as `fit$cila$laplace$weights` /
# `$log_marginal`. `fit$cila$cell_weights` and `$cell_log_marginal` are the
# adopted vectors themselves, so pairing them with `fit$weights` cannot go
# wrong: they are the same numbers.
#
# HOW IT IS GRADED. The importance weights are the same object PSIS already
# scores elsewhere in this engine, so `tulpa_psis()` fits them directly -- one
# Pareto fit in the package, not a third one. A relative ESS and a Pareto shape
# travel on the fit next to the correction, so a fit says how well its own
# correction is behaving instead of the caller assuming it behaved.
# =============================================================================

# Variant codes shared with src/inner_cila.h's CilaVariant.
.CILA_VARIANTS <- c(qmc = 0L, is = 1L, rqmc = 2L)

# The recovered posterior of Proposition 5 is literally sum_i w_i delta_{z_i},
# so at small M a truth outside the particle range gets PIT exactly 0 or 1 and
# the reported marginal is an atom rather than a distribution. Every variant
# was measured leaving the simultaneous calibration band at M = 64, and the iid
# one still leaving it at M = 256. The front door therefore floors the
# effort rather than reporting a particle set too coarse to be a marginal.
.CILA_MIN_POINTS <- 512L

# Normalize `control$cila` into a settings list, or NULL when the correction is
# off. `TRUE` takes every default; a list overrides the entries it names.
.cila_config <- function(spec) {
    if (is.null(spec) || identical(spec, FALSE)) return(NULL)
    if (isTRUE(spec)) spec <- list()
    if (!is.list(spec)) {
        stop("`control$cila` must be TRUE, FALSE, or a list of settings.",
             call. = FALSE)
    }
    known <- c("n_points", "variant", "n_shift", "n_draws", "seed")
    unknown <- setdiff(names(spec), known)
    if (length(unknown)) {
        stop(sprintf("Unknown `control$cila` setting(s): %s.\nAllowed: %s.",
                     paste(sQuote(unknown, q = FALSE), collapse = ", "),
                     paste(known, collapse = ", ")), call. = FALSE)
    }
    variant <- tolower(as.character(spec$variant %||% "qmc"))
    if (!variant %in% names(.CILA_VARIANTS)) {
        stop("`control$cila$variant` must be one of ",
             paste(names(.CILA_VARIANTS), collapse = ", "), ".", call. = FALSE)
    }
    n_points <- as.integer(spec$n_points %||% 1024L)
    if (!is.finite(n_points) || n_points <= 0) {
        stop("`control$cila$n_points` must be a positive integer.",
             call. = FALSE)
    }
    if (n_points < .CILA_MIN_POINTS) {
        stop(sprintf(paste0("`control$cila$n_points` is %d; the correction ",
                            "reports a weighted particle set, which is not a ",
                            "usable marginal below %d points per cell ",
                            "(the calibration band was measured leaving at ",
                            "M = 64 and M = 256)."),
                     n_points, .CILA_MIN_POINTS), call. = FALSE)
    }
    list(n_points = n_points,
         variant  = variant,
         n_shift  = as.integer(spec$n_shift %||% 8L),
         n_draws  = as.integer(spec$n_draws %||% .nl_diag("debias_n_draws")),
         seed     = as.numeric(spec$seed %||% 5898573388634117.0))
}

# The kernel-facing form of the request: the effort, the variant code, the
# retained latent prefix, and the stream seed, in ONE list. Every nested kernel
# entry takes it under that name (`unwrap_cila` in src/laplace_spec_fit.h).
.cila_request <- function(cfg, n_fixed) {
    if (is.null(cfg) || !isTRUE(as.integer(n_fixed) > 0L)) return(NULL)
    list(n_points = as.integer(cfg$n_points),
         variant  = as.integer(.CILA_VARIANTS[[cfg$variant]]),
         n_shift  = as.integer(cfg$n_shift),
         n_fixed  = as.integer(n_fixed),
         seed     = as.numeric(cfg$seed))
}

# What a fit records about the correction, whether or not anything was
# corrected -- so a fit that WAS corrected can be told from one that was not
# without re-reading the weights.
#
# `laplace` is the PRE-correction grid read, kept under a name that says so. It
# is the only other grid weighting a corrected fit carries: `cell_weights` /
# `cell_log_marginal` are the adopted ones, identical to `fit$weights` /
# `fit$log_marginal` by construction.
.cila_record <- function(cfg, declined = NA_character_, n_cells = NA_integer_,
                         n_declined = NA_integer_, variant_used = NA_character_,
                         fallback = NA_character_, pareto_k = NA_real_,
                         rel_ess = NA_real_, cell_weights = NULL,
                         cell_log_marginal = NULL, adopted = FALSE,
                         retained_mass = NA_real_, laplace = NULL) {
    list(n_points = cfg$n_points, variant = cfg$variant,
         n_shift = cfg$n_shift, n_draws = cfg$n_draws, seed = cfg$seed,
         variant_used = variant_used, fallback = fallback,
         n_cells = n_cells, n_cells_declined = n_declined,
         pareto_k = pareto_k, rel_ess = rel_ess,
         cell_weights = cell_weights, cell_log_marginal = cell_log_marginal,
         weights_adopted = adopted, retained_mass = retained_mass,
         laplace = laplace,
         declined = declined)
}

# Pool one grid's per-cell particles into the corrected posterior.
#
# `lw_per_cell` / `fixed_per_cell` are the kernel's per-cell log ratios and
# [M x p] draw prefixes; `log_q` the grid's own log quadrature-times-hyperprior
# weight per cell. Returns the pooled particles, their normalized weights, the
# corrected per-cell masses, and the PSIS grade of the pooled weights -- or NULL
# when no cell produced a usable set.

# eq (4)'s cell estimator: the mean is over ALL M auxiliary points, so a point
# the joint density could not be evaluated at counts as zero weight rather than
# shrinking the denominator.
.cila_logmeanexp <- function(x) {
    n <- length(x)
    x <- x[is.finite(x)]
    if (!length(x) || !n) return(-Inf)
    .tulpa_logsumexp(x) - log(n)
}

.cila_pool <- function(lw_per_cell, fixed_per_cell, log_q) {
    keep <- which(vapply(seq_along(lw_per_cell), function(k) {
        lw <- lw_per_cell[[k]]
        fx <- fixed_per_cell[[k]]
        !is.null(lw) && !is.null(fx) && length(lw) == nrow(fx) &&
            is.finite(log_q[k]) && any(is.finite(lw))
    }, logical(1)))
    if (!length(keep)) return(NULL)

    cell_lm <- vapply(keep, function(k) {
        .cila_logmeanexp(as.numeric(lw_per_cell[[k]]))
    }, numeric(1))
    cell_log_mass <- cell_lm + log_q[keep]
    cell_w <- rep(0, length(log_q))
    cell_w[keep] <- exp(cell_log_mass - .tulpa_logsumexp(cell_log_mass))
    cell_lm_full <- rep(NA_real_, length(log_q))
    cell_lm_full[keep] <- cell_lm

    pooled_lw <- unlist(lapply(seq_along(keep), function(j) {
        as.numeric(lw_per_cell[[keep[j]]]) + log_q[keep[j]]
    }), use.names = FALSE)
    particles <- do.call(rbind, lapply(keep, function(k) {
        as.matrix(fixed_per_cell[[k]])
    }))
    ok <- is.finite(pooled_lw)
    if (!any(ok)) return(NULL)
    pooled_lw <- pooled_lw[ok]
    particles <- particles[ok, , drop = FALSE]
    w <- exp(pooled_lw - .tulpa_logsumexp(pooled_lw))

    grade <- tryCatch(tulpa_psis(pooled_lw - max(pooled_lw)),
                      error = function(e) NULL)
    list(particles = particles, w = w, cell_weights = cell_w,
         cell_log_marginal = cell_lm_full, keep = keep,
         n_cells = length(log_q), n_declined = length(log_q) - length(keep),
         pareto_k = if (is.null(grade)) NA_real_ else as.numeric(grade$pareto_k),
         rel_ess = 1 / (sum(w^2) * length(w)))
}

# Resample the pooled particle set into the fixed-effect draw matrix every
# coefficient-facing method already reads. Same contract as the subspace
# debias: once a coordinate is corrected the posterior is no longer the grid's
# Gaussian mixture, so the fit reports draws instead of moments.
.cila_draws <- function(pool, n_draws, beta_names) {
    p <- ncol(pool$particles)
    if (is.null(p) || p < 1L) return(NULL)
    rows <- sample.int(nrow(pool$particles), as.integer(n_draws),
                       replace = TRUE, prob = pool$w)
    out <- pool$particles[rows, , drop = FALSE]
    if (!is.null(beta_names) && length(beta_names) == p) colnames(out) <- beta_names
    out
}

# The escalation itself, for a backend that integrates an outer hyperparameter
# grid. One redispatch, one pooling, one record.
#
# `redispatch(request)` re-runs the fit's OWN settled grid with the correction
# on and returns the kernel result carrying the per-cell log ratios and draw
# prefixes. That second pass is the cost, and it is the same shape the subspace
# debias pays: the corrected shape is a property of each cell, and which cells
# the fit settled on is only known once the first pass has run.
#
# `remoments(res) -> res` is the fitter's OWN hyperparameter-summary tail, run
# again once the corrected grid read has been adopted. Each
# fitter passes the sequence it already runs, so `theta_mean` / `theta_sd` /
# `theta_median` / `theta_ci_*` are recomputed by the same code that produced
# them rather than by a second summariser written for the correction.
.nl_cila_attach <- function(res, cfg, redispatch, p_fixed, beta_names = NULL,
                            remoments = identity) {
    if (is.null(cfg)) return(res)
    res$weights_source <- "laplace_grid"
    p_fixed <- as.integer(p_fixed %||% 0L)
    if (!length(p_fixed) || is.na(p_fixed) || p_fixed < 1L) {
        res$cila <- .cila_record(cfg, declined = "no_fixed_effects")
        return(res)
    }
    lm <- as.numeric(res$log_marginal %||% numeric(0))
    wt <- as.numeric(res$weights %||% numeric(0))
    if (!length(lm) || length(lm) != length(wt)) {
        res$cila <- .cila_record(cfg, declined = "no_grid_weights")
        return(res)
    }
    # The grid's own quadrature-times-hyperprior weight, with the Laplace
    # marginal the correction replaces divided back out. Defined up to an
    # additive constant, which the pooling's softmax removes.
    log_q <- log(wt) - lm
    log_q[!is.finite(log_q)] <- -Inf

    out <- tryCatch(redispatch(.cila_request(cfg, p_fixed)),
                    error = function(e) NULL)
    if (is.null(out) || is.null(out$cila_log_w_per_grid) ||
        is.null(out$cila_fixed_per_grid)) {
        res$cila <- .cila_record(cfg, declined = "redispatch_failed")
        return(res)
    }
    pool <- .cila_pool(out$cila_log_w_per_grid, out$cila_fixed_per_grid, log_q)
    if (is.null(pool)) {
        res$cila <- .cila_record(
            cfg, declined = .cila_first_reason(out) %||% "no_usable_cell",
            n_cells = length(log_q), n_declined = length(log_q))
        return(res)
    }
    draws <- .cila_draws(pool, cfg$n_draws, beta_names %||% res$fixed_names)
    if (is.null(draws)) {
        res$cila <- .cila_record(cfg, declined = "no_particles",
                                 n_cells = pool$n_cells,
                                 n_declined = pool$n_declined)
        return(res)
    }
    res$draws <- draws
    res$n_fixed <- res$n_fixed %||% p_fixed

    # Adopt the corrected grid read. `fit$draws` are pooled
    # from the corrected per-cell masses, so those masses ARE the fit's grid
    # posterior and the fit carries them under the name every reader of a grid
    # weighting already uses. The pre-correction pair travels under `$laplace`,
    # a name that cannot be mistaken for the reported one, and the hyperparameter
    # summaries are recomputed from the adopted pair by the fitter's own tail --
    # so no two same-shaped vectors on the fit disagree about the grid's mass.
    #
    # A cell that produced no usable particle set is dropped, and the read is
    # then conditional on the cells that remain, the same convention a
    # repaired grid takes. `retained_mass` is the share
    # of the ORIGINAL Laplace mass those cells carried, 1 on a complete grid, so
    # a reader tells the two apart from the fit alone.
    lm_adopt <- as.numeric(pool$cell_log_marginal)
    lm_adopt[!is.finite(lm_adopt)] <- -Inf
    retained <- if (sum(wt, na.rm = TRUE) > 0) {
        sum(wt[pool$keep], na.rm = TRUE) / sum(wt, na.rm = TRUE)
    } else NA_real_
    res$log_marginal   <- lm_adopt
    res$weights        <- pool$cell_weights
    res$weights_source <- "cila"

    res$cila <- .cila_record(
        cfg, n_cells = pool$n_cells, n_declined = pool$n_declined,
        variant_used = .cila_variant_used(out),
        fallback = .cila_first_fallback(out) %||% NA_character_,
        pareto_k = pool$pareto_k, rel_ess = pool$rel_ess,
        cell_weights = pool$cell_weights,
        cell_log_marginal = lm_adopt,
        adopted = TRUE, retained_mass = retained,
        laplace = list(weights = wt, log_marginal = lm))
    remoments(res)
}

# The first reason any cell gave, when no cell produced a usable set.
.cila_first_reason <- function(out) {
    d <- as.character(out$cila_declined %||% character(0))
    d <- d[!is.na(d) & nzchar(d)]
    if (!length(d)) NULL else d[1L]
}

.cila_first_fallback <- function(out) {
    f <- as.character(out$cila_fallback %||% character(0))
    f <- f[!is.na(f) & nzchar(f)]
    if (!length(f)) NULL else f[1L]
}

# The variant the cells actually ran, which is the requested one unless the
# Sobol table could not cover the latent dimension.
.cila_variant_used <- function(out) {
    v <- as.integer(out$cila_variant %||% integer(0))
    v <- v[!is.na(v)]
    if (!length(v)) return(NA_character_)
    names(.CILA_VARIANTS)[match(v[1L], .CILA_VARIANTS)]
}
