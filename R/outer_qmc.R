# Importance quadrature in transformed hyperparameters. A Student mixture has
# positive density throughout the coordinate space, including variance arms
# whose mass is small on the placement grid.
.nl_qmc_uniform <- function(n, d, start = 0L, seed = NULL) {
    if (!is.null(seed)) {
        had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
        if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
        on.exit(if (had_seed) assign(".Random.seed", old_seed, .GlobalEnv)
                else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
                    rm(".Random.seed", envir = .GlobalEnv))
        set.seed(seed)
    }
    primes <- integer(0)
    p <- 2L
    while (length(primes) < d) {
        if (!any(p %% primes[primes <= sqrt(p)] == 0L)) primes <- c(primes, p)
        p <- p + 1L
    }
    out <- matrix(0, n, d)
    for (j in seq_len(d)) {
        k <- start + seq_len(n)
        f <- 1 / primes[j]
        while (if (is.null(seed)) any(k > 0L) else f > .Machine$double.eps) {
            digit <- k %% primes[j]
            if (!is.null(seed)) {
                perm <- sample.int(primes[j]) - 1L
                digit <- perm[digit + 1L]
            }
            out[, j] <- out[, j] + f * digit
            k <- k %/% primes[j]
            f <- f / primes[j]
        }
        if (!is.null(seed)) out[, j] <- (out[, j] + runif(1L)) %% 1
    }
    out
}

.nl_qmc_proposal <- function(U, w, components = .NL_QMC$components,
                              df = .NL_QMC$df) {
    w <- w / sum(w)
    d <- ncol(U)
    mu <- as.numeric(crossprod(w, U))
    Z <- sweep(U, 2L, mu)
    variance <- colSums(w * Z^2)
    groups <- list(seq_len(nrow(U)))
    while (length(groups) < components) {
        mass <- vapply(groups, function(ix) sum(w[ix]), numeric(1))
        g <- which.max(mass)
        ix <- groups[[g]]
        if (length(ix) < 2L) break
        wg <- w[ix] / sum(w[ix])
        m <- as.numeric(crossprod(wg, U[ix, , drop = FALSE]))
        z <- sweep(U[ix, , drop = FALSE], 2L, m)
        j <- which.max(colSums(wg * z^2) / pmax(variance, .Machine$double.eps))
        o <- order(U[ix, j])
        cut <- which(cumsum(wg[o]) >= .5)[1L]
        cut <- max(1L, min(length(o) - 1L, cut))
        groups[[g]] <- ix[o[seq_len(cut)]]
        groups[[length(groups) + 1L]] <- ix[o[seq.int(cut + 1L, length(o))]]
    }
    make <- function(m, v, mass) {
        ridge <- pmax(variance, .Machine$double.eps * max(variance)) * .05
        scale <- sqrt(v + ridge)
        list(mu = m, scale = scale, mass = mass, df = df,
             logdet = sum(log(scale)))
    }
    fit <- function(ix, mass) {
        wg <- w[ix] / sum(w[ix])
        m <- as.numeric(crossprod(wg, U[ix, , drop = FALSE]))
        z <- sweep(U[ix, , drop = FALSE], 2L, m)
        make(m, 1.5 * colSums(wg * z^2), mass)
    }
    ans <- lapply(groups, function(ix) fit(ix, .55 * sum(w[ix])))
    tails <- list()
    for (j in seq_len(d)) for (upper in c(FALSE, TRUE)) {
        o <- order(U[, j], decreasing = upper)
        ix <- o[seq_len(which(cumsum(w[o]) >= .1)[1L])]
        tails[[length(tails) + 1L]] <- fit(ix, .3 / (2 * d))
    }
    c(ans, tails, list(make(mu, 4 * variance, .15)))
}

.nl_qmc_nodes <- function(proposal, n, start = 0L, seed = 1729L) {
    d <- length(proposal[[1L]]$mu)
    if (!n) return(list(u = matrix(numeric(0), 0L, d), log_q = numeric(0),
                        proposal = proposal))
    mass <- vapply(proposal, `[[`, numeric(1), "mass")
    mass <- mass / sum(mass)
    previous <- diff(c(0L, round(cumsum(mass) * start)))
    counts <- diff(c(0L, round(cumsum(mass) * (start + n)))) - previous
    id <- rep(seq_along(proposal), counts)
    for (g in seq_along(proposal)) proposal[[g]]$mass <- counts[g] / n
    U <- matrix(0, n, d)
    for (g in seq_along(proposal)) {
        ix <- which(id == g)
        if (!length(ix)) next
        p <- proposal[[g]]
        v <- .nl_qmc_uniform(length(ix), d, previous[g], seed + 7919L * g)
        z <- matrix(0, length(ix), d)
        for (j in seq_len(d)) {
            z[, j] <- stats::qt(v[, j], p$df)
        }
        U[ix, ] <- sweep(sweep(z, 2L, p$scale, "*"), 2L, p$mu, "+")
    }
    lp <- vapply(proposal, function(p) {
        z <- sweep(sweep(U, 2L, p$mu), 2L, p$scale, "/")
        log(p$mass) + rowSums(stats::dt(z, p$df, log = TRUE)) - p$logdet
    }, numeric(n))
    mx <- apply(lp, 1L, max)
    list(u = U, log_q = mx + log(rowSums(exp(lp - mx))), proposal = proposal)
}

.nl_outer_quantile_summary <- function(values, variance, weights, probs) {
    gaussian <- !is.null(variance)
    quantile <- function(p, rows = seq_along(values)) {
        if (!gaussian) return(.nl_wtd_quantile(values[rows], weights[rows], p, outside = "clamp"))
        summary <- .nl_gauss_mixture_summary(matrix(values[rows], ncol = 1L),
            matrix(variance[rows], ncol = 1L), weights[rows], p)
        if (is.null(summary)) rep(NA_real_, length(p)) else as.numeric(summary$quantiles)
    }
    if (gaussian) {
        full <- .nl_gauss_mixture_summary(matrix(values, ncol = 1L),
            matrix(variance, ncol = 1L), weights, probs)
        if (is.null(full)) return(NULL)
        sd <- full$sd
        location <- as.numeric(full$quantiles)
    } else {
        mean <- sum(weights * values)
        sd <- sqrt(sum(weights * (values - mean)^2))
        location <- quantile(probs)
    }
    list(quantile = quantile, location = location, sd = sd)
}

.nl_outer_error_context <- function(weights, replicate, proposed, family) {
    groups <- lapply(seq_along(proposed), function(g) which(replicate == g))
    critical <- stats::qt(1 - .05 / (2 * family), length(groups) - 1L)
    z <- vapply(groups, function(at) sum(weights[at]), numeric(1)) / proposed
    relative <- critical * stats::sd(z) / sqrt(length(z)) / mean(z)
    list(groups = groups, critical = critical, z = z, mean = mean(z), relative = relative,
         log_evidence = if (relative < 1) -log1p(-relative) else Inf)
}

.nl_outer_jackknife_dominated <- function(values, variance, weights, groups,
                                          distribution, budget, critical, probs) {
    if (length(budget) != 1L || !is.finite(budget) || budget <= 0 ||
        length(groups) < 2L || !is.finite(critical) || critical <= 0 ||
        any(!is.finite(distribution$location)) || any(!is.finite(probs)) ||
        any(!is.finite(values) | !is.finite(variance) | variance < 0)) return(FALSE)
    sigma <- sqrt(variance)
    root_bound <- 8 * .Machine$double.eps * max(1, abs(values) + 10 * sigma)
    radius <- budget * distribution$sd / (critical * sqrt(length(groups) - 1L)) - root_bound
    if (!is.finite(radius) || radius <= 0) return(FALSE)
    retained <- sum(weights) - vapply(groups, function(at) sum(weights[at]), numeric(1))
    if (any(retained <= 0)) return(FALSE)
    bounds <- as.vector(rbind(distribution$location - radius, distribution$location + radius))
    components <- matrix(vapply(bounds, function(x) stats::pnorm(x, values, sigma),
        numeric(length(values))), nrow = length(values))
    weighted <- weights * components
    total <- colSums(weighted)
    leave <- vapply(seq_along(groups), function(g)
        (total - colSums(weighted[groups[[g]], , drop = FALSE])) / retained[g],
        numeric(length(bounds)))
    gamma <- length(weights) * .Machine$double.eps /
        (1 - length(weights) * .Machine$double.eps)
    roundoff <- 8 * gamma / min(retained)
    lower <- seq.int(1L, length(bounds), by = 2L)
    # CDF brackets bound every deleted-group quantile. Popoviciu's variance
    # bound then places its jackknife error below the common budget.
    all(leave[lower, , drop = FALSE] + roundoff < probs) &&
        all(leave[lower + 1L, , drop = FALSE] - roundoff >= probs)
}

.nl_outer_read_error <- function(u, log_weight, replicate, proposed, fixed = NULL,
                                  probs = c(.025, .5, .975)) {
    weights <- .nl_normalise_weights_safe(log_weight)
    keep <- which(is.finite(weights) & weights > 0)
    u <- u[keep, , drop = FALSE]
    weights <- weights[keep]
    replicate <- replicate[keep]
    if (!is.null(fixed)) fixed <- lapply(fixed, function(x) x[keep, , drop = FALSE])
    d <- ncol(u)
    p <- if (is.null(fixed)) 0L else ncol(fixed$mu)
    if (!length(weights)) return(list(hyper_quantiles = matrix(Inf, length(probs), d),
        fixed_quantiles = if (p) matrix(Inf, length(probs), p),
        log_evidence = Inf, max = Inf, complete = TRUE,
        error_method = "simultaneous_delete_scramble_envelope"))
    context <- .nl_outer_error_context(weights, replicate, proposed, 1L + length(probs) * (d + p))
    error <- function(values, variance = NULL, budget = NULL) {
        distribution <- .nl_outer_quantile_summary(values, variance, weights, probs)
        if (is.null(distribution)) return(rep(Inf, length(probs)))
        if (isTRUE(distribution$sd == 0)) return(rep(0, length(probs)))
        if (!is.null(variance) && .nl_outer_jackknife_dominated(values, variance,
            weights, context$groups, distribution, budget, context$critical, probs))
            return(rep(budget, length(probs)))
        leave <- vapply(seq_along(context$groups), function(g)
            distribution$quantile(probs, which(replicate != g)), numeric(length(probs)))
        if (any(!is.finite(leave))) return(rep(Inf, length(probs)))
        context$critical * apply(leave, 1L, stats::sd) *
            (length(context$groups) - 1L) / sqrt(length(context$groups)) / distribution$sd
    }
    hyper <- vapply(seq_len(d), function(j) error(u[, j]), numeric(length(probs)))
    fixed_error <- if (p) vapply(seq_len(p), function(j)
        error(fixed$mu[, j], fixed$var[, j], max(hyper)), numeric(length(probs)))
    budget <- max(c(hyper, fixed_error))
    hyper[] <- budget
    if (p) fixed_error[] <- budget
    list(hyper_quantiles = hyper, fixed_quantiles = fixed_error,
         log_evidence = context$log_evidence, max = max(budget, context$log_evidence),
         complete = TRUE, error_method = "simultaneous_delete_scramble_envelope")
}

.nl_outer_quantile_distribution <- function(values, variance, weights, probs,
                                              groups, proposed, critical, normalizer) {
    summary <- .nl_outer_quantile_summary(values, variance, weights, probs)
    if (is.null(summary)) return(list(quantile = function(p) rep(NA_real_, length(p)),
        location = rep(NA_real_, length(probs)), sd = NA_real_, cdf_error = rep(Inf, length(probs))))
    quantile <- summary$quantile
    sd <- summary$sd
    location <- summary$location
    if (is.null(variance)) {
        order <- order(values)
        mass <- weights[order]
        cumulative <- cumsum(mass)
        label <- integer(length(values))
        for (g in seq_along(groups)) label[groups[[g]]] <- g
        label <- label[order]
        total <- squared <- numeric(length(values))
        for (g in seq_along(groups)) {
            part <- mass
            part[label != g] <- 0
            score <- cumsum(part) / proposed[g] - cumulative * normalizer$z[g]
            total <- total + score
            squared <- squared + score^2
        }
        distinct <- c(which(diff(values[order]) != 0), length(values))
        variance <- pmax(0, (squared - total^2 / length(groups)) / (length(groups) - 1L))
        uncertainty <- if (normalizer$relative >= 1) Inf else
            critical * sqrt(max(variance[distinct])) / sqrt(length(groups)) /
                normalizer$mean / (1 - normalizer$relative)
    } else {
        cdf <- function(q, p) stats::pnorm(q, values, sqrt(variance))
        uncertainty <- vapply(seq_along(probs), function(k) {
            distribution <- cdf(location[k], probs[k])
            if (any(!is.finite(distribution))) return(Inf)
            center <- sum(weights * distribution)
            score <- vapply(seq_along(groups), function(g) {
                at <- groups[[g]]
                sum(weights[at] * (distribution[at] - center)) / proposed[g]
            }, numeric(1))
            if (normalizer$relative >= 1) return(Inf)
            critical * stats::sd(score) / sqrt(length(groups)) /
                normalizer$mean / (1 - normalizer$relative)
        }, numeric(1))
    }
    list(quantile = quantile, location = location, sd = sd, cdf_error = uncertainty)
}

.nl_outer_quantile_band <- function(distribution, probability_error, probs) {
    if (isTRUE(distribution$sd == 0)) return(rep(0, length(probs)))
    error <- rep(Inf, length(probs))
    inside <- which(probs - probability_error > 0 & probs + probability_error < 1)
    if (length(inside)) {
        lower <- distribution$quantile(probs[inside] - probability_error)
        upper <- distribution$quantile(probs[inside] + probability_error)
        error[inside] <- pmax(abs(lower - distribution$location[inside]),
                             abs(upper - distribution$location[inside])) / distribution$sd
    }
    error
}

# Scramble residuals estimate CDF and normalizer uncertainty. Inverting their
# CDF bands retains the nonlinear tail scale of each reported quantile.
.nl_qmc_error <- function(u, log_weight, replicate, out, proposed_per_rep,
                          probs = c(.025, .5, .975), tol = Inf, family_size = NULL) {
    n <- nrow(u)
    # A common normalization cancels in sd(z) / mean(z).
    a <- .nl_normalise_weights_safe(log_weight)
    keep <- which(is.finite(a) & a > 0)
    if (!length(keep)) return(list(log_evidence = Inf,
        hyper_quantiles = matrix(Inf, length(probs), ncol(u)),
        fixed_quantiles = NULL, max = Inf, complete = TRUE))
    if (length(keep) != n) {
        u <- u[keep, , drop = FALSE]
        a <- a[keep]
        replicate <- replicate[keep]
        for (field in c("grid_modes", "grid_hessians")) {
            if (length(out[[field]]) == n) out[[field]] <- out[[field]][keep]
        }
    }
    n_fixed <- if (length(out$grid_modes)) max(lengths(out$grid_modes)) else 0L
    family <- family_size %||% (1L + length(probs) * (ncol(u) + n_fixed))
    normalizer <- .nl_outer_error_context(a, replicate, proposed_per_rep, family)
    groups <- normalizer$groups
    critical <- normalizer$critical
    log_se <- normalizer$log_evidence
    w <- a
    distribution <- function(values, variance = NULL)
        .nl_outer_quantile_distribution(values, variance, w, probs, groups,
            proposed_per_rep, critical, normalizer)
    hyper_distribution <- lapply(seq_len(ncol(u)), function(j) distribution(u[, j]))
    probability_error <- max(vapply(hyper_distribution, function(x) max(x$cdf_error), numeric(1)))
    band <- function(x) .nl_outer_quantile_band(x, max(x$cdf_error), probs)
    hyper <- vapply(hyper_distribution, band, numeric(length(probs)))
    if (n_fixed && is.finite(tol) && (any(!is.finite(c(log_se, hyper))) || max(log_se, hyper) > tol)) {
        return(list(log_evidence = log_se, hyper_quantiles = hyper,
            fixed_quantiles = matrix(Inf, length(probs), n_fixed), max = Inf, complete = FALSE))
    }
    fixed <- NULL
    if (n_fixed) {
        variance <- lapply(out$grid_hessians, function(h)
            if (length(h)) diag(.spd_inverse(h)) else rep(NA_real_, n_fixed))
        fixed <- list(mu = t(matrix(vapply(out$grid_modes, function(m)
            if (length(m) == n_fixed) m else rep(NA_real_, n_fixed), numeric(n_fixed)), nrow = n_fixed)),
            var = t(matrix(vapply(variance, function(v) v, numeric(n_fixed)), nrow = n_fixed)))
    }
    fixed_error <- NULL
    if (n_fixed) {
        fixed_distribution <- lapply(seq_len(n_fixed), function(j)
            distribution(fixed$mu[, j], fixed$var[, j]))
        probability_error <- max(probability_error,
            vapply(fixed_distribution, function(x) max(x$cdf_error), numeric(1)))
        hyper <- vapply(hyper_distribution, band, numeric(length(probs)))
        fixed_error <- vapply(fixed_distribution, band, numeric(length(probs)))
    }
    errors <- c(log_se, hyper, fixed_error)
    list(log_evidence = log_se, hyper_quantiles = hyper, fixed_quantiles = fixed_error,
         cdf = probability_error,
         max = if (all(is.finite(errors))) max(errors) else Inf, complete = TRUE)
}

.nl_qmc_integrate <- function(res, solve, tags, n = .NL_QMC$nodes,
                               bounds = NULL, view = NULL, tol = .NL_QMC$tol) {
    tg <- .nl_theta_matrix(res)
    if (nrow(tg) == 1L) return(res)
    if (is.null(view)) view <- .hp_design_view(tg,
        .hp_grid_designs(tg, res$refining_axis)$designs)
    mapped <- length(view$cols) > 0L
    if (mapped) {
        tg <- view$grid
        native_solve <- solve
        solve <- function(cells, retain) native_solve(view$to_columns(cells), retain)
    }
    tags <- stats::setNames(tags, colnames(tg))
    if (length(tags) != ncol(tg) || anyNA(tags)) {
        stop("Importance quadrature requires a transform for every outer axis.", call. = FALSE)
    }
    for (j in seq_len(ncol(tg))) {
        if (isTRUE(.hyper_axis_scale(.hyper_axis_bare(colnames(tg)[j]))) &&
            any(tg[, j] == 0)) {
            stop("Importance quadrature requires continuous axes; `",
                 colnames(tg)[j], "` declares a point mass at zero.", call. = FALSE)
        }
    }
    U <- vapply(seq_len(ncol(tg)), function(j)
        .joint_pareto_fwd(tags[j], tg[, j]), numeric(nrow(tg)))
    keep <- res$weights > 0 & apply(is.finite(U), 1L, all)
    varying <- which(apply(U, 2L, function(v) diff(range(v))) > 0)
    if (!length(varying)) return(res)
    proposal <- .nl_qmc_proposal(U[keep, varying, drop = FALSE], res$weights[keep])
    stages <- rep(min(n, .NL_QMC$pilot_nodes), .NL_QMC$adapt_rounds)
    n_evaluated <- 0L
    for (stage in seq_along(stages)) {
        size <- stages[stage]
        nodes <- .nl_qmc_nodes(proposal, size)
        u <- matrix(rep(U[1L, ], each = size), size, ncol(tg))
        u[, varying] <- nodes$u
        valid <- rep(TRUE, size)
        if (!is.null(bounds)) for (j in varying) {
            valid <- valid & u[, j] >= bounds[1L, j] & u[, j] <= bounds[2L, j]
        }
        u <- u[valid, , drop = FALSE]
        cells <- u
        jac <- numeric(nrow(u))
        for (j in seq_len(ncol(tg))) {
            inv <- .joint_pareto_inv(tags[j], u[, j])
            cells[, j] <- inv$theta
            if (j %in% varying) jac <- jac + inv$log_jac
        }
        colnames(cells) <- colnames(tg)
        n_evaluated <- n_evaluated + nrow(cells)
        out <- .nl_outer_solve(cells, solve, retain = FALSE)
        measure <- jac - nodes$log_q[valid] - log(size)
        w <- .nl_normalise_weights_safe(out$log_marginal, log_quad = measure)
        good <- is.finite(w) & w > 0
        proposal <- .nl_qmc_proposal(u[good, varying, drop = FALSE], w[good])
    }
    out <- NULL
    u <- cells <- matrix(numeric(0), 0L, ncol(tg))
    log_weight <- measure <- numeric(0)
    replicate <- integer(0)
    proposed <- 0L
    r <- min(.NL_QMC$replicates, n)
    proposed_per_rep <- numeric(r)
    proposal <- .nl_qmc_nodes(proposal, .NL_QMC$block_nodes %/% r)$proposal
    history <- list()
    repeat {
        size <- min(.NL_QMC$block_nodes, n - proposed)
        counts <- diff(floor(seq(0, size, length.out = r + 1L)))
        parts <- lapply(seq_len(r), function(g) .nl_qmc_nodes(proposal, counts[g],
            start = proposed_per_rep[g], seed = 104729L + g))
        nodes <- list(u = do.call(rbind, lapply(parts, `[[`, "u")),
                      log_q = unlist(lapply(parts, `[[`, "log_q")), proposal = proposal)
        ub <- matrix(rep(U[1L, ], each = size), size, ncol(tg))
        ub[, varying] <- nodes$u
        valid <- rep(TRUE, size)
        if (!is.null(bounds)) for (j in varying)
            valid <- valid & ub[, j] >= bounds[1L, j] & ub[, j] <= bounds[2L, j]
        cb <- ub
        jac <- numeric(size)
        for (j in seq_len(ncol(tg))) {
            inv <- .joint_pareto_inv(tags[j], ub[, j])
            cb[, j] <- inv$theta
            if (j %in% varying) jac <- jac + inv$log_jac
        }
        colnames(cb) <- colnames(tg)
        rb <- .nl_outer_solve(cb[valid, , drop = FALSE], solve, retain = TRUE)
        # Rejected bound points carry zero mass in the integral's denominator.
        mb <- jac[valid] - nodes$log_q[valid]
        lb <- rb$log_marginal + mb
        if (is.null(out)) out <- rb else out <- .nl_bind_cell_results(out,
            list(list(res = rb, n = sum(valid))), nrow(cells))
        cells <- rbind(cells, cb[valid, , drop = FALSE])
        u <- rbind(u, ub[valid, , drop = FALSE])
        measure <- c(measure, mb)
        log_weight <- c(log_weight, lb)
        replicate <- c(replicate, rep(seq_len(r), counts)[valid])
        proposed <- proposed + size
        proposed_per_rep <- proposed_per_rep + counts
        n_evaluated <- n_evaluated + sum(valid)
        error <- .nl_qmc_error(u, log_weight, replicate, out, proposed_per_rep,
            tol = if (proposed >= n) Inf else tol)
        history[[length(history) + 1L]] <- c(n = proposed, error = error$max,
            hyper_error = max(error$hyper_quantiles), complete = error$complete)
        if (error$max <= tol || proposed >= n) break
    }
    measure <- measure - log(proposed)
    out$outer_qmc <- list(n = proposed, n_evaluated = n_evaluated, proposal = proposal,
                         error = error, tol = tol, converged = error$max <= tol,
                         error_method = "simultaneous_cdf_envelope",
                         error_confidence = .95,
                         error_scope = "simultaneous",
                         history = do.call(rbind, history))
    if (mapped) out$outer_qmc$design_columns <- view$cols
    .nl_outer_finish(out, res, cells, measure, "qmc", view)
}
