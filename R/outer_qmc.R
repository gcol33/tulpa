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

# Independent scrambles estimate the error of a self-normalized integral.
# Quantile error deletes one scramble from the pooled posterior, including the
# complete conditional Gaussian mixture for each retained fixed effect. Student-t
# bounds at 95% account for the finite number of independent scrambles.
.nl_qmc_error <- function(u, log_weight, replicate, out, proposed_per_rep,
                          probs = c(.025, .5, .975), tol = Inf) {
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
    groups <- lapply(seq_along(proposed_per_rep), function(g) which(replicate == g))
    leave <- lapply(seq_along(groups), function(g) which(replicate != g))
    critical <- stats::qt(.975, length(groups) - 1L)
    jackknife <- function(q) {
        critical * apply(q, 1L, stats::sd) *
            (length(groups) - 1L) / sqrt(length(groups))
    }
    z <- vapply(groups, function(ix) sum(a[ix]), numeric(1)) / proposed_per_rep
    log_se <- critical * stats::sd(z) / sqrt(length(z)) / mean(z)
    w <- a
    qse <- function(values, variance = NULL) {
        if (is.null(variance)) {
            mu <- sum(w * values)
            sd <- sqrt(sum(w * (values - mu)^2))
            qs <- vapply(leave, function(ix) {
                if (!length(ix) || !any(a[ix] > 0)) return(rep(NA_real_, length(probs)))
                .nl_wtd_quantile(values[ix], a[ix], probs, outside = "clamp")
            }, numeric(length(probs)))
        } else {
            full <- .nl_gauss_mixture_summary(matrix(values, ncol = 1L),
                matrix(variance, ncol = 1L), w, probs)
            if (is.null(full)) return(rep(Inf, length(probs)))
            sd <- full$sd
            qs <- vapply(leave, function(ix) {
                s <- .nl_gauss_mixture_summary(matrix(values[ix], ncol = 1L),
                    matrix(variance[ix], ncol = 1L), a[ix], probs)
                if (is.null(s)) rep(NA_real_, length(probs)) else as.numeric(s$quantiles)
            }, numeric(length(probs)))
        }
        if (isTRUE(sd == 0)) return(rep(0, length(probs)))
        jackknife(qs) / sd
    }
    hyper <- vapply(seq_len(ncol(u)), function(j) qse(u[, j]), numeric(length(probs)))
    fixed <- NULL
    if (length(out$grid_modes) == nrow(u) && length(out$grid_hessians) == nrow(u)) {
        p <- max(lengths(out$grid_modes))
        if (p) {
            errors <- c(log_se, hyper)
            # A failed criterion rules out stopping before fixed-effect CDF inversion.
            if (is.finite(tol) && (any(!is.finite(errors)) || max(errors) > tol)) {
                return(list(log_evidence = log_se, hyper_quantiles = hyper,
                    fixed_quantiles = matrix(Inf, length(probs), p),
                    max = Inf, complete = FALSE))
            }
            variance <- lapply(out$grid_hessians, function(h)
                if (length(h)) diag(.spd_inverse(h)) else rep(NA_real_, p))
            fixed <- vapply(seq_len(p), function(j) {
                mu <- vapply(out$grid_modes, function(m)
                    if (length(m)) m[j] else NA_real_, numeric(1))
                vv <- vapply(variance, function(v) v[j], numeric(1))
                qse(mu, vv)
            }, numeric(length(probs)))
        }
    }
    errors <- c(log_se, hyper, fixed)
    list(log_evidence = log_se, hyper_quantiles = hyper, fixed_quantiles = fixed,
         max = if (all(is.finite(errors))) max(errors) else Inf, complete = TRUE)
}

.nl_qmc_solve <- function(cells, solve, retain) {
    n <- nrow(cells)
    if (n <= .NL_QMC$chunk_nodes) return(solve(cells, retain))
    first <- solve(cells[1L, , drop = FALSE], retain)
    ix <- unname(split(seq.int(2L, n), (seq_len(n - 1L) - 1L) %/% .NL_QMC$chunk_nodes))
    chunks <- lapply(ix, function(i)
        list(res = solve(cells[i, , drop = FALSE], retain), n = length(i)))
    out <- .nl_bind_cell_results(first, chunks, 1L)
    if (!is.null(out$n_grid)) out$n_grid <- n
    out
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
        out <- .nl_qmc_solve(cells, solve, retain = FALSE)
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
        rb <- .nl_qmc_solve(cb[valid, , drop = FALSE], solve, retain = TRUE)
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
    if (!is.null(out$n_grid)) out$n_grid <- nrow(cells)
    out$theta_grid <- if (mapped) view$to_columns(cells) else cells
    out$theta_names <- colnames(cells)
    out$integration <- "qmc"
    out$outer_log_measure <- measure
    out$refining_axis <- rep("qmc", nrow(cells))
    out$weight_kind <- rep("importance", nrow(cells))
    out$theta_interval_read <- "sample"
    out$outer_qmc <- list(n = proposed, n_evaluated = n_evaluated, proposal = proposal,
                         error = error, tol = tol, converged = error$max <= tol,
                         error_method = "delete_scramble_jackknife",
                         error_confidence = .95,
                         error_scope = "pointwise",
                         history = do.call(rbind, history))
    if (mapped) out$outer_qmc$design_columns <- view$cols
    fields <- c("outer_grid_placement", "outer_grid_axis_declined", "outer_grid_moved_axes",
                "outer_grid_recenter_clamped", "outer_grid_recenter_declined")
    out[fields] <- res[fields]
    .nl_attach_outer_integration(out, "importance quadrature")
}

.tulpa_qmc_state <- function(fit) {
    fit[["outer_qmc"]] %||% fit[["joint_fit"]][["outer_qmc"]] %||%
        fit[["nested"]][["outer_qmc"]]
}

.tulpa_qmc_note <- function(qmc) {
    if (is.null(qmc)) return(NULL)
    sprintf("Importance quadrature: estimated error %.3g, target %.3g, %d points (%s).",
        qmc$error$max, qmc$tol, qmc$n,
        if (isTRUE(qmc$converged)) "target attained" else "target unattained")
}
