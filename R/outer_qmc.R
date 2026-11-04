# Importance quadrature in transformed hyperparameters. A Student mixture has
# positive density throughout the coordinate space, including variance arms
# whose mass is small on the placement grid.
.nl_qmc_uniform <- function(n, d) {
    primes <- integer(0)
    p <- 2L
    while (length(primes) < d) {
        if (!any(p %% primes[primes <= sqrt(p)] == 0L)) primes <- c(primes, p)
        p <- p + 1L
    }
    out <- matrix(0, n, d)
    for (j in seq_len(d)) {
        k <- seq_len(n)
        f <- 1 / primes[j]
        while (any(k > 0L)) {
            out[, j] <- out[, j] + f * (k %% primes[j])
            k <- k %/% primes[j]
            f <- f / primes[j]
        }
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

.nl_qmc_nodes <- function(proposal, n) {
    d <- length(proposal[[1L]]$mu)
    mass <- vapply(proposal, `[[`, numeric(1), "mass")
    counts <- diff(c(0L, round(cumsum(mass) * n)))
    id <- rep(seq_along(proposal), counts)
    for (g in seq_along(proposal)) proposal[[g]]$mass <- counts[g] / n
    U <- matrix(0, n, d)
    for (g in seq_along(proposal)) {
        ix <- which(id == g)
        if (!length(ix)) next
        p <- proposal[[g]]
        v <- .nl_qmc_uniform(length(ix), d)
        v <- sweep(v, 2L, (g * sqrt(seq_len(d) + 1L)) %% 1, "+") %% 1
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
                               bounds = NULL, view = NULL) {
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
    stages <- c(rep(min(n, .NL_QMC$pilot_nodes), .NL_QMC$adapt_rounds), n)
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
        final <- stage == length(stages)
        out <- .nl_qmc_solve(cells, solve, retain = final)
        measure <- jac - nodes$log_q[valid] - log(size)
        w <- .nl_normalise_weights_safe(out$log_marginal, log_quad = measure)
        if (!final) {
            good <- is.finite(w) & w > 0
            proposal <- .nl_qmc_proposal(u[good, varying, drop = FALSE], w[good])
        }
    }
    out$theta_grid <- if (mapped) view$to_columns(cells) else cells
    out$theta_names <- colnames(cells)
    out$integration <- "qmc"
    out$outer_log_measure <- measure
    out$refining_axis <- rep("qmc", nrow(cells))
    out$weight_kind <- rep("importance", nrow(cells))
    out$theta_interval_read <- "sample"
    out$outer_qmc <- list(n = n, n_evaluated = n_evaluated, proposal = nodes$proposal)
    if (mapped) out$outer_qmc$design_columns <- view$cols
    fields <- c("outer_grid_placement", "outer_grid_axis_declined", "outer_grid_moved_axes",
                "outer_grid_recenter_clamped", "outer_grid_recenter_declined")
    out[fields] <- res[fields]
    .nl_attach_outer_integration(out, "importance quadrature")
}
