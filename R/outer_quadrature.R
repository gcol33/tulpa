.nl_outer_solve <- function(cells, solve, retain) {
    n <- nrow(cells)
    if (n <= .NL_OUTER_CHUNK_NODES) return(solve(cells, retain))
    first <- solve(cells[1L, , drop = FALSE], retain)
    ix <- unname(split(seq.int(2L, n), (seq_len(n - 1L) - 1L) %/% .NL_OUTER_CHUNK_NODES))
    chunks <- lapply(ix, function(i)
        list(res = solve(cells[i, , drop = FALSE], retain), n = length(i)))
    out <- .nl_bind_cell_results(first, chunks, 1L)
    if (!is.null(out$n_grid)) out$n_grid <- n
    out
}

.nl_validate_outer_design <- function(design, columns = NULL) {
    if (!is.list(design) || !identical(sort(names(design)), c("points", "weights"))) {
        stop("`control$outer_design` must be list(points =, weights =).", call. = FALSE)
    }
    points <- design$points
    weights <- design$weights
    if (!is.matrix(points) || !is.numeric(points) || !nrow(points) ||
        !ncol(points) || any(!is.finite(points))) {
        stop("`control$outer_design$points` must be a nonempty finite numeric matrix.",
             call. = FALSE)
    }
    if (is.null(colnames(points)) || anyNA(colnames(points)) ||
        any(!nzchar(colnames(points))) || anyDuplicated(colnames(points))) {
        stop("`control$outer_design$points` must have unique outer-axis column names.",
             call. = FALSE)
    }
    if (!is.numeric(weights) || length(weights) != nrow(points) ||
        any(!is.finite(weights)) || any(weights < 0) || !any(weights > 0)) {
        stop("`control$outer_design$weights` must have one finite nonnegative weight ",
             "per point and positive total mass.", call. = FALSE)
    }
    if (!is.null(columns)) {
        if (!setequal(colnames(points), columns)) {
            stop("`control$outer_design$points` columns must match the outer axes: ",
                 paste(columns, collapse = ", "), ".", call. = FALSE)
        }
        points <- points[, columns, drop = FALSE]
    }
    list(points = points, weights = weights)
}

.nl_outer_jackknife <- function(q, critical) {
    critical * apply(q, 1L, stats::sd) * (ncol(q) - 1L) / sqrt(ncol(q))
}

.nl_outer_geometric_error <- function(change, previous) {
    abs(change) / (1 - abs(change / previous))
}

.nl_outer_side_map <- function(radii) {
    scale <- rowSums(radii) / 4
    smoothing <- pmin(1, 2 * apply(radii, 1L, min) / (rowSums(radii) * log(2)))
    log_cosh <- function(x) abs(x) + log1p(exp(-2 * abs(x))) - log(2)
    skew <- (radii[, 2L] - radii[, 1L]) /
        (2 * smoothing * log_cosh(2 / smoothing))
    forward <- function(x) sweep(x, 2L, scale, "*") +
        sweep(log_cosh(sweep(x, 2L, smoothing, "/")), 2L, skew * smoothing, "*")
    derivative <- function(x) sweep(tanh(sweep(x, 2L, smoothing, "/")),
        2L, skew, "*") + rep(scale, each = nrow(x))
    inverse <- function(y) {
        lower <- pmin(sweep(y, 2L, scale - abs(skew), "/"),
                      sweep(y, 2L, scale + abs(skew), "/"))
        upper <- pmax(sweep(y, 2L, scale - abs(skew), "/"),
                      sweep(y, 2L, scale + abs(skew), "/"))
        x <- sweep(y, 2L, scale, "/")
        for (iteration in seq_len(60L)) {
            residual <- forward(x) - y
            if (max(abs(residual)) <= 1e-12 * max(1, abs(y))) break
            lower[residual < 0] <- x[residual < 0]
            upper[residual > 0] <- x[residual > 0]
            next_x <- x - residual / derivative(x)
            outside <- next_x <= lower | next_x >= upper
            next_x[outside] <- (lower[outside] + upper[outside]) / 2
            x <- next_x
        }
        x
    }
    list(forward = forward, inverse = inverse,
         log_derivative = function(x) rowSums(log(derivative(x))),
         radii = radii, scale = scale, skew = skew, smoothing = smoothing)
}

.nl_outer_mode_transport <- function(center, L, log_density) {
    value <- log_density(matrix(center, nrow = 1L))
    radii <- matrix(0, length(center), 2L)
    for (j in seq_along(center)) for (side in seq_len(2L)) {
        direction <- L[, j] * c(-1, 1)[side]
        radii[j, side] <- stats::uniroot(function(distance)
            value - log_density(matrix(center + direction * distance, nrow = 1L)) - 2,
            c(0, 2), extendInt = "upX", tol = 1e-4)$root
    }
    .nl_outer_side_map(radii)
}

.nl_outer_quantile_error <- function(change, previous, noise = 0, previous_noise = 0) {
    error <- abs(change)
    if (is.null(previous)) return(error)
    contracting <- sign(change) == sign(previous) & abs(change) < abs(previous) &
        abs(change) > noise & abs(previous) > previous_noise
    error[contracting] <- .nl_outer_geometric_error(change[contracting], previous[contracting])
    error
}

.nl_outer_evidence_error <- function(log_evidence) {
    changes <- diff(log_evidence)
    if (length(changes) < 2L || any(!is.finite(changes))) return(Inf)
    magnitude <- abs(changes)
    recent <- utils::tail(magnitude, 2L)
    roundoff <- .Machine$double.eps * max(1, abs(utils::tail(log_evidence, 1L)))
    if (recent[2L] == 0) return(max(recent, roundoff))
    contracting <- sign(utils::tail(changes, 1L)) == sign(changes[length(changes) - 1L]) &&
        recent[2L] < recent[1L]
    if (!contracting) return(max(magnitude, roundoff))
    remainder <- .nl_outer_geometric_error(recent[2L], recent[1L])
    max(recent, remainder, roundoff)
}

.nl_log_softplus <- function(x) {
    out <- x
    middle <- which(x >= log(.Machine$double.eps) & x <= 0)
    positive <- which(x > 0)
    out[middle] <- log(log1p(exp(x[middle])))
    out[positive] <- log(x[positive] + log1p(exp(-x[positive])))
    out
}

.tulpa_outer_integration_state <- function(fit) {
    for (object in list(fit, fit[["joint_fit"]], fit[["nested"]])) {
        state <- object[["outer_quadrature"]] %||% object[["outer_qmc"]]
        if (!is.null(state)) return(state)
    }
    NULL
}

.tulpa_outer_integration_note <- function(state) {
    if (is.null(state)) return(NULL)
    if (identical(state$method, "user")) {
        return(sprintf("User quadrature: %d supplied points.", state$n))
    }
    attained <- if (isTRUE(state$converged)) "target attained" else "target unattained"
    if (identical(state$method, "aghq_log_spline")) {
        return(sprintf(paste0("Adaptive quadrature: estimated error %.3g SD and %.3g nats; ",
            "targets %.3g SD and %.3g nats, %d solves (%s)."),
            state$error$max, state$error$log_evidence, state$tol,
            state$evidence_tol, state$n_evaluated, attained))
    }
    sprintf("Importance quadrature: estimated error %.3g, target %.3g, %d points (%s).",
        state$error$max, state$tol, state$n, attained)
}

.nl_outer_finish <- function(out, res, cells, measure, integration, view = NULL) {
    if (!is.null(out$n_grid)) out$n_grid <- nrow(cells)
    mapped <- !is.null(view) && length(view$cols) > 0L
    out$theta_grid <- if (mapped) view$to_columns(cells) else cells
    out$theta_names <- colnames(cells)
    out$integration <- integration
    out$outer_log_measure <- measure
    out$refining_axis <- rep(integration, nrow(cells))
    out$weight_kind <- rep("importance", nrow(cells))
    out$theta_interval_read <- "sample"
    fields <- c("outer_grid_placement", "outer_grid_axis_declined", "outer_grid_moved_axes",
                "outer_grid_recenter_clamped", "outer_grid_recenter_declined",
                "outer_mode_u", "outer_mode_cov_u", "outer_mode_axis_tags",
                "outer_mode_axis_names", "outer_mode_status", "outer_mode_rounds",
                "outer_mode_evals")
    out[fields] <- res[fields]
    .nl_attach_outer_integration(out, "outer quadrature")
}

.nl_validate_outer_design_axes <- function(design, tg, tags) {
    design <- .nl_validate_outer_design(design, colnames(tg))
    cells <- design$points
    for (j in seq_len(ncol(cells))) {
        domain <- unname(.JOINT_AXIS_DOMAIN[tags[j]])
        if (length(domain) == 1L && !is.na(domain) &&
            any(!.NL_DOMAIN_TRANSFORM[[domain]]$in_domain(cells[, j]))) {
            stop("`control$outer_design$points` must lie in the ", domain,
                 " domain on axis `", colnames(cells)[j], "`.", call. = FALSE)
        }
        if (length(unique(tg[, j])) == 1L && any(cells[, j] != tg[1L, j])) {
            stop("`control$outer_design$points` changes fixed axis `",
                 colnames(cells)[j], "`.", call. = FALSE)
        }
    }
    design
}

.nl_validate_outer_design_prior <- function(design, prior) {
    grids <- lapply(prior, function(p) .nl_block_axis_grid(p)$grid)
    axes <- unlist(lapply(seq_along(grids), function(b)
        paste0("b", b, ".", colnames(grids[[b]]))))
    tg <- do.call(cbind, lapply(grids, function(g)
        g[rep(seq_len(nrow(g)), length.out = max(vapply(grids, nrow, integer(1)))), ,
          drop = FALSE]))
    colnames(tg) <- axes
    res <- list(theta_grid = tg, axis_offsets = c(0L, cumsum(vapply(grids, ncol, integer(1)))))
    tags <- .nl_registry_axis_tags(res, prior, TRUE)
    .nl_validate_outer_design_axes(design, tg, tags)
}

.nl_user_integrate <- function(res, solve, design, tags) {
    design <- .nl_validate_outer_design_axes(design, .nl_theta_matrix(res), tags)
    cells <- design$points
    out <- .nl_outer_solve(cells, solve, retain = TRUE)
    out$outer_quadrature <- list(method = "user", n = nrow(cells),
        n_evaluated = nrow(cells), weights = design$weights)
    .nl_outer_finish(out, res, cells, log(design$weights), "user")
}
