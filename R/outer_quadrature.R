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
    out[.NL_PLACEMENT_FIELDS] <- res[.NL_PLACEMENT_FIELDS]
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

# The outer-design settings a nested-Laplace door reads from `control`: the
# caller's own design (`outer_design`), the importance quadrature's final node
# budget and error target (`qmc_nodes`, `qmc_tol`), and the adaptive sparse
# rule's configuration (`quad_*`, `.nl_aghq_config()`). Every value is checked
# here whatever the integration the fit ends up taking, so a misspecified
# setting is named where the caller wrote it.
.nl_outer_design_control <- function(control) {
    outer_design <- control$outer_design
    if (!is.null(outer_design)) outer_design <- .nl_validate_outer_design(outer_design)
    qmc_nodes <- control$qmc_nodes %||% .NL_QMC$nodes
    qmc_tol <- control$qmc_tol %||% .NL_QMC$tol
    if (!is.numeric(qmc_tol) || length(qmc_tol) != 1L ||
        !is.finite(qmc_tol) || qmc_tol <= 0) {
        stop("`control$qmc_tol` must be a positive finite number.", call. = FALSE)
    }
    if (!is.numeric(qmc_nodes) || length(qmc_nodes) != 1L || !is.finite(qmc_nodes) ||
        qmc_nodes < 2L || qmc_nodes > .Machine$integer.max ||
        qmc_nodes != floor(qmc_nodes)) {
        stop("`control$qmc_nodes` must be an integer >= 2.", call. = FALSE)
    }
    list(outer_design = outer_design, qmc_nodes = qmc_nodes, qmc_tol = qmc_tol,
         quad = .nl_aghq_config(control))
}

# The integrations that replace the outer grid by a design of their own.
.NL_OUTER_DESIGNS <- c("qmc", "user", "aghq")

# The integration a multi-block prior asks for: `requested` when the caller
# named one, `"user"` when the caller supplied an outer design, else `default`.
# An outer design and user integration go together, and the adaptive sparse
# rule needs a local and a regional HSGP block with four free axes.
.nl_outer_integration <- function(requested, cfg, prior, default, choices) {
    integration <- match.arg(requested %||%
        if (!is.null(cfg$outer_design)) "user" else default, choices)
    if (identical(integration, "aghq") && is.null(.nl_aghq_blocks(prior))) {
        stop("`control$integration = \"aghq\"` requires local and regional HSGP blocks ",
             "with four free axes.", call. = FALSE)
    }
    if (identical(integration, "user") != !is.null(cfg$outer_design)) {
        stop("`control$integration = \"user\"` requires `control$outer_design`; ",
             "an outer design is used only with user integration.", call. = FALSE)
    }
    integration
}

# Integrate a solved outer grid's hyperparameters on an outer design: the
# placed grid seeds the importance proposal (`"qmc"`), the caller's points are
# solved as given (`"user"`), or the adaptive sparse rule is laid around the
# grid's mode (`"aghq"`). `solve(cells, retain)` is the door's inner solve at
# arbitrary hyperparameter cells, its hyperprior folded into the log marginals,
# with the per-cell side data the fit keeps when `retain`. `tags` are the
# axes' transform tags and `extend` flags the axes whose proposal support is
# the whole real line rather than the span of their placed nodes.
.nl_outer_design_integrate <- function(res, integration, solve, tags, extend,
                                       prior, cfg) {
    tg <- res$theta_grid
    view <- .hp_design_view(tg, .hp_grid_designs(tg, res$refining_axis)$designs)
    switch(integration,
        qmc = .nl_qmc_integrate(res, solve, tags, n = cfg$qmc_nodes,
            bounds = vapply(seq_len(ncol(tg)), function(j) {
                if (isTRUE(extend[j])) c(-Inf, Inf)
                else range(.joint_pareto_fwd(tags[j], view$grid[, j]))
            }, numeric(2)),
            view = view, tol = cfg$qmc_tol),
        user = .nl_user_integrate(res, solve, cfg$outer_design, tags),
        aghq = .nl_aghq_integrate(res, solve, prior, cfg$quad))
}

# The number of cells an outer design solves in full, which sizes the per-cell
# side data a fit retains for them.
.nl_outer_design_cells <- function(integration, cfg, n_axes) {
    switch(integration, qmc = cfg$qmc_nodes,
        user = nrow(cfg$outer_design$points),
        aghq = prod(vapply(seq_len(n_axes), function(j)
            length(.nl_sparse_rules(j > 2L)[[cfg$quad$max_level]]$x), integer(1))))
}

# The outer design a joint fit integrates on, NULL for a grid or CCD. A design
# reads one multi-block prior's block axes: a copy block's scale and a per-arm
# dispersion axis are not among the axes a design is laid over, and the
# k_quality escalation refines a tensor grid a design does not keep.
.nl_joint_outer_design <- function(requested, cfg, prior, copy, phi_grid,
                                   k_quality = NULL) {
    integration <- .nl_outer_integration(
        requested, cfg, prior, "auto",
        c("auto", "ccd", "grid", "grid_adaptive", .NL_OUTER_DESIGNS))
    if (!integration %in% .NL_OUTER_DESIGNS) return(NULL)
    if (!.is_multi_block_prior(prior) || !is.null(copy) ||
        any(lengths(phi_grid %||% list()) > 1L)) {
        stop("`control$integration = \"", integration, "\"` integrates the block ",
             "axes of a multi-block `prior` without `copy` or a `phi_grid` axis.",
             call. = FALSE)
    }
    if (isTRUE(k_quality %in% c("ok", "good"))) {
        stop("`control$k_quality = \"", k_quality, "\"` refines a tensor grid; ",
             "an outer design (`control$integration = \"", integration,
             "\"`) does not keep one.", call. = FALSE)
    }
    integration
}

# The placement record a fit carries from the grid its axes were laid on to an
# integration laid over that grid.
.NL_PLACEMENT_FIELDS <- c(
    "outer_grid_placement", "outer_grid_axis_declined", "outer_grid_moved_axes",
    "outer_grid_recenter_clamped", "outer_grid_recenter_declined",
    "outer_mode_u", "outer_mode_cov_u", "outer_mode_axis_tags",
    "outer_mode_axis_names", "outer_mode_status", "outer_mode_rounds",
    "outer_mode_evals")
