.nl_aghq_blocks <- function(prior) {
    if (!.is_multi_block_prior(prior) || length(prior) < 2L || !all(vapply(prior, function(p)
        tolower(p$type %||% "") %in% c("hsgp", "iid"), logical(1)))) return(NULL)
    names <- vapply(prior, function(p) p$name %||% "", character(1))
    if (!all(c("local", "regional") %in% names) ||
        !all(vapply(prior[match(c("local", "regional"), names)],
            function(p) identical(tolower(p$type), "hsgp"), logical(1)))) return(NULL)
    grids <- lapply(prior, function(p) .nl_block_axis_grid(p)$grid)
    free <- all(vapply(grids, function(grid) {
        ncol(grid) > 0L && all(grid > 0) &&
            all(apply(grid, 2L, function(x) length(unique(x)) > 1L))
    }, logical(1)))
    if (!free) return(NULL)
    list(local = match("local", names), regional = match("regional", names), grids = grids)
}

.nl_aghq_supported <- function(prior, control) {
    blocks <- .nl_aghq_blocks(prior)
    if (is.null(blocks)) return(FALSE)
    provenance <- .nl_grid_provenance(prior)
    axes <- unlist(lapply(seq_along(prior), function(b)
        paste0("b", b, ".", colnames(blocks$grids[[b]]))))
    modes <- .nl_registry_refine_modes(axes, axes, provenance$prior, TRUE,
        provenance$auto, user = control$axis_refine)
    all(modes == "extend")
}

.nl_aghq_config <- function(control) {
    cfg <- list(max_level = control$quad_max_level %||% 5L,
        progress = isTRUE(control$progress),
        read_nodes = control$quad_read_nodes %||% 65536L,
        read_df = 7,
        tol = control$quad_tol %||% .05,
        evidence_tol = control$quad_evidence_tol %||% .005)
    for (name in c("max_level", "read_nodes")) {
        value <- cfg[[name]]
        if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
            value != floor(value) || value > .Machine$integer.max ||
            value < if (name == "read_nodes") 128L else 3L) {
            stop("`control$quad_", name, "` must be a finite integer >= ",
                 if (name == "read_nodes") 128L else 3L, ".", call. = FALSE)
        }
    }
    if (cfg$max_level > 5L || cfg$read_nodes %% 8 != 0L) {
        stop("Quadrature max_level must be at most five, ",
             "and read_nodes must be divisible by eight.", call. = FALSE)
    }
    for (name in c("tol", "evidence_tol")) {
        value <- cfg[[name]]
        if (!is.numeric(value) || length(value) != 1L || !is.finite(value) || value <= 0) {
            stop("`control$quad_", name, "` must be positive and finite.", call. = FALSE)
        }
    }
    cfg
}

.nl_gp_regional_transport <- function(points, weights, prior) {
    scale <- .hp_scale_anchor(prior)
    range <- .hp_range_anchor(prior)$anchor
    rate <- -log(scale[2L]) / scale[1L]
    range_rate <- -log(range[2L]) * range[1L] / 2
    median <- vapply(seq_len(2L), function(j)
        .nl_wtd_quantile(log(points[, j]), weights, .5, outside = "clamp"), numeric(1))
    log_anchor <- c(log(rate) + median[1L] / 2, log(range_rate) - median[2L])
    anchor <- exp(log_anchor)
    log_cdf <- log_anchor
    on <- log_anchor >= log(.Machine$double.eps)
    log_cdf[on] <- .log1mexp(-anchor[on])
    log_stretch <- -anchor - log_cdf
    warp <- function(v) {
        log_v <- log(v)
        log_survival <- log1p(-v)
        odds <- sweep(log_v - log_survival, 2L, log_stretch, "-")
        list(log_value = .nl_log_softplus(odds),
             log_denominator = .logspace_add(
                 sweep(log_survival, 2L, log_stretch, "+"), log_v))
    }
    physical <- function(v) {
        w <- warp(v)
        cbind(exp(2 * (w$log_value[, 1L] - log(rate))),
              exp(log(range_rate) - w$log_value[, 2L]))
    }
    log_density <- function(v) {
        w <- warp(v)
        rowSums(w$log_value + w$log_denominator + log1p(-v)) - log(2)
    }
    list(physical = physical, log_density = log_density, log_stretch = log_stretch)
}

.nl_aghq_transform <- function(res, prior, solve, read_df) {
    blocks <- .nl_aghq_blocks(prior)
    if (is.null(blocks)) {
        stop("Quadrature requires local and regional HSGP blocks with four free axes.",
             call. = FALSE)
    }
    local <- blocks$local
    regional <- blocks$regional
    columns <- lapply(c(local, regional), function(b)
        (res$axis_offsets[b] + 1L):res$axis_offsets[b + 1L])
    mode <- .nl_fit_outer_mode(res)
    evaluations <- 0L
    if (is.null(mode)) {
        mode <- .nl_placement_mode(res$theta_grid, res$weights,
            rep("log", ncol(res$theta_grid)), function(cells) {
                evaluations <<- evaluations + nrow(cells)
                solve(cells, FALSE)$log_marginal
            })
    }
    li <- columns[[1L]]
    ri <- columns[[2L]]
    extra <- setdiff(seq_len(ncol(res$theta_grid)), c(li, ri))
    extra_center <- mode$mode_u[extra]
    extra_L <- if (length(extra)) t(chol(mode$cov_u[extra, extra, drop = FALSE]))
    center <- mode$mode_u[li]
    L <- t(chol(mode$cov_u[li, li, drop = FALSE]))
    gaussian_axes <- c(li, extra)
    gaussian_columns <- seq.int(3L, ncol(res$theta_grid))
    initial_L <- matrix(0, length(gaussian_axes), length(gaussian_axes))
    initial_L[1:2, 1:2] <- L
    if (length(extra)) initial_L[-(1:2), -(1:2)] <- extra_L
    grid_u <- log(res$theta_grid[, gaussian_axes, drop = FALSE])
    grid_w <- res$weights / sum(res$weights)
    read_center <- as.numeric(crossprod(grid_w, grid_u))
    delta <- sweep(grid_u, 2L, read_center)
    read_L <- t(chol(crossprod(delta, grid_w * delta) + tcrossprod(initial_L)))
    region <- .nl_gp_regional_transport(res$theta_grid[, ri, drop = FALSE],
                                      grid_w, prior[[regional]])
    mode_rule <- .nl_gauss_legendre_unit(5L)
    mode_design <- as.matrix(expand.grid(mode_rule$x, mode_rule$x, 0, 0))
    regional_points <- region$physical(mode_design[, 1:2, drop = FALSE])
    regional_measure <- rowSums(log(as.matrix(expand.grid(mode_rule$w, mode_rule$w)))) -
        region$log_density(mode_design[, 1:2, drop = FALSE])
    marginal_density <- new.env(hash = TRUE, parent = emptyenv())
    local_log_density <- function(local_points) {
        keys <- .nl_outer_point_key(local_points)
        missing <- !vapply(keys, exists, logical(1), envir = marginal_density, inherits = FALSE)
        fresh_keys <- unique(keys[missing])
        if (length(fresh_keys)) {
            local_points <- local_points[match(fresh_keys, keys), , drop = FALSE]
            count <- nrow(regional_points)
            cells <- matrix(0, nrow(local_points) * count, ncol(res$theta_grid),
                dimnames = list(NULL, colnames(res$theta_grid)))
            cells[, li] <- local_points[rep(seq_len(nrow(local_points)), each = count), , drop = FALSE]
            cells[, ri] <- regional_points[rep(seq_len(count), nrow(local_points)), , drop = FALSE]
            if (length(extra)) cells[, extra] <- rep(exp(extra_center), each = nrow(cells))
            evaluations <<- evaluations + nrow(cells)
            lp <- .nl_outer_solve(cells, solve, FALSE)$log_marginal
            scores <- matrix(lp, nrow = count) + regional_measure
            values <- vapply(seq_len(ncol(scores)), function(j)
                .tulpa_logsumexp(scores[, j]), numeric(1))
            for (j in seq_along(fresh_keys)) marginal_density[[fresh_keys[j]]] <- values[j]
        }
        vapply(keys, function(key) marginal_density[[key]], numeric(1), USE.NAMES = FALSE)
        }
    mode <- .nl_placement_mode(res$theta_grid[, li, drop = FALSE], grid_w,
        rep("log", 2L), local_log_density)
    center <- mode$mode_u
    L <- t(chol(mode$cov_u))
    sides <- .nl_outer_mode_transport(center, L, function(u) local_log_density(exp(u)))
    read <- function(query) {
        standard <- query[, gaussian_columns, drop = FALSE]
        shifted <- sweep(standard %*% t(read_L), 2L,
                         read_center - c(center, extra_center), "+")
        query[, 3:4] <- sides$inverse(t(forwardsolve(L, t(shifted[, 1:2, drop = FALSE]))))
        correction <- -log(2 * pi) - rowSums(query[, 3:4, drop = FALSE]^2) / 2 -
            rowSums(stats::dt(standard[, 1:2, drop = FALSE], read_df, log = TRUE)) +
            sum(log(diag(read_L))) - sum(log(diag(L))) -
            sides$log_derivative(query[, 3:4, drop = FALSE])
        if (length(extra)) {
            columns <- 4L + seq_along(extra)
            query[, columns] <- t(forwardsolve(extra_L, t(shifted[, -(1:2), drop = FALSE])))
            correction <- correction + rowSums(
                stats::dnorm(query[, columns, drop = FALSE], log = TRUE) -
                stats::dt(standard[, -(1:2), drop = FALSE], read_df, log = TRUE)) -
                sum(log(diag(extra_L)))
        }
        list(query = query, correction = correction)
    }
    physical <- function(v) {
        cells <- matrix(0, nrow(v), ncol(res$theta_grid),
            dimnames = list(NULL, colnames(res$theta_grid)))
        cells[, li] <- exp(sweep(sides$forward(v[, 3:4, drop = FALSE]) %*% t(L),
                                2L, center, "+"))
        cells[, ri] <- region$physical(v[, 1:2, drop = FALSE])
        if (length(extra)) cells[, extra] <- exp(sweep(
            v[, 4L + seq_along(extra), drop = FALSE] %*% t(extra_L), 2L, extra_center, "+"))
        cells
    }
    log_density <- function(v) {
        value <- -log(2 * pi) - sum(log(diag(L))) - rowSums(v[, 3:4, drop = FALSE]^2) / 2 +
            region$log_density(v[, 1:2, drop = FALSE]) -
            sides$log_derivative(v[, 3:4, drop = FALSE])
        if (length(extra)) value <- value + rowSums(stats::dnorm(
            v[, 4L + seq_along(extra), drop = FALSE], log = TRUE)) - sum(log(diag(extra_L)))
        value
    }
    adapt_read <- function(points, weights) {
        weights <- weights / sum(weights)
        values <- log(points[, gaussian_axes, drop = FALSE])
        read_center <<- as.numeric(crossprod(weights, values))
        delta <- sweep(values, 2L, read_center)
        covariance <- crossprod(delta, weights * delta)
        read_L <<- t(chol(covariance * (read_df - 2) / read_df))
        list(center = read_center, L = read_L)
    }
    list(physical = physical, log_density = log_density,
         read = read, adapt_read = adapt_read, center = center, L = L,
         regional_log_stretch = region$log_stretch,
         read_center = read_center, read_L = read_L,
         local_radii = sides$radii,
         evaluations = evaluations)
}
