.nl_aghq_blocks <- function(prior) {
    if (!.is_multi_block_prior(prior) || length(prior) != 2L || !all(vapply(prior, function(p)
        identical(tolower(p$type %||% ""), "hsgp"), logical(1)))) return(NULL)
    names <- vapply(prior, function(p) p$name %||% "", character(1))
    if (!setequal(names, c("local", "regional"))) return(NULL)
    grids <- lapply(prior, function(p) .nl_block_axis_grid(p)$grid)
    free <- all(vapply(grids, function(grid) {
        ncol(grid) == 2L && all(apply(grid, 2L, function(x) length(unique(x)) > 1L))
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
    cfg <- list(order = control$quad_order %||% 5L,
        max_order = control$quad_max_order %||% 17L,
        read_nodes = control$quad_read_nodes %||% 65536L,
        read_df = 7,
        tol = control$quad_tol %||% .05,
        evidence_tol = control$quad_evidence_tol %||% .005)
    for (name in c("order", "max_order", "read_nodes")) {
        value <- cfg[[name]]
        if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
            value != floor(value) || value > .Machine$integer.max ||
            value < if (name == "read_nodes") 128L else 5L) {
            stop("`control$quad_", name, "` must be a finite integer >= ",
                 if (name == "read_nodes") 128L else 5L, ".", call. = FALSE)
        }
    }
    if (cfg$order %% 2 != 1L || cfg$max_order %% 2 != 1L ||
        cfg$max_order < cfg$order + 2L || cfg$read_nodes %% 8 != 0L) {
        stop("Quadrature orders must be odd, max_order must exceed order, ",
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
    center <- mode$mode_u[li]
    L <- t(chol(mode$cov_u[li, li, drop = FALSE]))
    grid_u <- log(res$theta_grid[, li, drop = FALSE])
    grid_w <- res$weights / sum(res$weights)
    read_center <- as.numeric(crossprod(grid_w, grid_u))
    delta <- sweep(grid_u, 2L, read_center)
    read_L <- t(chol(crossprod(delta, grid_w * delta) + tcrossprod(L)))
    region <- .nl_gp_regional_transport(res$theta_grid[, ri, drop = FALSE],
                                      grid_w, prior[[regional]])
    mode_rule <- .nl_gauss_legendre_unit(5L)
    mode_design <- as.matrix(expand.grid(mode_rule$x, mode_rule$x, 0, 0))
    regional_points <- region$physical(mode_design[, 1:2, drop = FALSE])
    regional_measure <- rowSums(log(as.matrix(expand.grid(mode_rule$w, mode_rule$w)))) -
        region$log_density(mode_design[, 1:2, drop = FALSE])
    local_log_density <- function(local_points) {
            count <- nrow(regional_points)
            cells <- matrix(0, nrow(local_points) * count, ncol(res$theta_grid),
                dimnames = list(NULL, colnames(res$theta_grid)))
            cells[, li] <- local_points[rep(seq_len(nrow(local_points)), each = count), , drop = FALSE]
            cells[, ri] <- regional_points[rep(seq_len(count), nrow(local_points)), , drop = FALSE]
            evaluations <<- evaluations + nrow(cells)
            lp <- .nl_outer_solve(cells, solve, FALSE)$log_marginal
            scores <- matrix(lp, nrow = count) + regional_measure
            vapply(seq_len(ncol(scores)), function(j)
                .tulpa_logsumexp(scores[, j]), numeric(1))
        }
    mode <- .nl_placement_mode(res$theta_grid[, li, drop = FALSE], grid_w,
        rep("log", 2L), local_log_density)
    center <- mode$mode_u
    L <- t(chol(mode$cov_u))
    sides <- .nl_outer_mode_transport(center, L, function(u) local_log_density(exp(u)))
    read <- function(query) {
        standard <- query[, 3:4, drop = FALSE]
        shifted <- sweep(standard %*% t(read_L), 2L, read_center - center, "+")
        query[, 3:4] <- sides$inverse(t(forwardsolve(L, t(shifted))))
        correction <- -log(2 * pi) - rowSums(query[, 3:4, drop = FALSE]^2) / 2 -
            rowSums(stats::dt(standard, read_df, log = TRUE)) +
            sum(log(diag(read_L))) - sum(log(diag(L))) -
            sides$log_derivative(query[, 3:4, drop = FALSE])
        list(query = query, correction = correction)
    }
    physical <- function(v) {
        cells <- matrix(0, nrow(v), ncol(res$theta_grid),
            dimnames = list(NULL, colnames(res$theta_grid)))
        cells[, li] <- exp(sweep(sides$forward(v[, 3:4, drop = FALSE]) %*% t(L),
                                2L, center, "+"))
        cells[, ri] <- region$physical(v[, 1:2, drop = FALSE])
        cells
    }
    log_density <- function(v) {
        -log(2 * pi) - sum(log(diag(L))) - rowSums(v[, 3:4, drop = FALSE]^2) / 2 +
            region$log_density(v[, 1:2, drop = FALSE]) -
            sides$log_derivative(v[, 3:4, drop = FALSE])
    }
    list(physical = physical, log_density = log_density,
         read = read, center = center, L = L, regional_log_stretch = region$log_stretch,
         read_center = read_center, read_L = read_L,
         local_radii = sides$radii,
         evaluations = evaluations)
}

.nl_aghq_fixed_projection <- function(native, partition, weights, groups, nodes, query) {
    moments <- .nested_fixed_moments(native, include_zero = TRUE)
    if (is.null(moments) || !ncol(moments$mu)) {
        return(list(primary = NULL, alternative = NULL, resolution = 0,
                    read_error = NULL))
    }
    P <- partition$local[, moments$keep, drop = FALSE]
    primary <- .nl_gauss_mixture_summary(moments$mu, moments$var, moments$w)
    p <- ncol(moments$mu)
    conditional <- cpp_outer_tensor_interpolate(nodes,
        cbind(moments$mu, log(moments$var)), query)
    alternative <- .nl_gauss_mixture_summary(conditional[, seq_len(p), drop = FALSE],
        exp(conditional[, p + seq_len(p), drop = FALSE]), weights)
    resolution <- max(abs(sweep(primary$quantiles - alternative$quantiles,
                                1L, alternative$sd, "/")))
    group_matrix <- Matrix::sparseMatrix(i = seq_along(groups), j = groups,
        x = 1, dims = c(length(groups), 8L))
    group_weights <- as.matrix(Matrix::crossprod(P, group_matrix)) +
        outer(partition$column[moments$keep],
              as.numeric(Matrix::crossprod(group_matrix, partition$row)))
    leave <- vapply(seq_len(8L), function(g) {
        w <- rowSums(group_weights[, -g, drop = FALSE])
        read <- .nl_gauss_mixture_summary(moments$mu, moments$var, w)
        if (is.null(read)) rep(NA_real_, length(primary$quantiles)) else
            as.numeric(read$quantiles)
    }, numeric(length(primary$quantiles)))
    error <- .nl_outer_jackknife(leave, stats::qt(.975, 7L)) /
        rep(alternative$sd, 3L)
    list(primary = primary, alternative = alternative, resolution = resolution,
         read_error = t(matrix(error, nrow = length(primary$sd))))
}

.nl_aghq_integrate <- function(res, solve, prior, cfg) {
    transform <- .nl_aghq_transform(res, prior, solve, cfg$read_df)
    groups <- rep(seq_len(8L), each = cfg$read_nodes / 8L)
    query <- do.call(rbind, lapply(seq_len(8L), function(g)
        cpp_outer_read_design(cfg$read_nodes / 8L, 4L, 761L + g)))
    query[, 3:4] <- stats::qt(query[, 3:4], cfg$read_df)
    read <- transform$read(query)
    query <- read$query
    cells <- transform$physical(query)
    log_q <- transform$log_density(query)
    u <- log(cells)
    hyper_quantiles <- function(weights) vapply(seq_len(ncol(u)), function(j)
        .nl_wtd_quantile(u[, j], weights, c(.025, .5, .975), outside = "clamp"),
        numeric(3L))
    previous <- NULL
    history <- list()
    reads <- list()
    evaluated <- transform$evaluations
    for (order in seq.int(cfg$order, cfg$max_order, by = 2L)) {
        gh <- gauss_hermite_prob(order)
        gl <- .nl_gauss_legendre_unit(order + 2L)
        nodes <- list(gl$x, gl$x, gh$nodes, gh$nodes)
        design <- as.matrix(expand.grid(nodes, KEEP.OUT.ATTRS = FALSE))
        points <- transform$physical(design)
        native <- .nl_outer_solve(points, solve, TRUE)
        native$theta_grid <- points
        evaluated <- evaluated + nrow(points)
        measure <- as.matrix(expand.grid(list(gl$w, gl$w, gh$weights, gh$weights),
                                        KEEP.OUT.ATTRS = FALSE))
        log_ratio <- native$log_marginal - transform$log_density(design)
        log_z <- .tulpa_logsumexp(log_ratio + rowSums(log(measure)))
        predicted <- as.numeric(cpp_outer_tensor_interpolate(nodes,
            matrix(log_ratio, ncol = 1L), query))
        log_weight <- predicted + read$correction - log(cfg$read_nodes)
        log_read_integral <- .tulpa_logsumexp(log_weight)
        weights <- .nl_normalise_weights_safe(log_weight)
        native_weights <- .nl_normalise_weights_safe(log_ratio + rowSums(log(measure)))
        partition <- .nl_outer_coupling(cpp_outer_tensor_partition(nodes, query, weights),
                                       native_weights)
        weights <- .nl_outer_coupling_mass(partition, "row")
        log_weight <- log(weights) + log_read_integral
        native$weights <- .nl_outer_coupling_mass(partition, "column")
        fixed <- .nl_aghq_fixed_projection(native, partition, weights, groups, nodes, query)
        read_error <- .nl_qmc_error(u, log_weight, groups, list(),
            rep(cfg$read_nodes / 8L, 8L))
        read_error$fixed_quantiles <- fixed$read_error
        read_errors <- c(read_error$hyper_quantiles, fixed$read_error)
        read_error$max <- if (all(is.finite(read_errors))) max(read_errors) else Inf
        hyper <- hyper_quantiles(weights)
        hyper_mean <- as.numeric(crossprod(weights, u))
        hyper_sd <- sqrt(colSums(weights * sweep(u, 2L, hyper_mean)^2))
        hyper_read <- sweep(read_error$hyper_quantiles, 2L, hyper_sd, "*")
        hyper_change <- if (is.null(previous)) NULL else hyper - previous$hyper
        hyper_change_noise <- if (is.null(previous)) NULL else hyper_read + previous$hyper_read
        fixed_read <- if (is.null(fixed$alternative)) NULL else
            sweep(t(fixed$read_error), 1L, fixed$alternative$sd, "*")
        fixed_change <- if (is.null(previous)) NULL else fixed$alternative$quantiles - previous$fixed
        fixed_change_noise <- if (is.null(previous)) NULL else fixed_read + previous$fixed_read
        state <- list(hyper = hyper, fixed = fixed$alternative$quantiles,
            fixed_read = fixed_read, fixed_change = fixed_change,
            fixed_change_noise = fixed_change_noise,
            hyper_read = hyper_read, hyper_change = hyper_change,
            hyper_change_noise = hyper_change_noise)
        if (length(history) < 2L) {
            error <- list(max = Inf, log_evidence = Inf, quantiles = Inf,
                          read = read_error, refinement = Inf, complete = FALSE)
        } else {
            hyper_refinement <- max(sweep(.nl_outer_quantile_error(hyper_change,
                previous$hyper_change, hyper_change_noise, previous$hyper_change_noise),
                2L, hyper_sd, "/"))
            fixed_refinement <- if (is.null(fixed$alternative)) 0 else max(sweep(
                .nl_outer_quantile_error(fixed_change, previous$fixed_change,
                    fixed_change_noise, previous$fixed_change_noise),
                1L, fixed$alternative$sd, "/"))
            refinement <- max(hyper_refinement, fixed_refinement)
            hyper_error <- hyper_refinement + max(read_error$hyper_quantiles)
            fixed_error <- fixed_refinement + fixed$resolution +
                if (is.null(fixed$read_error)) 0 else max(fixed$read_error)
            quantile_error <- max(hyper_error, fixed_error)
            error <- list(max = quantile_error,
                log_evidence = .nl_outer_evidence_error(c(vapply(history,
                    function(h) h[["log_evidence"]], numeric(1)), log_z)),
                quantiles = quantile_error,
                hyper_refinement = hyper_refinement, fixed_refinement = fixed_refinement,
                refinement = refinement, conditional_resolution = fixed$resolution,
                read = read_error, complete = TRUE)
        }
        converged <- error$max <= cfg$tol && error$log_evidence <= cfg$evidence_tol
        history[[length(history) + 1L]] <- c(order = order,
            points = nrow(points), evaluated = evaluated, error = error$max,
            evidence_error = error$log_evidence, log_evidence = log_z)
        reads[[length(reads) + 1L]] <- list(hyper = hyper,
            hyper_sd = hyper_sd, fixed = fixed$primary$quantiles,
            fixed_interpolated = fixed$alternative$quantiles, error = error)
        if (converged || order == cfg$max_order) break
        previous <- state
    }
    out <- native
    error$hyper_quantiles <- matrix(error$quantiles, 3L, ncol(cells))
    if (!is.null(fixed$primary)) error$fixed_quantiles <-
        matrix(error$quantiles, 3L, length(fixed$primary$mean))
    interpolated_z <- .tulpa_logsumexp(log_weight)
    log_quad <- -log(cfg$read_nodes) - log_q + read$correction
    out$outer_posterior <- list(points = cells, weights = weights,
        partition = partition, log_marginal = log(weights) + log_z - log_quad,
        log_quad = log_quad)
    out$outer_quadrature <- list(method = "aghq_log_spline", n = nrow(points),
        n_evaluated = evaluated, read_nodes = cfg$read_nodes, read_df = cfg$read_df,
        order = order, error = error, tol = cfg$tol, evidence_tol = cfg$evidence_tol,
        converged = converged, history = do.call(rbind, history), reads = reads,
        center = transform$center, L = transform$L,
        read_center = transform$read_center, read_L = transform$read_L,
        local_radii = transform$local_radii,
        regional_log_stretch = transform$regional_log_stretch,
        normalizer_adjustment = log_z - interpolated_z,
        error_method = "successive_orders_and_shifted_read_designs")
    measure <- log_z + log(out$weights) - out$log_marginal
    .nl_outer_finish(out, res, points, measure, "aghq")
}
