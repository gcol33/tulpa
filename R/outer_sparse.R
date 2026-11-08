.nl_sparse_key <- function(index) paste(index, collapse = ":")

.nl_outer_point_key <- function(points) {
    apply(points, 1L, function(row) paste(sprintf("%.17g", row), collapse = ":"))
}

.nl_sparse_tensor <- function(rules, index) {
    selected <- Map(`[[`, rules, index)
    nodes <- lapply(selected, `[[`, "x")
    points <- as.matrix(expand.grid(nodes, KEEP.OUT.ATTRS = FALSE))
    weights <- Reduce(`*`, as.data.frame(expand.grid(lapply(selected, `[[`, "w"),
                                                    KEEP.OUT.ATTRS = FALSE)))
    difference <- Map(function(rule, level, family) {
        w <- rule$w
        if (level > 1L) {
            previous <- family[[level - 1L]]
            at <- match(previous$x, rule$x)
            w[at] <- w[at] - previous$w
        }
        w
    }, selected, index, rules)
    delta <- Reduce(`*`, as.data.frame(expand.grid(difference, KEEP.OUT.ATTRS = FALSE)))
    list(nodes = nodes, points = points, weights = weights, difference = delta)
}

.nl_sparse_combination <- function(indices) {
    d <- length(indices[[1L]])
    shifts <- as.matrix(expand.grid(rep(list(0:1), d)))
    coefficients <- new.env(hash = TRUE, parent = emptyenv())
    levels <- new.env(hash = TRUE, parent = emptyenv())
    for (index in indices) for (r in seq_len(nrow(shifts))) {
        lower <- index - shifts[r, ]
        if (any(lower < 1L)) next
        key <- .nl_sparse_key(lower)
        value <- if (exists(key, coefficients, inherits = FALSE)) coefficients[[key]] else 0
        coefficients[[key]] <- value + (-1)^sum(shifts[r, ])
        levels[[key]] <- lower
    }
    keys <- ls(coefficients)
    keys <- keys[vapply(keys, function(key) coefficients[[key]] != 0, logical(1))]
    lapply(keys, function(key) list(index = levels[[key]], coefficient = coefficients[[key]]))
}

.nl_sparse_cache <- function(rules, physical, solve) {
    points <- matrix(numeric(0), 0L, length(rules))
    cells <- NULL
    native <- NULL
    keys <- character(0)
    tensors <- new.env(hash = TRUE, parent = emptyenv())
    tensor <- function(index) {
        key <- .nl_sparse_key(index)
        if (exists(key, tensors, inherits = FALSE)) return(tensors[[key]])
        grid <- .nl_sparse_tensor(rules, index)
        names <- .nl_outer_point_key(grid$points)
        at <- match(names, keys)
        missing <- which(is.na(at))
        if (length(missing)) {
            fresh <- grid$points[missing, , drop = FALSE]
            physical_points <- physical(fresh)
            result <- .nl_outer_solve(physical_points, solve, TRUE)
            native <<- if (is.null(native)) result else .nl_bind_cell_results(native,
                list(list(res = result, n = length(missing))), nrow(points))
            at[missing] <- nrow(points) + seq_along(missing)
            points <<- rbind(points, fresh)
            cells <<- rbind(cells, physical_points)
            keys <<- c(keys, names[missing])
        }
        grid$rows <- at
        tensors[[key]] <- grid
        grid
    }
    list(tensor = tensor, state = function() list(points = points, cells = cells, native = native))
}

.nl_sparse_prediction <- function(grid, values, query, rows, groups, previous = NULL) {
    n <- nrow(query)
    selected <- if (is.null(rows)) seq_len(n) else rows
    per_group <- n / groups
    ids <- ((selected - 1L) %% per_group) * groups +
        (selected - 1L) %/% per_group + 1L
    if (is.null(previous)) previous <- list(dense = 0L, ids = integer(),
        values = matrix(numeric(), 0L, ncol(values)))
    lookup <- function(ids, state) {
        at <- ids
        sparse <- which(ids > state$dense)
        at[sparse] <- state$dense + match(ids[sparse], state$ids)
        at
    }
    at <- lookup(ids, previous)
    missing <- unique(ids[is.na(at)])
    fresh <- NULL
    if (length(missing)) {
        query_rows <- ((missing - 1L) %% groups) * per_group +
            (missing - 1L) %/% groups + 1L
        fresh <- cpp_outer_tensor_interpolate(grid$nodes,
            values[grid$rows, , drop = FALSE], query[query_rows, , drop = FALSE])
    }
    if (is.null(rows) && previous$dense < n) {
        stored <- matrix(0, n, ncol(values))
        if (previous$dense) stored[seq_len(previous$dense), ] <-
            previous$values[seq_len(previous$dense), , drop = FALSE]
        inside <- which(previous$ids <= n)
        if (length(inside)) stored[previous$ids[inside], ] <-
            previous$values[previous$dense + inside, , drop = FALSE]
        if (length(missing)) stored[missing, ] <- fresh
        outside <- which(previous$ids > n)
        stored <- rbind(stored, previous$values[previous$dense + outside, , drop = FALSE])
        previous <- list(dense = n, ids = previous$ids[outside], values = stored)
    } else if (length(missing)) {
        previous$ids <- c(previous$ids, missing)
        previous$values <- rbind(previous$values, fresh)
    }
    list(values = previous$values[lookup(ids, previous), , drop = FALSE], cache = previous)
}

.nl_sparse_interpolate <- function(combination, cache, values, query, predictions = NULL,
                                   rows = NULL, groups = 1L) {
    result <- matrix(0, if (is.null(rows)) nrow(query) else length(rows), ncol(values))
    for (term in combination) {
        key <- .nl_sparse_key(term$index)
        previous <- if (!is.null(predictions) && exists(key, predictions, inherits = FALSE))
            predictions[[key]]
        predicted <- .nl_sparse_prediction(cache$tensor(term$index), values, query,
                                            rows, groups, previous)
        if (!is.null(predictions)) predictions[[key]] <- predicted$cache
        result <- result + term$coefficient * predicted$values
    }
    result
}

.nl_sparse_frontier <- function(accepted, active, maximum) {
    keys <- vapply(c(accepted, active), .nl_sparse_key, character(1))
    accepted_keys <- vapply(accepted, .nl_sparse_key, character(1))
    result <- list()
    for (index in accepted) for (j in seq_along(index)) {
        next_index <- index
        next_index[j] <- next_index[j] + 1L
        key <- .nl_sparse_key(next_index)
        if (next_index[j] > maximum[j] || key %in% keys) next
        predecessors <- lapply(which(next_index > 1L), function(k) {
            lower <- next_index
            lower[k] <- lower[k] - 1L
            .nl_sparse_key(lower)
        })
        if (all(unlist(predecessors) %in% accepted_keys)) {
            result[[length(result) + 1L]] <- next_index
            keys <- c(keys, key)
        }
    }
    result
}

.nl_sparse_integral <- function(indices, cache, log_ratio) {
    offset <- max(log_ratio)
    contributions <- vapply(indices, function(index) {
        grid <- cache$tensor(index)
        sum(grid$difference * exp(log_ratio[grid$rows] - offset))
    }, numeric(1))
    list(value = sum(contributions), contributions = contributions, offset = offset)
}

.nl_sparse_positive_weights <- function(features, target, base, tolerance = 1e-9) {
    scale <- sqrt(colMeans(sweep(features, 2L, target)^2))
    varying <- which(is.finite(scale) & scale > sqrt(.Machine$double.eps))
    A <- sweep(sweep(features[, varying, drop = FALSE], 2L, target[varying]),
               2L, scale[varying], "/")
    decomposition <- qr(A, tol = 1e-10)
    A <- A[, decomposition$pivot[seq_len(decomposition$rank)], drop = FALSE]
    log_base <- log(base / sum(base))
    # Relative entropy is bounded by -min(log_base) on the supported cells.
    # At the optimum this penalty bounds feasible moment mismatch by tolerance.
    penalty <- tolerance^2 / (2 * max(1, -log_base[is.finite(log_base)]))
    parameter <- numeric(ncol(A))
    objective <- function(x) .tulpa_logsumexp(log_base + as.numeric(A %*% x)) +
        penalty * sum(x^2) / 2
    for (iteration in seq_len(100L)) {
        weights <- .nl_normalise_weights_safe(log_base + as.numeric(A %*% parameter))
        moment <- as.numeric(crossprod(weights, A))
        gradient <- moment + penalty * parameter
        residual <- max(abs(gradient), 0)
        if (residual <= tolerance) break
        centered <- sweep(A, 2L, moment)
        hessian <- crossprod(centered, weights * centered)
        spectral <- eigen(hessian, symmetric = TRUE)
        direction <- as.numeric(spectral$vectors %*% (crossprod(spectral$vectors, gradient) /
            (pmax(0, spectral$values) + penalty)))
        value <- objective(parameter)
        step <- 1
        repeat {
            proposed <- parameter - step * direction
            if (objective(proposed) <= value - step * sum(gradient * direction) / 4) break
            step <- step / 2
            if (step < .Machine$double.eps) break
        }
        parameter <- proposed
        if (step < .Machine$double.eps) break
    }
    residual <- max(abs((as.numeric(crossprod(weights, features)) - target)[varying] /
                        scale[varying]), 0)
    list(weights = weights, residual = residual, iterations = iteration, penalty = penalty)
}

.nl_sparse_fixed_features <- function(mu, variance, locations, moments = FALSE) {
    features <- do.call(cbind, lapply(seq_len(ncol(mu)), function(j)
        vapply(locations[j, ], function(x) stats::pnorm(x, mu[, j], sqrt(variance[, j])),
               numeric(nrow(mu)))))
    if (moments) features <- cbind(features, mu, variance + mu^2)
    features
}

.nl_sparse_positive_mixture <- function(native, points, read_points, base, interpolated, weights) {
    native$weights <- base / sum(base)
    moments <- .nested_fixed_moments(native, include_zero = TRUE)
    p <- if (is.null(moments)) 0L else ncol(moments$mu)
    keep <- if (p) moments$keep else seq_along(base)
    features <- matrix(numeric(0), length(keep), 0L)
    target <- numeric(0)
    if (p) {
        mu <- interpolated[, 1L + seq_len(p), drop = FALSE]
        variance <- exp(interpolated[, 1L + p + seq_len(p), drop = FALSE])
        alternative <- .nl_gauss_mixture_summary(mu, variance, weights,
            probs = c(.025, .1, .25, .5, .75, .9, .975))
        locations <- alternative$quantiles
        features <- .nl_sparse_fixed_features(moments$mu, moments$var, locations, moments = TRUE)
        target <- as.numeric(crossprod(weights,
            .nl_sparse_fixed_features(mu, variance, locations, moments = TRUE)))
    }
    center <- as.numeric(crossprod(weights, log(read_points)))
    scale <- sqrt(colSums(weights * sweep(log(read_points), 2L, center)^2))
    coordinates <- function(x) {
        x <- sweep(sweep(log(x), 2L, center), 2L, scale, "/")
        products <- lapply(seq_len(ncol(x)), function(j) x[, j] * x[, seq_len(j), drop = FALSE])
        cbind(x, do.call(cbind, products))
    }
    features <- cbind(features, coordinates(points[keep, , drop = FALSE]))
    target <- c(target, as.numeric(crossprod(weights, coordinates(read_points))))
    fit <- .nl_sparse_positive_weights(features, target, native$weights[keep])
    mass <- numeric(length(base))
    mass[keep] <- fit$weights
    if (!p) return(list(weights = mass, primary = NULL, alternative = NULL,
                        resolution = 0, residual = fit$residual))
    primary <- .nl_gauss_mixture_summary(moments$mu, moments$var, fit$weights)
    alternative$quantiles <- alternative$quantiles[, c(1L, 4L, 7L), drop = FALSE]
    resolution <- max(abs(sweep(primary$quantiles - alternative$quantiles,
                                1L, alternative$sd, "/")))
    list(weights = mass, primary = primary, alternative = alternative,
         resolution = resolution, residual = fit$residual)
}

.nl_sparse_read_design <- function(transform, n, d, df, seed = 761L) {
    groups <- rep(seq_len(8L), each = n / 8L)
    query <- do.call(rbind, lapply(seq_len(8L), function(g)
        cpp_outer_read_design(n / 8L, d, seed + g)))
    query[, 3:d] <- stats::qt(query[, 3:d, drop = FALSE], df)
    read <- transform$read(query)
    cells <- transform$physical(read$query)
    list(query = read$query, cells = cells, log_q = transform$log_density(read$query),
         u = log(cells), correction = read$correction, groups = groups)
}

.nl_sparse_read_action <- function(n, maximum, grid, read, tolerance,
                                    previous = NULL, refit = FALSE) {
    remaining <- tolerance - grid
    needed <- read > remaining
    goal <- which.max((grid + read) / tolerance)
    binding <- needed[goal] && grid[goal] < tolerance[goal]
    attainable <- read[goal] * sqrt(n / maximum) <= remaining[goal]
    falling <- is.null(previous) ||
        (is.finite(read[goal]) &&
         read[goal] <= previous$error[goal] * sqrt(previous$nodes / n))
    action <- if (binding && !falling && refit) "refresh" else
        if (binding && attainable && falling && n < maximum) "grow" else "refine"
    list(action = action, goal = goal)
}

.nl_sparse_compile <- function(combination, cache, values, n, previous = NULL) {
    terms <- lapply(combination, function(term) {
        grid <- cache$tensor(term$index)
        list(index = term$index, nodes = grid$nodes, rows = grid$rows, coefficient = term$coefficient)
    })
    cpp_outer_sparse_compile(terms, values, n,
        if (is.null(previous)) NULL else previous$pointer)
}

.nl_aghq_integrate <- function(res, solve, prior, cfg) {
    transform <- .nl_aghq_transform(res, prior, solve, cfg$read_df)
    d <- ncol(res$theta_grid)
    normal <- transform$normal_axes
    rules <- lapply(normal, .nl_sparse_rules)
    maximum <- pmin(lengths(rules), cfg$max_level)
    cache <- .nl_sparse_cache(rules, transform$physical, solve)
    cache$tensor(rep(1L, d))
    accepted <- list(rep(1L, d))
    active <- .nl_sparse_frontier(accepted, list(), maximum)
    boundary <- list()
    all_read <- .nl_sparse_read_design(transform, cfg$read_nodes, d, cfg$read_df)
    groups <- all_read$groups
    query <- all_read$query
    cells <- all_read$cells
    log_q <- all_read$log_q
    u <- all_read$u
    read <- list(correction = all_read$correction)
    full_read <- cfg$read_nodes <= 4096L
    if (!full_read) {
        at <- unlist(lapply(seq_len(8L), function(g) utils::head(which(groups == g), 512L)))
        query <- query[at, , drop = FALSE]
        cells <- cells[at, , drop = FALSE]
        log_q <- log_q[at]
        u <- u[at, , drop = FALSE]
        read$correction <- read$correction[at]
        groups <- groups[at]
    }
    history <- reads <- list()
    predictions <- new.env(hash = TRUE, parent = emptyenv())
    previous <- NULL
    previous_read <- NULL
    fitted_read_grid <- NULL
    read_generation <- 0L
    read_seed <- 761L
    read_history <- list()
    compiled <- NULL
    compiled_key <- NULL
    on.exit(if (!is.null(compiled)) cpp_outer_sparse_clear(compiled$pointer), add = TRUE)
    repeat {
        boundary <- Filter(function(index) !any(vapply(c(accepted, active), function(other)
            all(other >= index) && any(other > index), logical(1))), boundary)
        for (index in active) cache$tensor(index)
        state <- cache$state()
        native <- state$native
        native$theta_grid <- state$cells
        ratio <- native$log_marginal - transform$log_density(state$points)
        native$weights <- rep(1 / length(ratio), length(ratio))
        moments <- .nested_fixed_moments(native, include_zero = TRUE)
        values <- matrix(ratio, ncol = 1L)
        if (!is.null(moments)) values <- cbind(values, moments$mu, log(moments$var))
        indices <- c(accepted, active)
        grid_key <- paste(sort(vapply(indices, .nl_sparse_key, character(1))), collapse = ";")
        combination <- .nl_sparse_combination(indices)
        key <- paste(grid_key, full_read, sep = "|")
        if (!identical(key, compiled_key)) {
            compiled <- .nl_sparse_compile(combination, cache, values, nrow(query), compiled)
            compiled_key <- key
        }
        interpolated <- cpp_outer_sparse_evaluate(compiled$pointer, query)
        log_weight <- interpolated[, 1L] + read$correction - log(nrow(query))
        weights <- .nl_normalise_weights_safe(log_weight)
        integral <- .nl_sparse_integral(indices, cache, ratio)
        log_quadrature <- if (integral$value > 0) log(integral$value) + integral$offset else NaN
        log_z <- .tulpa_logsumexp(log_weight)
        base <- numeric(length(ratio))
        for (index in indices) {
            grid <- cache$tensor(index)
            base[grid$rows] <- base[grid$rows] + abs(grid$difference) *
                exp(ratio[grid$rows] - max(ratio))
        }
        fixed <- .nl_sparse_positive_mixture(native, state$cells, cells,
            base, interpolated, weights)
        native$weights <- fixed$weights
        hyper <- vapply(seq_len(d), function(j)
            .nl_wtd_quantile(u[, j], weights, c(.025, .5, .975), outside = "clamp"), numeric(3L))
        hyper_mean <- as.numeric(crossprod(weights, u))
        hyper_sd <- sqrt(colSums(weights * sweep(u, 2L, hyper_mean)^2))
        p <- if (is.null(moments)) 0L else ncol(moments$mu)
        read_fixed <- if (p) list(mu = interpolated[, 1L + seq_len(p), drop = FALSE],
            var = exp(interpolated[, 1L + p + seq_len(p), drop = FALSE]))
        read_error <- .nl_outer_read_error(u, log_weight, groups,
            rep(nrow(query) / 8L, 8L), fixed = read_fixed)
        read_budget <- max(c(read_error$hyper_quantiles, read_error$fixed_quantiles))
        probe <- unlist(lapply(seq_len(8L), function(g)
            utils::head(which(groups == g), 256L)))
        probe_weights <- weights[probe] / sum(weights[probe])
        frontier <- c(active, boundary)
        conditional_surplus <- numeric(length(frontier))
        hyper_surplus <- matrix(0, 3L, d)
        hyper_cdf <- lapply(seq_len(d), function(j)
            vapply(hyper[, j], function(x) as.numeric(u[probe, j] <= x), numeric(length(probe))))
        surplus <- vapply(seq_along(frontier), function(i) {
            index <- frontier[[i]]
            delta <- .nl_sparse_interpolate(.nl_sparse_combination(list(index)), cache,
                values, query, predictions, rows = probe, groups = 8L)
            lower_weights <- .nl_normalise_weights_safe(log(probe_weights) - delta[, 1L])
            differences <- vapply(hyper_cdf, function(cdf)
                as.numeric(crossprod(lower_weights - probe_weights, cdf)), numeric(3L))
            hyper_surplus <<- hyper_surplus + abs(differences)
            if (!is.null(moments)) {
                p <- ncol(moments$mu)
                mu <- interpolated[probe, 1L + seq_len(p), drop = FALSE]
                variance <- exp(interpolated[probe, 1L + p + seq_len(p), drop = FALSE])
                cdf <- .nl_sparse_fixed_features(mu, variance, fixed$alternative$quantiles)
                lower_cdf <- .nl_sparse_fixed_features(
                    mu - delta[, 1L + seq_len(p), drop = FALSE],
                    variance * exp(-delta[, 1L + p + seq_len(p), drop = FALSE]),
                    fixed$alternative$quantiles)
                conditional_surplus[i] <<- max(abs(colSums(lower_weights * lower_cdf) -
                                                       colSums(probe_weights * cdf)))
            }
            max(abs(differences))
        }, numeric(1))
        remainder <- max(hyper_surplus)
        probability_error <- min(1, remainder)
        quantile_error <- 0
        for (j in seq_len(d)) {
            bounds <- vapply(c(.025, .5, .975), function(p) {
                endpoints <- .nl_wtd_quantile(u[, j], weights,
                    pmin(1, pmax(0, p + c(-1, 1) * hyper_surplus[match(p, c(.025, .5, .975)), j])),
                    outside = "clamp")
                max(abs(endpoints - hyper[, j][match(p, c(.025, .5, .975))])) / hyper_sd[j]
            }, numeric(1))
            quantile_error <- max(quantile_error, bounds)
        }
        if (!is.null(moments)) {
            cdf_error <- sum(conditional_surplus)
            if (cdf_error >= .025) {
                quantile_error <- Inf
            } else {
                p <- ncol(moments$mu)
                interval <- .nl_gauss_mixture_summary(
                    interpolated[, 1L + seq_len(p), drop = FALSE],
                    exp(interpolated[, 1L + p + seq_len(p), drop = FALSE]), weights,
                    probs = rep(c(.025, .5, .975), each = 2L) + c(-1, 1) * cdf_error)
                target <- fixed$alternative$quantiles[, rep(1:3, each = 2L), drop = FALSE]
                quantile_error <- max(quantile_error,
                    abs(sweep(interval$quantiles - target, 1L, fixed$alternative$sd, "/")))
            }
        }
        at <- match(vapply(frontier, .nl_sparse_key, character(1)),
                    vapply(indices, .nl_sparse_key, character(1)))
        evidence_frontier <- sum(abs(integral$contributions[at])) /
            abs(integral$value)
        quadrature_error <- if (is.finite(log_quadrature) && evidence_frontier < 1)
            -log1p(-evidence_frontier) else Inf
        normalizer_discrepancy <- if (is.finite(log_quadrature)) abs(log_z - log_quadrature) else Inf
        evidence_error <- quadrature_error + normalizer_discrepancy + read_error$log_evidence
        grid_budget <- quantile_error + fixed$resolution + fixed$residual
        promote <- !full_read && (!length(active) || (length(history) >= 2L &&
            grid_budget <= cfg$tol))
        allocation <- .nl_sparse_read_action(nrow(query), cfg$read_max_nodes,
            c(grid_budget, quadrature_error + normalizer_discrepancy),
            c(read_budget, read_error$log_evidence), c(cfg$tol, cfg$evidence_tol), previous_read,
            refit = !identical(fitted_read_grid, grid_key))
        grow <- full_read && identical(allocation$action, "grow")
        refresh <- full_read && identical(allocation$action, "refresh")
        read_history[[length(read_history) + 1L]] <- c(points = nrow(state$points),
            read_nodes = nrow(query), ess = 1 / sum(weights^2), max_weight = max(weights),
            quantile_grid = grid_budget, quantile_read = read_budget,
            evidence_grid = quadrature_error, evidence_read = read_error$log_evidence,
            normalizer_discrepancy = normalizer_discrepancy,
            goal = allocation$goal,
            promote = promote, refresh = refresh, grow = grow)
        if (promote || refresh || grow) {
            if (promote || refresh) {
                parameters <- transform$adapt_read(cells, weights)
                transform$read_center <- parameters$center
                transform$read_L <- parameters$L
                read_generation <- read_generation + 1L
                read_seed <- 2761L + 8L * (read_generation - 1L)
                if (refresh) fitted_read_grid <- grid_key
            }
            count <- if (grow) min(2 * nrow(query), cfg$read_max_nodes) else
                if (refresh) nrow(query) else cfg$read_nodes
            previous_read <- if (grow) list(nodes = nrow(query),
                error = c(read_budget, read_error$log_evidence)) else NULL
            if (isTRUE(cfg$progress)) {
                cat("READ", count, "error", read_budget, "ESS", 1 / sum(weights^2), "\n")
                flush.console()
            }
            all_read <- .nl_sparse_read_design(transform, count, d, cfg$read_df,
                seed = read_seed)
            query <- all_read$query
            cells <- all_read$cells
            log_q <- all_read$log_q
            u <- all_read$u
            read$correction <- all_read$correction
            groups <- all_read$groups
            full_read <- TRUE
            if (!grow) predictions <- new.env(hash = TRUE, parent = emptyenv())
            next
        }
        quantile_error <- quantile_error + read_budget +
            fixed$resolution + fixed$residual
        if (!full_read || is.null(previous) || length(history) < 2L) {
            quantile_error <- evidence_error <- Inf
        } else {
            if (isTRUE(previous$full_read)) {
                movement <- max(abs(sweep(hyper - previous$hyper, 2L, hyper_sd, "/")))
                quantile_error <- max(quantile_error, movement)
            }
            if (is.finite(log_quadrature) && is.finite(previous$log_quadrature))
                evidence_error <- max(evidence_error, abs(log_quadrature - previous$log_quadrature))
        }
        error <- list(max = quantile_error, quantiles = quantile_error,
            hyper_quantiles = matrix(quantile_error, 3L, d),
            fixed_quantiles = if (!is.null(moments)) matrix(quantile_error, 3L, ncol(moments$mu)),
            log_evidence = evidence_error, read = read_error, complete = full_read,
            evidence_quadrature = quadrature_error, normalizer_discrepancy = normalizer_discrepancy,
            conditional_resolution = fixed$resolution, frontier = remainder,
            positive_residual = fixed$residual)
        error$quantile_grid <- grid_budget
        converged <- is.finite(log_z) && quantile_error <= cfg$tol && evidence_error <= cfg$evidence_tol
        history[[length(history) + 1L]] <- c(order = max(vapply(indices, max, numeric(1))),
            points = nrow(state$points), evaluated = nrow(state$points) + transform$evaluations,
            read_nodes = nrow(query), read_ess = 1 / sum(weights^2), error = quantile_error,
            evidence_error = evidence_error, log_evidence = log_z, log_quadrature = log_quadrature)
        reads[[length(reads) + 1L]] <- list(hyper = hyper, hyper_sd = hyper_sd,
            fixed = fixed$primary$quantiles, fixed_interpolated = fixed$alternative$quantiles,
            fixed_mean = fixed$primary$mean, fixed_sd = fixed$primary$sd,
            fixed_interpolated_mean = fixed$alternative$mean,
            fixed_interpolated_sd = fixed$alternative$sd,
            error = error)
        if (is.function(cfg$observe)) cfg$observe(utils::tail(history, 1L)[[1L]],
                                                utils::tail(reads, 1L)[[1L]])
        if (isTRUE(cfg$progress)) {
            cat("SPARSE", length(history), nrow(state$points), quantile_error, evidence_error,
                "grid", grid_budget, "read_error", read_budget, "quadrature", quadrature_error,
                "density", probability_error, "conditional", sum(conditional_surplus),
                "positive", fixed$residual, "read", nrow(query), format(Sys.time()), "\n")
            flush.console()
        }
        if (converged || !length(active)) break
        previous <- list(hyper = hyper, log_quadrature = log_quadrature, full_read = full_read)
        previous_read <- NULL
        active_surplus <- if (allocation$goal == 2L)
            abs(integral$contributions[match(vapply(active, .nl_sparse_key, character(1)),
                vapply(indices, .nl_sparse_key, character(1)))]) / abs(integral$value) else
            pmax(surplus, conditional_surplus)[seq_along(active)]
        work <- vapply(active, function(index) prod(vapply(seq_len(d), function(j) {
            size <- length(rules[[j]][[index[j]]]$x)
            size - if (index[j] > 1L) length(rules[[j]][[index[j] - 1L]]$x) else 0L
        }, integer(1))), numeric(1))
        active_surplus <- pmax(active_surplus, fixed$residual / work)
        profit <- active_surplus / work
        selected <- which(profit >= max(profit) / 4)
        boundary <- c(boundary, Filter(function(index) any(index == maximum), active[selected]))
        accepted <- c(accepted, active[selected])
        active <- active[-selected]
        active <- c(active, .nl_sparse_frontier(accepted, active, maximum))
    }
    local <- cpp_outer_nearest_partition(state$points, query, weights)
    partition <- .nl_outer_coupling(local, native$weights)
    log_quad <- -log(nrow(query)) - log_q + read$correction
    native$outer_posterior <- list(points = cells, weights = weights, partition = partition,
        log_marginal = log(weights) + log_z - log_quad, log_quad = log_quad)
    native$outer_quadrature <- list(method = "aghq_log_spline", construction = "adaptive_smolyak",
        n = nrow(state$points), n_evaluated = nrow(state$points) + transform$evaluations,
        read_nodes = nrow(query), read_df = cfg$read_df, order = length(history),
        read_max_nodes = cfg$read_max_nodes, read_variance = transform$read_variance,
        compiled_groups = compiled$groups, compiled_coefficients = compiled$coefficients,
        error = error, tol = cfg$tol, evidence_tol = cfg$evidence_tol, converged = converged,
        history = do.call(rbind, history), reads = reads, read_history = do.call(rbind, read_history),
        indices = indices, active = active,
        center = transform$center, L = transform$L, read_center = transform$read_center,
        read_L = transform$read_L, local_radii = transform$local_radii,
        regional_log_stretch = transform$regional_log_stretch,
        coordinate_center = transform$coordinate_center, coordinate_L = transform$coordinate_L,
        normal_axes = normal,
        error_method = "hierarchical_frontier_and_shifted_reads")
    measure <- log_z + log(native$weights) - native$log_marginal
    .nl_outer_finish(native, res, state$cells, measure, "aghq")
}
