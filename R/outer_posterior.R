.nl_outer_coupling <- function(local, target, tolerance = 1e-13) {
    source <- as.numeric(Matrix::rowSums(local))
    source <- source / sum(source)
    target <- as.numeric(target) / sum(target)
    discrepancy <- sum(abs(as.numeric(Matrix::colSums(local)) - target)) / 2
    completion <- discrepancy^2
    degree <- tabulate(local@i + 1L, nbins = nrow(local))
    groups <- seq_along(source)
    reduced <- local
    group_source <- source
    if (all(degree <= 1L)) {
        groups[] <- 1L
        groups[local@i + 1L] <- rep(seq_len(ncol(local)), diff(local@p))
        mass <- as.numeric(Matrix::colSums(local))
        reduced <- Matrix::Diagonal(x = mass)
        group_source <- mass / sum(mass)
    }
    a <- rep(1, length(group_source))
    b <- rep(1, length(target))
    rows <- group_source > 0
    columns <- target > 0
    residual <- 2 * discrepancy
    iteration <- 0L
    if (residual > tolerance / 4) for (iteration in seq_len(5000L)) {
        local_mass <- as.numeric(reduced %*% b)
        a[] <- 0
        a[rows] <- 1 / (local_mass[rows] / group_source[rows] + completion * sum(target * b))
        local_mass <- as.numeric(Matrix::crossprod(reduced, a))
        b[] <- 0
        b[columns] <- 1 / (local_mass[columns] / target[columns] +
            completion * sum(group_source * a))
        mass <- a * as.numeric(reduced %*% b) + completion * a * group_source * sum(target * b)
        residual <- sum(abs(mass - group_source))
        if (is.finite(residual) && residual <= tolerance / 4) break
    }
    if (!is.finite(residual) || residual > tolerance) {
        stop("The outer posterior coupling did not balance its marginal masses.", call. = FALSE)
    }
    a <- a[groups]
    local@x <- exp(log(local@x) + log(a[local@i + 1L]) +
        rep(log(b), diff(local@p)))
    row <- a * source
    column <- b * target
    global_mass <- completion * sum(row) * sum(column)
    out <- list(local = local, row = global_mass * row / sum(row),
                column = column / sum(column), iterations = iteration)
    out$residual <- sum(abs(.nl_outer_coupling_mass(out, "row") - source)) +
        sum(abs(.nl_outer_coupling_mass(out, "column") - target))
    if (!is.finite(out$residual) || out$residual > tolerance) {
        stop("The outer posterior coupling lost its marginal masses.", call. = FALSE)
    }
    out
}

.nl_outer_coupling_mass <- function(coupling, margin = c("row", "column")) {
    margin <- match.arg(margin)
    if (margin == "row") {
        as.numeric(Matrix::rowSums(coupling$local)) + coupling$row * sum(coupling$column)
    } else {
        as.numeric(Matrix::colSums(coupling$local)) + coupling$column * sum(coupling$row)
    }
}

.nl_outer_hyper_view <- function(object) {
    posterior <- object$outer_posterior
    if (is.null(posterior)) return(object)
    object$theta_grid <- posterior$points
    object$weights <- posterior$weights
    object$log_marginal <- posterior$log_marginal
    object$log_quad <- posterior$log_quad
    object$refining_axis <- rep("aghq", nrow(posterior$points))
    object$weight_kind <- rep("importance", nrow(posterior$points))
    object$outer_posterior <- NULL
    object
}

.nl_outer_hyper_draws <- function(fit, cells, within = NULL) {
    posterior <- fit$outer_posterior
    coupling <- posterior$partition
    local <- coupling$local
    read_cells <- integer(length(cells))
    groups <- split(seq_along(cells), cells)
    row_mass <- sum(coupling$row)
    for (key in names(groups)) {
        cell <- as.integer(key)
        ix <- groups[[key]]
        lo <- local@p[cell] + 1L
        hi <- local@p[cell + 1L]
        entries <- if (lo <= hi) seq.int(lo, hi) else integer(0)
        local_mass <- sum(local@x[entries])
        global_mass <- coupling$column[cell] * row_mass
        total <- local_mass + global_mass
        if (total <= 0) stop("Requested outer cell has zero posterior mass.", call. = FALSE)
        nearby <- stats::runif(length(ix)) * total < local_mass
        if (any(nearby)) {
            at <- sample.int(length(entries), sum(nearby), replace = TRUE,
                             prob = local@x[entries])
            read_cells[ix[nearby]] <- local@i[entries[at]] + 1L
        }
        if (any(!nearby)) {
            read_cells[ix[!nearby]] <- sample.int(length(coupling$row), sum(!nearby),
                replace = TRUE, prob = coupling$row)
        }
    }
    out <- tulpa_hyper_draws(.nl_outer_hyper_view(fit), cells = read_cells, within = within)
    attr(out, "cells") <- cells
    out
}
