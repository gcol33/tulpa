# Outer-grid refinement on the registry door (`tulpa_nested_laplace()`).
#
# The joint drivers refine their outer grid through `.joint_refine_outer_grid()`
# (`R/nested_laplace_joint_helpers.R`): the opt-in boundary / interior pass and
# the var-of-means consistency pass, driven by the axis specs and a
# `kernel_fn(new_cells)` closure. The registry door runs the same two passes
# through the same runner. What it supplies is its own: the specs of the grid
# it solved, a `kernel_fn` over its dispatcher (`.nl_dispatch()` writes the
# cells onto the block's grid fields, `.nl_dispatch_multi()` takes them as a
# `theta_grid_override`), and the per-cell side data of every cell the passes
# add, bound onto the base result.
#
# The passes need only each new cell's log marginal, with the block
# hyperprior already folded in by the dispatcher. The rest of what a cell
# carries (its mode, the precision, the Newton diagnostics, ...) is whatever the
# kernel returns for it, so the chunks the passes solved are kept whole and
# bound onto the base result field by field (`.nl_bind_cell_results()`), rather
# than described by a table that would drift from the kernel's own output.

# The axes of a registry grid the passes may refine: a field SD (`sigma`, or the
# precision `tau`) the dispatcher can write a cell onto (`.nl_registry_axis_slots()`,
# the families `.NL_REGISTRY_AXIS_FIELD` binds). Named as the grid names them.
.nl_refinable_registry_axes <- function(prior, multi) {
    blocks <- if (isTRUE(multi)) prior else list(prior)
    slots <- .nl_registry_axis_slots(blocks)
    nm <- vapply(slots, function(s)
        if (isTRUE(multi)) paste0("b", s$block, ".", s$axis) else s$axis,
        character(1))
    nm[vapply(nm, .joint_axis_refine_eligible, logical(1),
              dispersions = FALSE)]
}

# The refinement mode of each refinable axis of one registry grid, from who wrote
# its nodes (`.nl_axis_hold()`, the rule the joint doors read), `user` the
# caller's `control$axis_refine`.
.nl_registry_refine_modes <- function(axes, refinable, prior, multi, auto,
                                      user = NULL) {
    type <- if (isTRUE(multi)) NULL else tolower(prior$type %||% "")
    stated <- function(a) {
        if (isTRUE(multi)) return(.joint_multi_axis_is_stated(a, prior, auto))
        !is.null(.nl_axis_hold(prior, paste0(a, "_grid"),
                               .nl_auto_fields_at(auto), type = type))
    }
    .joint_axis_refine_modes_by(
        axes, stated, user,
        eligible = function(a) a %in% refinable)
}

# One `kernel_fn(new_cells, warm_start, store_extras, screen)` for the passes,
# over `solve(theta_mat, screen)`: the dispatcher's result for those rows,
# screened against the grid they join when `screen` is given. Each call's whole
# result is kept in `chunks`, in the order the passes append their cells. A call
# is split where its size equals the base grid's, so that a field's length can
# tell a per-cell field from one that only happens to have as many entries as
# the grid has cells; each part takes its own cells' share of the screen.
.nl_chunked_kernel <- function(solve, n_base) {
    chunks <- list()
    kernel_fn <- function(new_cells, warm_start = NULL, store_extras = FALSE,
                          screen = NULL) {
        n <- nrow(new_cells)
        parts <- if (n == n_base && n > 1L) list(seq_len(n - 1L), n)
                 else list(seq_len(n))
        lm <- numeric(0)
        for (ix in parts) {
            scr <- if (!is.null(screen) && length(ix) > 1L)
                list(log_measure = screen$log_measure[ix],
                     log_ref = screen$log_ref)
            r <- solve(new_cells[ix, , drop = FALSE], scr)
            chunks[[length(chunks) + 1L]] <<- list(res = r, n = length(ix))
            lm <- c(lm, as.numeric(r$log_marginal))
        }
        list(log_marginal = lm, extras = NULL)
    }
    list(kernel_fn = kernel_fn, chunks = function() chunks)
}

# One field of a result as a per-cell array of `n` cells, or NULL when it is not
# one: a matrix with `n` rows, a vector or list of length `n`.
.nl_cell_shape <- function(v, n) {
    if (is.null(v)) return(NULL)
    if (is.matrix(v)) return(if (nrow(v) == n) "matrix")
    if (is.null(dim(v)) && length(v) == n) return(if (is.list(v)) "list" else "vector")
    NULL
}

# What fills the cells of a field a chunk did not return: a screened-out flag
# reads FALSE, a number NA, a list element NULL.
.nl_cell_fill <- function(v, kind, n) {
    switch(kind,
           matrix = matrix(NA, n, ncol(v)),
           list   = vector("list", n),
           vector = if (is.logical(v)) rep(FALSE, n) else rep(NA, n))
}

# The base result with the chunks' cells appended, field by field. A field is
# per-cell when the base result holds it for its `n_base` cells; it is a
# constant, and left alone, when a chunk returns it at the same length whatever
# the chunk's own size (`.nl_chunked_kernel()` keeps the two sizes apart). A
# per-cell field a chunk does not return -- the screen's own record, solved
# unscreened here -- is filled (`.nl_cell_fill()`).
.nl_bind_cell_results <- function(res, chunks, n_base) {
    for (f in names(res)) {
        v <- res[[f]]
        kind <- .nl_cell_shape(v, n_base)
        if (is.null(kind)) next
        constant <- any(vapply(chunks, function(ch)
            ch$n != n_base && !is.null(.nl_cell_shape(ch$res[[f]], n_base)),
            logical(1)))
        if (constant) next
        parts <- lapply(chunks, function(ch) {
            x <- ch$res[[f]]
            if (is.null(.nl_cell_shape(x, ch$n))) .nl_cell_fill(v, kind, ch$n)
            else x
        })
        res[[f]] <- switch(
            kind,
            matrix = do.call(rbind, c(list(v), parts)),
            list   = do.call(c, c(list(v), parts)),
            vector = do.call(c, c(list(v), lapply(parts, as.vector))))
    }
    res
}

# Refine a registry result's outer grid. `solve(theta_mat)` is the dispatcher's
# result for those rows; `prior` is the declared block (single) or block list
# (multi), `auto` the provenance record of the markers `.nl_grid_provenance()`
# stripped. Returns `res` unchanged when no axis is refinable, no pass is asked
# for or none added a node; otherwise the grid with the nodes appended, the
# per-cell side data bound, the `refining_axis` tags and the passes' records.
.nl_refine_registry <- function(res, solve, prior, multi, auto,
                                consistency = TRUE, adaptive_grid = FALSE,
                                edge_thresh = 0.02, max_passes = 1L,
                                user = NULL) {
    tg <- .nl_theta_matrix(res)
    if (is.null(tg) || is.null(colnames(tg))) return(res)
    refinable <- intersect(.nl_refinable_registry_axes(prior, multi),
                           colnames(tg))
    # A mis-named axis is refused whatever the grid turns out to need, so the
    # knob never reads as applied where it was not.
    user <- .joint_check_axis_refine(
        user, colnames(tg), eligible_fn = function(a) a %in% refinable)
    if ((!isTRUE(consistency) && !isTRUE(adaptive_grid)) || nrow(tg) < 2L ||
        !length(refinable)) return(res)
    modes <- .nl_registry_refine_modes(colnames(tg), refinable, prior, multi,
                                       auto, user)
    specs <- .joint_axis_specs_from_grid(
        tg, folded_axes = res$log_hyperprior_axes, axis_refine = modes)
    if (is.null(specs)) return(res)
    n_base <- nrow(tg)
    ck <- .nl_chunked_kernel(solve, n_base)
    ref <- .joint_refine_outer_grid(
        tg, as.numeric(res$log_marginal), NULL, specs, ck$kernel_fn, NULL,
        adaptive_grid = adaptive_grid, edge_thresh = edge_thresh,
        max_passes = max_passes, consistency = consistency)
    if (ref$n_added == 0L) {
        res$var_of_means_consistency_info <- ref$consistency_info
        return(res)
    }
    chunks <- ck$chunks()
    res <- .nl_bind_cell_results(res, chunks, n_base)
    res$theta_grid <- if (is.matrix(res$theta_grid)) ref$theta_grid
                      else as.numeric(ref$theta_grid[, 1L])
    res$log_marginal <- ref$log_marginal
    res$refining_axis <- ref$refining_axis
    if (!is.null(res$n_grid)) res$n_grid <- nrow(ref$theta_grid)
    res$adaptive_grid_info <- ref$adaptive_info
    res$var_of_means_consistency_info <- ref$consistency_info
    res
}

# The grid a re-dispatch over a settled registry fit solves (the subspace debias,
# the corrected integrated Laplace): the fit's own cells, refinement nodes
# included, as a block carrying them (single) or a `theta_grid_override`
# (multi). NULL where the fit refined nothing, which keeps the declared grid.
.nl_refined_theta <- function(res) {
    if (is.null(res$refining_axis) || !any(nzchar(res$refining_axis))) return(NULL)
    tg <- .nl_theta_matrix(res)
    if (is.null(colnames(tg))) NULL else tg
}

# `prior` (single block) written onto the fit's cells.
.nl_refined_block <- function(res, prior) {
    tg <- .nl_refined_theta(res)
    if (is.null(tg)) return(prior)
    .nl_registry_write_theta(list(prior), tg, colnames(tg))[[1L]]
}
