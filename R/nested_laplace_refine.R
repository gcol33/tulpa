# Per-cell results of outer cells solved in chunks, bound onto a base result.
#
# A pass that adds cells to a solved grid (an outer design's importance
# points, a quadrature's refinement) needs only each new cell's log marginal;
# the rest of what a cell carries (its mode, the precision, the Newton
# diagnostics, ...) is whatever the kernel returns for it, so the chunks are
# kept whole and bound onto the base result field by field
# (`.nl_bind_cell_results()`), rather than described by a table that would
# drift from the kernel's own output.

# The registry slots of a block list, each named as the grid names its axis.
.nl_registry_named_slots <- function(blocks) {
    slots <- .nl_registry_axis_slots(blocks)
    names(slots) <- vapply(slots, function(s) .axis_label(s$block, s$axis),
                           character(1))
    slots
}

# The refinement mode of each refinable axis of one registry grid, from who wrote
# its nodes (`.nl_slot_hold()` on the axis's slot, the rule the placement pass
# reads), `user` the caller's `control$axis_refine`.
.nl_registry_refine_modes <- function(axes, refinable, blocks, auto,
                                      user = NULL) {
    slots <- .nl_registry_named_slots(blocks)
    stated <- function(a) {
        s <- slots[[a]]
        if (is.null(s)) return(TRUE)
        !is.null(.nl_slot_hold(blocks, s, .nl_auto_fields_at(auto, s$block)))
    }
    .joint_axis_refine_modes_by(
        axes, stated, user,
        eligible = function(a) a %in% refinable)
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
