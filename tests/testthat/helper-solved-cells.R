# A nested-Laplace fit screens its outer grid by default: a cell the cheap pass
# drops is never solved, carries log_marginal = -Inf, and is flagged in
# `prune_mask`. "The fit solved its grid" is therefore a statement about the
# cells it solved, each of which has to be finite, while a dropped cell has to
# read exactly -Inf so it carries no weight.
cells_dropped <- function(fit) {
  as.logical(fit$prune_mask %||% logical(length(fit$log_marginal)))
}

expect_cells_solved <- function(fit, info = NULL) {
  lm <- fit$log_marginal
  dropped <- cells_dropped(fit)
  testthat::expect_length(dropped, length(lm))
  testthat::expect_true(all(is.finite(lm[!dropped])), info = info)
  testthat::expect_true(all(lm[dropped] == -Inf), info = info)
}

# The weight-averaged fixed effects `idx` over a fit's outer grid, read off the
# per-cell modes. A dropped cell holds no mode and no weight, so the average
# runs over the cells that carry weight.
grid_mean_fixed <- function(fit, idx, grid_modes = fit$grid_modes) {
  keep <- fit$weights > 0
  bm <- do.call(rbind, lapply(grid_modes[keep], function(m) m[idx]))
  as.numeric(crossprod(fit$weights[keep], bm))
}
