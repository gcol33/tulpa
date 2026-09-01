# test-nested-laplace-joint-grid-fixed-parallel.R
# The retained per-cell fixed-effect pair indexes the same grid as the weights
# (gcol33/tulpa#345).
#
# `.nested_fixed_moments()` reads `$weights`, `$grid_modes` and
# `$grid_hessians` as three views of one grid and requires them the same
# length; a caller pairing `weights[k]` with `grid_modes[[k]]` makes the same
# assumption with no guard at all. `.joint_attach_grid_fixed()` built the pair
# by writing each cell's slot in turn, and a cell with no usable block wrote
# NULL -- which on an R list REMOVES the element rather than leaving it empty.
# An interior empty cell was invisible (the next write re-extended the list),
# a TRAILING one shortened the pair below the grid and took the whole
# coefficient table to NA with `grid_fixed_declined` reporting NA.
#
# Refinement is what makes the trailing case reachable: it appends cells at the
# end of the grid, and an appended cell whose inner solve returns a non-finite
# marginal carries zero weight and no block.

# A joint kernel result over `n_grid` cells with `p` fixed effects, whose cells
# in `blank` returned no per-cell block. `w` defaults to putting the mass on the
# cells that did.
.j345_res <- function(n_grid, p = 2L, blank = integer(0), w = NULL) {
  set.seed(3L)
  V <- lapply(seq_len(n_grid), function(k) {
    if (k %in% blank) return(NULL)
    A <- matrix(stats::rnorm(p * p), p, p)
    tcrossprod(A) + diag(p)
  })
  if (is.null(w)) {
    w <- rep(0, n_grid)
    live <- setdiff(seq_len(n_grid), blank)
    w[live] <- 1 / length(live)
  }
  list(cov_block_per_grid = V,
       modes   = matrix(stats::rnorm(n_grid * p), nrow = n_grid),
       weights = w)
}


# --------------------------------------------------------------------------- #
# 1. A blank cell leaves an empty slot, never a shorter list                   #
# --------------------------------------------------------------------------- #

test_that("a trailing cell with no block keeps the retained pair grid-length", {
  n <- 6L
  out <- tulpa:::.joint_attach_grid_fixed(.j345_res(n, blank = n), 2L)

  expect_length(out$grid_modes,    n)
  expect_length(out$grid_hessians, n)
  expect_length(out$weights,       n)
  expect_true(is.na(out$grid_fixed_declined))

  expect_null(out$grid_modes[[n]])
  expect_null(out$grid_hessians[[n]])
  expect_false(any(vapply(out$grid_modes[seq_len(n - 1L)], is.null, logical(1))))
})

test_that("an interior cell with no block leaves its own slot empty", {
  n <- 6L
  out <- tulpa:::.joint_attach_grid_fixed(.j345_res(n, blank = 3L), 2L)

  expect_length(out$grid_modes,    n)
  expect_length(out$grid_hessians, n)
  expect_null(out$grid_modes[[3L]])
  expect_null(out$grid_hessians[[3L]])
  expect_true(is.na(out$grid_fixed_declined))
})

test_that("a run of trailing blank cells is still grid-length", {
  n <- 6L
  out <- tulpa:::.joint_attach_grid_fixed(.j345_res(n, blank = c(2L, 5L, 6L)), 2L)
  expect_length(out$grid_modes,    n)
  expect_length(out$grid_hessians, n)
  expect_equal(which(vapply(out$grid_hessians, is.null, logical(1))),
               c(2L, 5L, 6L))
})

test_that("the retained slot holds the cell its own index names", {
  n <- 6L
  r <- .j345_res(n, blank = c(3L, 6L))
  out <- tulpa:::.joint_attach_grid_fixed(r, 2L)
  for (k in setdiff(seq_len(n), c(3L, 6L))) {
    expect_equal(out$grid_modes[[k]], as.numeric(r$modes[k, ]))
    expect_equal(solve(out$grid_hessians[[k]]), r$cov_block_per_grid[[k]])
  }
})


# --------------------------------------------------------------------------- #
# 2. The marginalizer reads such a grid, and reads it conditional on the       #
#    cells that carry a block (gcol33/tulpa#342)                               #
# --------------------------------------------------------------------------- #

test_that(".nested_fixed_moments reads a grid whose last cell carried no block", {
  n <- 6L
  out <- tulpa:::.joint_attach_grid_fixed(.j345_res(n, blank = n), 2L)
  mom <- tulpa:::.nested_fixed_moments(out)

  expect_false(is.null(mom))
  expect_true(all(is.finite(mom$mean)))
  expect_true(all(is.finite(mom$cov)))
  # The blank cell carried no weight, so nothing was dropped from the grid the
  # weights describe.
  expect_equal(mom$mass, 1)
  expect_equal(mom$keep, seq_len(n - 1L))
})

test_that("the blank-cell grid reports what a grid without that cell reports", {
  n <- 6L
  full <- tulpa:::.joint_attach_grid_fixed(.j345_res(n, blank = n), 2L)
  # The same grid with the zero-weight cell never present at all.
  r <- .j345_res(n, blank = n)
  r$cov_block_per_grid <- r$cov_block_per_grid[seq_len(n - 1L)]
  r$modes   <- r$modes[seq_len(n - 1L), , drop = FALSE]
  r$weights <- r$weights[seq_len(n - 1L)]
  short <- tulpa:::.joint_attach_grid_fixed(r, 2L)

  a <- tulpa:::.nested_fixed_moments(full)
  b <- tulpa:::.nested_fixed_moments(short)
  expect_equal(a$mean, b$mean)
  expect_equal(a$cov,  b$cov)
})


# --------------------------------------------------------------------------- #
# 3. Every path that leaves no coefficient table says why                      #
# --------------------------------------------------------------------------- #

test_that("a grid with no block on any weighted cell declines with a reason", {
  n <- 4L
  out <- tulpa:::.joint_attach_grid_fixed(.j345_res(n, blank = seq_len(n)), 2L)
  expect_identical(out$grid_fixed_declined, "no_weighted_cell_block")
  expect_null(out$grid_modes)
  expect_null(out$grid_hessians)
  expect_null(tulpa:::.nested_fixed_moments(out))
})

test_that("all-NA weights decline rather than retain an unreadable pair", {
  n <- 4L
  out <- tulpa:::.joint_attach_grid_fixed(
    .j345_res(n, w = rep(NA_real_, n)), 2L)
  expect_identical(out$grid_fixed_declined, "no_weighted_cell_block")
  expect_null(tulpa:::.nested_fixed_moments(out))
})

test_that("a weighted cell with no block is still fatal, with its own reason", {
  n <- 4L
  out <- tulpa:::.joint_attach_grid_fixed(
    .j345_res(n, blank = 2L, w = rep(0.25, n)), 2L)
  expect_identical(out$grid_fixed_declined, "cell_block_unavailable")
  expect_null(tulpa:::.nested_fixed_moments(out))
})

test_that("every decline reason a retained-pair-less result can carry is stated", {
  reasons <- c(
    tulpa:::.joint_attach_grid_fixed(.j345_res(4L), 0L)$grid_fixed_declined,
    tulpa:::.joint_attach_grid_fixed(
      list(cov_block_per_grid = NULL, modes = matrix(0, 4, 2),
           weights = rep(0.25, 4)), 2L)$grid_fixed_declined,
    tulpa:::.joint_attach_grid_fixed(
      utils::modifyList(.j345_res(4L), list(weights = rep(0.2, 5))),
      2L)$grid_fixed_declined,
    tulpa:::.joint_attach_grid_fixed(.j345_res(4L, blank = seq_len(4L)),
                                     2L)$grid_fixed_declined)
  expect_identical(reasons,
                   c("no_fixed_effects", "block_not_extracted",
                     "grid_misaligned", "no_weighted_cell_block"))
  expect_false(any(is.na(reasons)))
})


# --------------------------------------------------------------------------- #
# 4. End to end on a fit that refined its grid                                 #
# --------------------------------------------------------------------------- #

# The refinement fixture of test-nested-laplace-joint-adaptive-grid.R: an
# alpha_grid whose maximum sits below the truth, so the boundary carries mass
# and the pass appends cells past it.
.j345_refining_fit <- function(seed = 6L, N = 600L, n_s = 50L) {
  set.seed(seed)
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  n_nb <- vapply(nbr, length, integer(1))
  spatial_idx <- sample.int(n_s, N, replace = TRUE)
  rw    <- cumsum(stats::rnorm(n_s, 0, 1 / sqrt(n_s)))
  phi_s <- rw - mean(rw)
  Xocc  <- cbind(1, stats::rnorm(N))
  occur <- stats::rbinom(N, 1, stats::plogis(
    as.numeric(Xocc %*% c(-0.3, 0.5)) + phi_s[spatial_idx]))
  is_pos <- occur == 1L
  Xpos <- Xocc[is_pos, , drop = FALSE]; spi <- spatial_idx[is_pos]
  y_pos <- stats::rnorm(sum(is_pos),
                        as.numeric(Xpos %*% c(0.2, -0.4)) + 2.0 * phi_s[spi],
                        0.3)
  tulpa_nested_laplace_joint(
    responses = list(
      occ = list(y = as.numeric(occur), n_trials = rep(1L, N), X = Xocc,
                 spatial_idx = as.integer(spatial_idx), re_idx = rep(0, N),
                 n_re_groups = 0L, sigma_re = 1.0, family = "binomial",
                 phi = 1.0),
      pos = list(y = y_pos, n_trials = rep(1L, length(y_pos)), X = Xpos,
                 spatial_idx = as.integer(spi), re_idx = rep(0, length(y_pos)),
                 n_re_groups = 0L, sigma_re = 1.0, family = "gaussian",
                 phi = 0.09,
                 field_coef = list(name = "alpha", grid = c(0.2, 0.4, 0.6)))),
    prior = list(type = "icar", n_spatial_units = n_s,
                 adj_row_ptr = as.integer(c(0L, cumsum(n_nb))),
                 adj_col_idx = as.integer(unlist(nbr)) - 1L,
                 n_neighbors = as.integer(n_nb),
                 sigma_grid = c(0.6, 1.0, 1.5)),
    control = list(adaptive_grid = TRUE, diagnose_k = FALSE,
                   var_of_means_consistency = FALSE))
}

test_that("a refined joint grid keeps its weights and retained pair parallel", {
  skip_on_cran()
  fit <- .j345_refining_fit()
  expect_false(is.null(fit$adaptive_grid_info))
  expect_gt(sum(fit$adaptive_grid_info$n_points_added), 0L)

  n <- length(fit$weights)
  expect_equal(nrow(fit$theta_grid), n)
  expect_length(fit$grid_modes,    n)
  expect_length(fit$grid_hessians, n)
  expect_true(is.na(fit$grid_fixed_declined))

  expect_false(is.null(tulpa:::.nested_fixed_moments(fit)))
  ci <- stats::confint(fit)
  expect_true(all(is.finite(ci)))
  expect_true(all(is.finite(tulpa:::.fit_fixed_table(fit)$std.error)))
})

test_that("a refined grid carrying a non-finite cell still reports its table", {
  skip_on_cran()
  fit <- .j345_refining_fit()
  n <- length(fit$weights)
  # The reported case: one appended cell whose inner solve returned a
  # non-finite marginal, so it carries zero weight and no block. Replay it on
  # the settled grid.
  r <- list(cov_block_per_grid = lapply(fit$grid_hessians, function(H)
              if (is.null(H)) NULL else solve(H)),
            modes   = fit$modes,
            weights = fit$weights)
  r$cov_block_per_grid[[n]] <- NULL
  r$cov_block_per_grid <- c(r$cov_block_per_grid, list(NULL))[seq_len(n)]
  r$weights[n] <- 0
  r$weights <- r$weights / sum(r$weights)

  out <- tulpa:::.joint_attach_grid_fixed(r, fit$n_fixed)
  expect_length(out$grid_modes,    n)
  expect_length(out$grid_hessians, n)
  expect_true(is.na(out$grid_fixed_declined))
  expect_false(is.null(tulpa:::.nested_fixed_moments(out)))
})
