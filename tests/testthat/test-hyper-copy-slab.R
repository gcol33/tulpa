# Prior measure on the copy scale (`control$copy_slab`).
#
# The alpha axis defaults to the exponential penalized-complexity continuum plus
# a point mass at zero. `copy_slab = "flat"` instead gives it the measure the
# other log-scale axes carry: flat in log alpha over the span the declared nodes
# tile. These tests pin which spec fields each choice sets, the weights that
# follow, that the atom can be removed with `copy_atom_mass = 0`, and that an
# unrecognised value is rejected.

alpha_axis <- function(specs) {
  Filter(function(s) identical(s$name, "alpha"), specs)[[1L]]
}

# Geometric alpha grid: evenly spaced on the integration coordinate, so a flat
# measure there has to come back as equal weights.
GEO_ALPHA <- c(0, exp(seq(log(0.1), log(3), length.out = 5)))
CP <- list(has_copy = TRUE)

test_that("copy_slab = 'flat' is flat in log alpha over the declared span", {
  specs <- .joint_axis_specs(list(alpha = GEO_ALPHA), CP, copy_slab = "flat")
  ax <- alpha_axis(specs)

  # Same treatment as sigma / phi: a declared span, no declared density.
  expect_null(ax$slab_log_density)
  expect_false(is.null(ax$slab_bounds))
  pos <- GEO_ALPHA[GEO_ALPHA > 0]
  step <- log(pos[2L]) - log(pos[1L])
  expect_equal(log(ax$slab_bounds),
               c(log(pos[1L]) - step / 2, log(pos[length(pos)]) + step / 2))

  w <- .hyper_axis_level_weights(GEO_ALPHA, ax, ax$atom_mass)
  expect_equal(sum(w), 1)
  expect_equal(unname(w[1L]), .TULPA_COPY_ATOM_MASS)
  cont <- unname(w[-1L])
  expect_equal(cont, rep((1 - .TULPA_COPY_ATOM_MASS) / length(cont), length(cont)))
})

test_that("a flat copy slab splits weight on refinement instead of adding it", {
  specs <- .joint_axis_specs(list(alpha = GEO_ALPHA), CP, copy_slab = "flat")
  ax <- alpha_axis(specs)
  w0 <- .hyper_axis_level_weights(GEO_ALPHA, ax, 0)

  # A node inserted midway (in log) between the first two continuum nodes takes
  # half of each neighbour's cell, so the total over the pair is unchanged.
  pos <- GEO_ALPHA[GEO_ALPHA > 0]
  mid <- exp(mean(log(pos[1:2])))
  refined <- sort(c(GEO_ALPHA, mid))
  w1 <- .hyper_axis_level_weights(refined, ax, 0)

  expect_equal(sum(w1), 1)
  expect_equal(unname(w1[refined %in% pos[3:5]]), unname(w0[GEO_ALPHA %in% pos[3:5]]))
  expect_equal(sum(w1[refined <= pos[2L] & refined > 0]),
               sum(w0[GEO_ALPHA <= pos[2L] & GEO_ALPHA > 0]))
})

test_that("copy_slab = 'exponential' is the default and is unchanged by it", {
  base <- .joint_axis_specs(list(alpha = GEO_ALPHA), CP)
  named <- .joint_axis_specs(list(alpha = GEO_ALPHA), CP,
                             copy_slab = "exponential")
  expect_identical(
    .hyper_axis_level_weights(GEO_ALPHA, alpha_axis(base), .TULPA_COPY_ATOM_MASS),
    .hyper_axis_level_weights(GEO_ALPHA, alpha_axis(named), .TULPA_COPY_ATOM_MASS))

  ax <- alpha_axis(base)
  expect_null(ax$slab_bounds)
  expect_false(is.null(ax$slab_log_density))

  # The declared density, integrated on the log coordinate: rate put so that 5 %
  # of the prior sits above the largest declared node.
  pos <- GEO_ALPHA[GEO_ALPHA > 0]
  lambda <- -log(0.05) / max(pos)
  u <- log(pos)
  edges <- c(u[1L] - diff(u[1:2]) / 2, (u[-length(u)] + u[-1L]) / 2,
             u[length(u)] + diff(u[(length(u) - 1L):length(u)]) / 2)
  expected <- diff(edges) * lambda * exp(-lambda * pos) * pos
  w <- .hyper_axis_level_weights(GEO_ALPHA, ax, .TULPA_COPY_ATOM_MASS)
  expect_equal(unname(w[-1L]), (1 - .TULPA_COPY_ATOM_MASS) * expected)

  # Not flat: the continuum weights fall away as alpha grows.
  expect_gt(stats::sd(unname(w[-1L])), 0)
})

test_that("copy_atom_mass = 0 removes the point mass under either slab", {
  for (slab in c("exponential", "flat")) {
    specs <- .joint_axis_specs(list(alpha = GEO_ALPHA), CP, copy_atom_mass = 0,
                               copy_slab = slab)
    ax <- alpha_axis(specs)
    expect_identical(ax$atom_mass, 0)
    w <- .hyper_axis_level_weights(GEO_ALPHA, ax, 0)
    expect_equal(unname(w[1L]), 0)
    expect_gt(sum(w[-1L]), 0)

    # And the whole-grid log weight puts the zero cell at -Inf rather than NaN.
    tg <- cbind(alpha = GEO_ALPHA)
    lq <- .hyper_log_quad_weights(tg, specs)
    expect_identical(lq[1L], -Inf)
    expect_true(all(is.finite(lq[-1L])))
  }
})

test_that("copy_slab leaves the other log-scale axes alone", {
  grids <- list(sigma = c(0.5, 1, 2), alpha = GEO_ALPHA,
                phi_pos = c(0.2, 0.5, 0.9))
  for (slab in c("exponential", "flat")) {
    specs <- .joint_axis_specs(grids, CP, copy_slab = slab)
    for (nm in c("sigma", "phi_pos")) {
      ax <- Filter(function(s) identical(s$name, nm), specs)[[1L]]
      expect_null(ax$slab_log_density)
      expect_false(is.null(ax$slab_bounds))
      expect_null(ax$atom_mass)
    }
  }
})

test_that("an unrecognised copy_slab is rejected", {
  expect_error(.joint_axis_specs(list(alpha = GEO_ALPHA), CP, copy_slab = "pc"),
               "`copy_slab` must be one of")
  expect_error(.joint_axis_specs(list(alpha = GEO_ALPHA), CP, copy_slab = 1),
               "`copy_slab` must be one of")
  expect_error(.joint_axis_specs(list(alpha = GEO_ALPHA), CP,
                                 copy_slab = c("flat", "flat")),
               "`copy_slab` must be one of")
  expect_error(.joint_axis_specs(list(alpha = GEO_ALPHA), CP,
                                 copy_slab = NA_character_),
               "`copy_slab` must be one of")
})

test_that("a grid-derived spec list keeps the exponential default", {
  tg <- as.matrix(expand.grid(sigma = c(0.5, 1, 2), alpha = GEO_ALPHA))
  specs <- .joint_axis_specs_from_grid(tg)
  ax <- alpha_axis(specs)
  expect_null(ax$slab_bounds)
  expect_false(is.null(ax$slab_log_density))
  expect_identical(ax$atom_mass, .TULPA_COPY_ATOM_MASS)
})

test_that("control_check admits copy_slab on the joint method", {
  allowed <- .CONTROL_KEYS$nested_laplace_joint
  expect_true("copy_slab" %in% allowed)
  expect_silent(tulpa_check_control(list(copy_slab = "flat"), allowed,
                                    "tulpa_nested_laplace_joint"))
})
