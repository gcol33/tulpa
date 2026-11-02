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

skip_on_cran_budget()

test_that("copy_slab = 'flat' is flat in log alpha over the declared span", {
  skip_on_cran_budget()
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
  skip_on_cran_budget()
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
  skip_on_cran_budget()
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
  # of the prior sits above the largest declared node. The density sets the
  # SHAPE of the continuum; how much of the axis the continuum holds is
  # `1 - atom_mass`, declared before the fit, so the shape is normalised.
  pos <- GEO_ALPHA[GEO_ALPHA > 0]
  lambda <- -log(0.05) / max(pos)
  u <- log(pos)
  edges <- c(u[1L] - diff(u[1:2]) / 2, (u[-length(u)] + u[-1L]) / 2,
             u[length(u)] + diff(u[(length(u) - 1L):length(u)]) / 2)
  expected <- diff(edges) * lambda * exp(-lambda * pos) * pos
  # The lowest cell reaches down to the atom: its mass is the exponential's
  # own probability of (0, upper edge).
  expected[1L] <- 1 - exp(-lambda * exp(edges[2L]))
  expected <- expected / sum(expected)
  w <- .hyper_axis_level_weights(GEO_ALPHA, ax, .TULPA_COPY_ATOM_MASS)
  expect_equal(unname(w[-1L]), (1 - .TULPA_COPY_ATOM_MASS) * expected)

  # Not flat: the continuum weights fall away as alpha grows.
  expect_gt(stats::sd(unname(w[-1L])), 0)
})

test_that("the declared atom mass is the split, whatever the nodes reach", {
  skip_on_cran_budget()
  # The exponential slab has support (0, Inf), so no node set reaches all of it.
  # Where the outermost node stops is a resolution choice; the prior probability
  # of "no coupling" is a modelling one, declared as `atom_mass`. Extending the
  # grid must therefore change the quadrature error, not the split
  # (gcol33/tulpa#626).
  for (grid in list(GEO_ALPHA,
                    c(0, exp(seq(log(0.1), log(3), length.out = 9))),
                    c(0, exp(seq(log(0.01), log(30), length.out = 7))))) {
    ax <- alpha_axis(.joint_axis_specs(list(alpha = grid), CP))
    w  <- .hyper_axis_level_weights(grid, ax, .TULPA_COPY_ATOM_MASS)
    expect_equal(sum(w), 1)
    expect_equal(unname(w[1L]), .TULPA_COPY_ATOM_MASS)
  }
})

test_that("copy_atom_mass = 0 removes the point mass under either slab", {
  skip_on_cran_budget()
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
  skip_on_cran_budget()
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
  skip_on_cran_budget()
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
  skip_on_cran_budget()
  tg <- as.matrix(expand.grid(sigma = c(0.5, 1, 2), alpha = GEO_ALPHA))
  specs <- .joint_axis_specs_from_grid(tg)
  ax <- alpha_axis(specs)
  expect_null(ax$slab_bounds)
  expect_false(is.null(ax$slab_log_density))
  expect_identical(ax$atom_mass, .TULPA_COPY_ATOM_MASS)
})

test_that("a column fixed across every cell is not an axis of the grid", {
  skip_on_cran_budget()
  tau <- exp(seq(log(0.1), log(5), length.out = 4L))
  bare  <- cbind(tau = tau)
  # A Poisson count grid pins the negative-binomial size at `Inf` so the kernel
  # takes its Poisson branch. That node is a setting, not a value to integrate.
  pinned <- cbind(tau = tau, r = Inf)
  expect_identical(.nl_grid_log_quad(pinned), .nl_grid_log_quad(bare))

  # A fixed finite column is inert for the same reason: it is the same factor in
  # every cell, so it cancels when the weights are normalised.
  held <- .nl_grid_log_quad(cbind(tau = tau, r = 7))
  expect_equal(held - held[1L],
               .nl_grid_log_quad(bare) - .nl_grid_log_quad(bare)[1L])

  # An axis that both varies and leaves the reals is still rejected.
  expect_error(.nl_grid_log_quad(cbind(tau = tau, r = c(1, 2, 3, Inf))),
               "must be a non-empty numeric vector of finite values")
})

test_that("control_check admits copy_slab on the joint method", {
  skip_on_cran_budget()
  allowed <- .CONTROL_KEYS$nested_laplace_joint
  expect_true("copy_slab" %in% allowed)
  expect_silent(tulpa_check_control(list(copy_slab = "flat"), allowed,
                                    "tulpa_nested_laplace_joint"))
})

test_that("the exponential continuum reaches down to its atom", {
  skip_on_cran_budget()
  ax  <- alpha_axis(.joint_axis_specs(list(alpha = GEO_ALPHA), CP))
  pos <- GEO_ALPHA[GEO_ALPHA > 0]
  lambda <- -log(0.05) / max(pos)
  u <- log(pos)
  hi <- u[length(u)] + diff(u[(length(u) - 1L):length(u)]) / 2
  # As an absolute prior mass the lowest continuum cell holds the exponential's
  # exact probability of (0, its upper edge): nothing below the lowest node is
  # left out. The cells above it are midpoint cells.
  w <- .hyper_axis_level_weights(GEO_ALPHA, ax, .TULPA_COPY_ATOM_MASS,
                                 absolute = TRUE)
  e1 <- exp((u[1L] + u[2L]) / 2)
  expect_equal(unname(w[2L]), (1 - .TULPA_COPY_ATOM_MASS) * (1 - exp(-lambda * e1)),
               tolerance = 1e-8)
  expect_equal(sum(w[-1L]), (1 - .TULPA_COPY_ATOM_MASS) * (1 - exp(-lambda * exp(hi))),
               tolerance = 0.02)
  # The reported support starts at zero.
  sup <- .hyper_axis_support(GEO_ALPHA, ax)
  expect_identical(sup[1L], 0)
  expect_equal(sup[2L], exp(hi))
  # A declared flat span keeps its own lower end.
  axf <- alpha_axis(.joint_axis_specs(list(alpha = GEO_ALPHA), CP, copy_slab = "flat"))
  expect_gt(.hyper_axis_support(GEO_ALPHA, axf)[1L], 0)
})

test_that("a continuum reaching its atom is partitioned from the atom up", {
  skip_on_cran_budget()
  v <- c(0.1, 0.2, 0.4)
  e <- .nl_box_edges(v, "positive_atom")
  expect_identical(e[1L], 0)
  expect_equal(e[-1L], .nl_box_edges(v, "positive")[-1L])
  # Its lowest box is uniform on the natural scale from the atom.
  q <- .nl_box_quantile(c(0, v), c(0.5, 0.3, 0.15, 0.05), c(0.5, 0.65),
                        domain = "positive_atom", atom = 0)
  expect_true(is.na(q$declined))
  expect_equal(q$q, c(0, 0.5 * sqrt(0.1 * 0.2)))
  # The log-quadratic read has no finite log width for that box and hands the
  # axis to the box read.
  rd <- .nl_lq_quantile(v, c(0.6, 0.3, 0.1), 0.5, domain = "positive_atom",
                        row_id = rep(1L, 3L), log_density = log(c(0.6, 0.3, 0.1)))
  expect_identical(rd$declined, "floor_cell")
})

test_that("a copy scale truly zero reads the atom mass a grid laid toward zero reads", {
  skip_if_not_slow()
  nr <- 8L; nc <- 8L; S <- nr * nc
  rc <- expand.grid(r = seq_len(nr), c = seq_len(nc))
  nb <- lapply(seq_len(S), function(s) which(abs(rc$r - rc$r[s]) + abs(rc$c - rc$c[s]) == 1))
  nn <- lengths(nb)
  icar <- list(type = "icar", n_spatial_units = S, adj_row_ptr = c(0L, cumsum(nn)),
               adj_col_idx = unlist(nb) - 1L, n_neighbors = nn)
  set.seed(1)
  w <- 0.8 * as.numeric(scale(sin(rc$r / 2) + cos(rc$c / 2.5)))
  s1 <- rep(seq_len(S), each = 3L); x1 <- rnorm(length(s1))
  y1 <- rpois(length(s1), exp(0.5 + 0.4 * x1 + w[s1]))
  visited <- sort(sample(S, 40L))
  s2 <- rep(visited, each = 4L); x2 <- rnorm(length(s2))
  counts <- list(y = y1, n_trials = rep(1L, length(y1)), X = cbind(1, x = x1),
                 spatial_idx = s1, family = "poisson")
  set.seed(4)
  det <- list(y = rbinom(length(s2), 10L, plogis(-0.3 + 0.8 * x2)), n_trials = rep(10L, length(s2)),
              X = cbind(1, x = x2), spatial_idx = s2, family = "binomial", field_coef = "alpha")
  ctl <- list(n_threads = 2L, progress = FALSE)
  fz <- tulpa_nested_laplace_joint(list(counts = counts, detect = det), prior = icar, control = ctl)
  expect_length(fz$outer_grid_railed_axes, 0L)
  sg <- sort(unique(tulpa_outer_grid(fz)$nodes[, "sigma"]))
  pr <- icar; pr$sigma_grid <- sg
  dz <- det; dz$field_coef <- list(name = "alpha",
                                   grid = c(0, exp(seq(log(1e-3), log(3), length.out = 29))))
  fd <- suppressWarnings(tulpa_nested_laplace_joint(
    list(counts = counts, detect = dz), prior = pr,
    control = c(ctl, list(axis_refine = "none", auto_recenter = FALSE, diagnose_k = FALSE))))
  # 0.900 against a reference of 0.828 when the continuum stopped at 0.081.
  expect_equal(fz$copy_atom$posterior_mass, fd$copy_atom$posterior_mass, tolerance = 0.01)
})
