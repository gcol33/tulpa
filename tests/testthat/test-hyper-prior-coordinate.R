# A declared hyperprior meets the outer grid in one coordinate, and a declared
# point mass keeps the prior probability it declares.
#
# Two statements, one rule. The nodes are a quadrature rule for a declared
# measure, so (1) a density declared on an axis's natural scale is carried to
# the coordinate the cell widths are measured on before it is weighted, and
# (2) that density describes the axis's continuum -- the zero level of an axis
# with an `atom_mass` is a point mass outside it, which is why its prior
# probability is declared rather than integrated.
#
# gcol33/tulpa#624 (a natural-coordinate density annihilated the atom through
# log(0)), #625 (the joint driver's natural-scale densities were integrated as
# if they were on the log coordinate), #626 (the density reweighted the atom
# against the continuum, so the declared split was not the integrated one).

ATOM_GRID <- c(0, 0.25, 0.5, 1, 2)

atom_spec <- function(coord, fn = function(x) stats::dexp(x, 1, log = TRUE),
                      atom = 0.5) {
  hyper_axis_spec("alpha", grid = ATOM_GRID, log_scale = TRUE,
                  bounds = c(0, Inf), atom_mass = atom,
                  log_prior = fn, log_prior_coord = coord)
}

# Prior weight per cell with a flat inner marginal: the measure the fit
# integrates, assembled exactly as the driver assembles it.
prior_weights <- function(spec) {
  tg <- matrix(spec$grid, ncol = 1L, dimnames = list(NULL, spec$name))
  hp <- .hyper_grid_make_hp_fn(list(spec))
  .nl_normalise_weights_safe(hp(tg), what = "test",
                             log_quad = .hyper_log_quad_weights(tg, list(spec)))
}

test_that("a declared point mass survives a natural-coordinate density", {
  # log(0) = -Inf is the change of variables of the continuum. The atom is not
  # a point of that continuum, so the term does not apply to it and the cell
  # keeps its declared mass rather than being given zero weight (#624).
  spec <- atom_spec("natural")
  w <- prior_weights(spec)
  expect_true(all(is.finite(w)))
  expect_equal(unname(w[1L]), 0.5)
})

test_that("a declared point mass is the split, whichever coordinate is used", {
  # Same axis, same declared split, two readings of the same density (#626).
  for (coord in c("integration", "natural")) {
    expect_equal(unname(prior_weights(atom_spec(coord))[1L]), 0.5)
  }
  for (a in c(0.1, 0.5, 0.9)) {
    expect_equal(unname(prior_weights(atom_spec("natural", atom = a))[1L]), a)
  }
})

test_that("the atom's share does not move with the density on the continuum", {
  # The density shapes the continuum; it does not decide how much of the axis
  # the continuum holds. Three densities disagreeing sharply at zero and in the
  # tail, one declared split.
  fns <- list(function(x) stats::dexp(x, 1, log = TRUE),
              function(x) stats::dexp(x, 8, log = TRUE),
              function(x) stats::dgamma(x, shape = 2, rate = 1, log = TRUE))
  shares <- vapply(fns, function(f)
    unname(prior_weights(atom_spec("natural", f))[1L]), numeric(1))
  expect_equal(shares, rep(0.5, 3L))

  # dgamma(shape = 2) is exactly zero at the origin. Read at the atom it would
  # annihilate it; read on the continuum it only tilts the shape -- x^2 exp(-x)
  # once carried to the log coordinate, so alpha = 1 outweighs alpha = 0.25.
  w <- prior_weights(atom_spec("natural", fns[[3L]]))
  expect_true(all(w[-1L] > 0))
  expect_gt(w[4L], w[2L])
})

test_that("an axis with no point mass is untouched by the atom rule", {
  # Nothing to split against, so the density is read at every node and the
  # weights are the declared prior over the span, unchanged.
  x <- c(0.25, 0.5, 1, 2)
  spec <- hyper_axis_spec("sigma", grid = x, log_scale = TRUE,
                          bounds = c(0, Inf),
                          log_prior = function(v) stats::dexp(v, 1, log = TRUE),
                          log_prior_coord = "natural")
  w <- prior_weights(spec)
  expect_equal(sum(w), 1)
  # Log-spaced nodes, so the cell widths are equal and the shape is the density
  # carried to the log coordinate: p(x) x, peaking at x = 1.
  expect_equal(unname(w), unname((x * stats::dexp(x, 1)) /
                                 sum(x * stats::dexp(x, 1))))

  # The coordinate is a statement, not a formality: read as a density on log x
  # the same function weights the same nodes differently.
  spec$log_prior_coord <- "integration"
  expect_false(isTRUE(all.equal(unname(w), unname(prior_weights(spec)))))
})

test_that("the joint driver integrates the prior it documents (#625)", {
  # Arbiter: the closed-form probability the declared PC prior puts on each of
  # the shipped log-spaced cells. `pi(theta) = lambda exp(-lambda theta)` is a
  # density on sigma, so integrating it against widths measured in log sigma
  # needs the change of variables; without it the realised prior is pi/theta.
  sig <- exp(seq(log(0.05), log(5), length.out = 9L))
  tg  <- matrix(sig, ncol = 1L, dimnames = list(NULL, "sigma"))
  U <- 2; a <- 0.05
  lambda <- -log(a) / U
  fn <- .joint_parse_sigma_prior(list("pc.prec", c(U, a)), "prior_sigma")

  specs <- .joint_axis_specs_from_grid(tg)
  w <- .nl_normalise_weights_safe(
    .joint_hp_vec_for_grids(tg, fn, NULL, NULL), what = "test",
    log_quad = .hyper_log_quad_weights(tg, specs))

  bd <- specs[[1L]]$slab_bounds
  u  <- log(sig)
  ed <- exp(c(log(bd[1L]), (u[-9L] + u[-1L]) / 2, log(bd[2L])))
  Z  <- stats::pexp(bd[2L], lambda) - stats::pexp(bd[1L], lambda)
  exact <- (stats::pexp(ed[-1L], lambda) - stats::pexp(ed[-10L], lambda)) / Z

  # Quadrature error on 9 nodes, not a change of measure: the pre-fix path put
  # 22.7 % of the mass on the smallest node where the prior puts 4.3 %.
  expect_lt(max(abs(w - exact)), 0.01)

  # And it is the generic path's answer, not a second reading of the same rule.
  gen_spec <- hyper_axis_spec("sigma", grid = sig, log_scale = TRUE,
                              bounds = c(0, Inf),
                              slab_bounds = specs[[1L]]$slab_bounds,
                              log_prior = fn, log_prior_coord = "natural")
  expect_equal(w, prior_weights(gen_spec))
})

test_that("the joint copy scale keeps its declared atom mass under a prior", {
  # The shipped axis that carries both: `alpha` declares a point mass at zero
  # and takes `prior_alpha`. The density is read on the continuum, and the
  # atom is weighed on the continuum's scale, so the declared split stands.
  tg <- matrix(ATOM_GRID, ncol = 1L, dimnames = list(NULL, "alpha"))
  fn <- .joint_parse_sigma_prior(list("pc.prec", c(1, 0.05)), "prior_alpha")
  specs <- .joint_axis_specs(list(alpha = ATOM_GRID), list(has_copy = TRUE),
                            user_priors = list(alpha = fn))
  w <- .nl_normalise_weights_safe(
    .joint_hp_vec_for_grids(tg, NULL, fn, NULL), what = "test",
    log_quad = .hyper_log_quad_weights(tg, specs))
  expect_equal(unname(w[1L]), .TULPA_COPY_ATOM_MASS)
  expect_true(all(is.finite(w)))
})
