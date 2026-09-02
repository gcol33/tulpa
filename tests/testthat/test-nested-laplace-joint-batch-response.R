# The batched CellResponse surface: per-species dispersion and trial counts.
#
# gcol33/tulpa#592 and gcol33/tulpa#591. The batched driver lays the per-arm
# per-species dispersion as `buf.phi[k * n_batch + s]` and its two readers
# unpack it independently -- the fused scatter re-keys it onto the coupled-arm
# index as `arm_phi_batch[kk * B + s]`, `species_cell_loglik` builds a
# length-n_coupled vector on a B = 1 view. Two re-keyings of one table, one on
# the curvature side of the fit and one on the objective side. Trial counts had
# a hole of a different kind: `BatchArmBuffers::n_trials` was never filled, so
# `CellResponse::n_trials()` dereferenced a null pointer on every batched fit.
#
# Neither could be seen from the suite, because `test_separable_bernoulli`,
# `test_bivariate_gaussian` and `test_occupancy_mixture` are all binomial with
# unit trials and none of them calls either accessor. `test_weighted_gaussian`
# is registered for exactly this: its density reads `phi(k, s)` and
# `n_trials(k, j)`, so a transposed index or an unfilled buffer changes an
# answer instead of changing nothing.
#
#   log p_cell = sum_j n_j * [ -0.5 log(2 pi phi_s) - 0.5 (y_j - eta_j)^2 / phi_s ]
#
# `n_j` enters as a replicate weight, so weight w at dispersion phi is exactly
# weight 1 at dispersion phi / w -- the trial-count read is pinned by an
# identity rather than by a tolerance.
#
# `phi` there is what reaches `CellResponse::phi()`, which `cell_coupling.h`
# documents as the gaussian SD and this fixture uses as its own density's
# variance. `phi_batch` at the R door is the residual VARIANCE and is converted
# at the C++ boundary, so a door value of `p^2` is what puts `p` in that slot
# (gcol33/tulpa#659: the identity was written before that convention and read
# `sqrt(p)` for two weeks without the suite being run).

skip_on_cran()

.wg_adj <- function(n_s) {
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  n_neighbors <- vapply(nbr, length, integer(1))
  list(adj_row_ptr     = as.integer(c(0L, cumsum(n_neighbors))),
       adj_col_idx     = as.integer(unlist(nbr)) - 1L,
       n_neighbors     = as.integer(n_neighbors),
       n_spatial_units = n_s)
}

# Two coupled gaussian arms over one shared design, B species. Two arms and
# B > 1 together are what make the [arm x species] dispersion layout
# arbitrable: with one arm, or with one species, a transposed index reads the
# same cell and no fixture can tell them apart.
.wg_sim <- function(seed = 592L, n_batch = 3L, n_s = 8L, n_per_unit = 4L) {
  set.seed(seed)
  N <- n_s * n_per_unit
  spatial_idx <- as.integer(rep(seq_len(n_s), each = n_per_unit))
  x <- stats::rnorm(N)
  X <- cbind(intercept = 1, x = x)
  draw <- function(mult) {
    matrix(vapply(seq_len(n_batch), function(s) {
      field <- stats::rnorm(n_s, 0, 0.5)
      as.numeric(mult * (0.2 * s) + 0.5 * x + field[spatial_idx] +
                   stats::rnorm(N, 0, 0.4))
    }, numeric(N)), nrow = N, ncol = n_batch)
  }
  list(n_s = n_s, N = N, n_batch = n_batch, X = X,
       spatial_idx = spatial_idx,
       y_a = draw(1), y_b = draw(-1), adj = .wg_adj(n_s))
}

.wg_arm <- function(sim, y, n_trials) {
  list(y = as.numeric(y), n_trials = as.integer(n_trials), X = sim$X,
       spatial_idx = sim$spatial_idx, family = "gaussian", phi = 1,
       coupled = TRUE, cell_obs_map = seq_len(sim$N))
}

.wg_fit <- function(sim, phi_batch, trials_a = 1L, trials_b = 1L,
                    sigma_grid = c(0.4, 0.8, 1.3)) {
  cpp_register_test_weighted_gaussian_coupling(c(0L, 1L))
  tulpa_nl_joint_batch(
    responses = list(a = .wg_arm(sim, sim$y_a[, 1L], rep(trials_a, sim$N)),
                     b = .wg_arm(sim, sim$y_b[, 1L], rep(trials_b, sim$N))),
    prior     = c(list(type = "icar", sigma_grid = sigma_grid,
                       spatial_idx = list(sim$spatial_idx, sim$spatial_idx)),
                  sim$adj),
    n_batch   = sim$n_batch,
    y_batch   = list(sim$y_a, sim$y_b),
    phi_batch = phi_batch,
    max_iter  = 80L, tol = 1e-10,
    cell_coupling = "test_weighted_gaussian",
    store_Q   = FALSE)
}

.wg_species <- function(fit, s) fit$per_species[[s]]

# --------------------------------------------------------------------------- #
# 1. Trial counts reach the spec at all, and are the shared design (#591)      #
# --------------------------------------------------------------------------- #

test_that("a batched spec can read n_trials without dereferencing null", {
  sim <- .wg_sim()
  fit <- .wg_fit(sim, matrix(1, nrow = 2L, ncol = sim$n_batch))
  for (s in seq_len(sim$n_batch)) {
    sp <- .wg_species(fit, s)
    expect_true(all(is.finite(sp$log_marginal)))
    expect_true(all(is.finite(sp$modes)))
  }
})

test_that("a trial count of w is a dispersion of phi / w, exactly", {
  sim <- .wg_sim()
  phi <- 0.7
  # The identity is a statement about the fixture's OWN dispersion, which is
  # whatever reaches `CellResponse::phi()` -- documented in `cell_coupling.h` as
  # the gaussian SD, and used as the variance of this fixture's own density. The
  # R door takes the residual VARIANCE and converts at the C++ boundary
  # (`.joint_phi_args_to_kernel()`), so the door value that puts `p` in that slot
  # is `p^2` (gcol33/tulpa#659). Stated on the door's own axis rather than
  # pre-squared numbers, so a further convention change moves one expression.
  # Stating it this way also pins the conversion ITSELF, which nothing else
  # did: the identity needs `slot(light) == slot(heavy) / w`, and only a LINEAR
  # door satisfies that from a squared argument. Drop the conversion and the
  # slots are `p^2` against `p^2 / 4`; make it a cube root and they are 0.788
  # against 0.497 where 0.394 is wanted. Either fails here.
  door <- function(p) p^2
  # Arm a at weight 2 and dispersion phi; arm b left at weight 1 so only one
  # arm's weighting moves and the identity is not a global rescale.
  heavy <- .wg_fit(sim, rbind(rep(door(phi), sim$n_batch),
                              rep(door(phi), sim$n_batch)),
                   trials_a = 2L, trials_b = 1L)
  light <- .wg_fit(sim, rbind(rep(door(phi / 2), sim$n_batch),
                              rep(door(phi), sim$n_batch)),
                   trials_a = 1L, trials_b = 1L)

  # The gradient and the curvature are w/phi either way, so the modes coincide.
  for (s in seq_len(sim$n_batch)) {
    expect_equal(.wg_species(heavy, s)$modes, .wg_species(light, s)$modes,
                 tolerance = 1e-9, info = paste("species", s))
  }
  # The normalizers differ by a constant per weighted row and nothing else, so
  # the log-marginal gap is the SAME at every outer-grid cell and species.
  gaps <- unlist(lapply(seq_len(sim$n_batch), function(s) {
    .wg_species(heavy, s)$log_marginal - .wg_species(light, s)$log_marginal
  }))
  expect_true(all(is.finite(gaps)))
  expect_lt(stats::sd(gaps), 1e-8)
  # And it is the constant the density implies: arm a has N rows, each
  # contributing 2 * (-0.5 log(2 pi phi)) against 1 * (-0.5 log(2 pi phi / 2)).
  expect_equal(mean(gaps),
               sim$N * (2 * (-0.5 * log(2 * pi * phi)) -
                          (-0.5 * log(2 * pi * phi / 2))),
               tolerance = 1e-7)
})

# --------------------------------------------------------------------------- #
# 2. The per-species dispersion reaches BOTH readers (#592)                    #
# --------------------------------------------------------------------------- #

test_that("moving one species' phi moves that species and no other", {
  sim <- .wg_sim()
  base <- matrix(1, nrow = 2L, ncol = sim$n_batch)
  moved <- base
  moved[1L, 2L] <- 2.5          # arm a (id 0), species 2

  f0 <- .wg_fit(sim, base)
  f1 <- .wg_fit(sim, moved)

  # Species 2 moves on BOTH sides of the fit: the log-marginal
  # (species_cell_loglik's read of the table) and the mode (the fused
  # scatter's). A re-keying slip in either one alone fails here.
  expect_false(isTRUE(all.equal(.wg_species(f0, 2L)$log_marginal,
                                .wg_species(f1, 2L)$log_marginal)))
  expect_false(isTRUE(all.equal(.wg_species(f0, 2L)$modes,
                                .wg_species(f1, 2L)$modes)))

  # The other species are bit-for-bit untouched -- their latent systems are
  # independent, so a table read one species off would show here.
  for (s in c(1L, 3L)) {
    expect_identical(.wg_species(f0, s)$log_marginal,
                     .wg_species(f1, s)$log_marginal, label = paste("species", s))
    expect_identical(.wg_species(f0, s)$modes,
                     .wg_species(f1, s)$modes, label = paste("species", s))
  }
})

test_that("the dispersion table is keyed [arm, species], not transposed", {
  # phi_batch is [n_arms x n_batch] = [2 x 3]. Moving entry (arm 1, species 3)
  # and entry (arm 2, species 1) are DIFFERENT fits; under a transposed read
  # they would key to each other's flat slot. Both arms have data and both are
  # coupled, so a slip cannot hide in a no-data arm.
  sim <- .wg_sim()
  mk <- function(k, s) { m <- matrix(1, nrow = 2L, ncol = sim$n_batch)
                         m[k, s] <- 3.0; m }
  a <- .wg_fit(sim, mk(1L, 3L))
  b <- .wg_fit(sim, mk(2L, 1L))

  # Under the correct keying a moves species 3 and b moves species 1.
  ref <- .wg_fit(sim, matrix(1, nrow = 2L, ncol = sim$n_batch))
  moved <- function(f) vapply(seq_len(sim$n_batch), function(s) {
    !isTRUE(all.equal(.wg_species(f, s)$log_marginal,
                      .wg_species(ref, s)$log_marginal))
  }, logical(1))
  expect_identical(moved(a), c(FALSE, FALSE, TRUE))
  expect_identical(moved(b), c(TRUE, FALSE, FALSE))
  # The two fits differ from each other, so the two entries are not aliases.
  expect_false(isTRUE(all.equal(.wg_species(a, 3L)$log_marginal,
                                .wg_species(b, 1L)$log_marginal)))
})

test_that("the coupled-arm index is not read as the arm id", {
  # Both arms carry the same design and the same trial counts, so the ONLY
  # thing distinguishing arm 1 from arm 2 in the dispersion read is the key.
  # Giving the two arms different phi and then swapping which arm gets which
  # must change the fit; if the readers collapsed arm to coupled index
  # incorrectly the two would coincide.
  sim <- .wg_sim()
  up   <- rbind(rep(0.5, sim$n_batch), rep(2.0, sim$n_batch))
  down <- rbind(rep(2.0, sim$n_batch), rep(0.5, sim$n_batch))
  fu <- .wg_fit(sim, up)
  fd <- .wg_fit(sim, down)
  for (s in seq_len(sim$n_batch)) {
    expect_false(isTRUE(all.equal(.wg_species(fu, s)$log_marginal,
                                  .wg_species(fd, s)$log_marginal)),
                 label = paste("species", s))
  }
})
