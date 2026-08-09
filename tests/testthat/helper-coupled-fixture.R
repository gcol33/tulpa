# Scaffolding for the engine's own genuinely coupled likelihood: the two-arm
# occupancy mixture registered from src/ under the name
# "test_occupancy_mixture" (gcol33/tulpa#300).
#
# Cell c carries one row on the occupancy arm (eta_a = logit psi_c) and J rows
# on the detection arm (eta_v = logit p_cv, binary y_cv). Its density is
#
#   p_cell = psi_c prod_v Bern(y_cv | p_cv) + (1 - psi_c) 1{sum_v y_cv = 0},
#
# so a cell with a detection factorises into log psi + a per-visit sum, while a
# cell with none does not: there the occupancy state sits inside the same
# logarithm as every visit, and d^2 log p_cell / d eta_a d eta_v is nonzero.
# That is what makes the fixture an arbiter for the coupled paths rather than a
# separable model wearing a coupled label.
#
# `coupled_occ_log_post()` writes the SAME density in R. Its agreement with the
# compiled spec is asserted (not assumed) in
# test-cell-coupling-occupancy-mixture.R, which is what licenses
# `coupled_occ_quadrature()` to stand as an exact reference for the conditional
# posterior the engine's inner Laplace approximates.

# Install the spec into the process-global registry. Idempotent.
coupled_occ_register <- function() {
  cpp_register_test_occupancy_mixture_coupling()
  testthat::skip_if_not(
    cpp_cell_coupling_registry_has("test_occupancy_mixture"),
    "test_occupancy_mixture coupling spec not registered")
}

# Simulate one occupancy data set. `field` is an optional per-cell offset added
# to the occupancy logit, so the same builder serves the intercept-only
# reference model and the spatial recovery fit.
coupled_occ_data <- function(seed, n_cells, n_visits, b_occ, b_det,
                             field = NULL) {
  set.seed(seed)
  offset <- if (is.null(field)) rep(0, n_cells) else field
  z <- stats::rbinom(n_cells, 1L, stats::plogis(b_occ + offset))
  y <- as.numeric(stats::rbinom(n_cells * n_visits, 1L,
                                stats::plogis(b_det) * rep(z, each = n_visits)))
  cell_of_visit <- rep(seq_len(n_cells), each = n_visits)
  list(n_cells = n_cells, n_visits = n_visits,
       b_occ = b_occ, b_det = b_det, field = field,
       y_det = y, cell_of_visit = cell_of_visit,
       n_seen = as.integer(tapply(y, cell_of_visit, sum)),
       occupied = z)
}

# The two coupled arm specs. The occupancy arm's `y` is unused by the spec (the
# cell's detection history lives on the detection arm) and is carried as zeros.
coupled_occ_arms <- function(d, beta_prec = NULL, spatial_idx = NULL) {
  n_v <- d$n_cells * d$n_visits
  build <- function(y, N, map, sidx) {
    a <- list(y = y, n_trials = rep(1L, N), X = matrix(1, N, 1),
              family = "binomial", phi = 1,
              coupled = TRUE, cell_obs_map = map)
    if (!is.null(sidx))      a$spatial_idx     <- sidx
    if (!is.null(beta_prec)) a$beta_prior_prec <- beta_prec
    a
  }
  list(occ = build(rep(0, d$n_cells), d$n_cells, seq_len(d$n_cells),
                   if (is.null(spatial_idx)) NULL else seq_len(d$n_cells)),
       det = build(d$y_det, n_v, d$cell_of_visit,
                   if (is.null(spatial_idx)) NULL else rep(0L, n_v)))
}

# A latent block neither arm reaches. The conditional posterior then factorises
# into the (beta_occ, beta_det) target and an independent Gaussian, so a
# two-dimensional quadrature of that target is exact for both probed
# coordinates -- no truncation of a field the reference would otherwise have to
# integrate over.
coupled_occ_flat_prior <- function(d) {
  n_v <- d$n_cells * d$n_visits
  list(list(type = "iid", n_units = 1L, sigma_grid = 1.0,
            obs_idx = list(rep(0L, d$n_cells), rep(0L, n_v))))
}

# The intercept-only log posterior in (beta_occ, beta_det), vectorised
# elementwise so `outer()` evaluates the whole grid in one call. `beta_prec` is
# the Gaussian fixed-effect prior precision the arms carry, so the reference and
# the engine target the same density.
coupled_occ_log_post <- function(d, beta_prec) {
  n_seen_total <- sum(d$n_seen)
  n_det_cells  <- sum(d$n_seen > 0)
  n_dark_cells <- d$n_cells - n_det_cells
  J            <- d$n_visits
  function(a, b) {
    psi <- stats::plogis(a); p <- stats::plogis(b)
    n_det_cells * log(psi) +
      n_seen_total * log(p) + (n_det_cells * J - n_seen_total) * log1p(-p) +
      n_dark_cells * log(psi * (1 - p)^J + 1 - psi) -
      0.5 * beta_prec * a^2 - 0.5 * beta_prec * b^2
  }
}

# Central-difference negative Hessian of a two-dimensional log posterior at `z`,
# and the smallest eigenvalue of it. The mixture's dark-cell term
# log(psi (1-p)^J + 1 - psi) is not concave in (eta_occ, eta_det), so a data set
# with few detections makes that eigenvalue NEGATIVE at the Newton start z = 0:
# the inner Newton has no plain Cholesky factor there and must condition the
# Hessian to step at all (gcol33/tulpa#344). Written from the R log posterior, so
# a test can establish that a fixture reaches that regime without asking the
# engine.
coupled_occ_curvature <- function(log_post, z = c(0, 0), h = 1e-4) {
  f <- function(u) log_post(u[1], u[2])
  H <- matrix(0, 2, 2)
  H[1, 1] <- -(f(z + c(h, 0)) - 2 * f(z) + f(z - c(h, 0))) / h^2
  H[2, 2] <- -(f(z + c(0, h)) - 2 * f(z) + f(z - c(0, h))) / h^2
  H[1, 2] <- H[2, 1] <- -(f(z + c(h, h)) - f(z + c(h, -h)) -
                            f(z + c(-h, h)) + f(z - c(h, h))) / (4 * h^2)
  list(H = H, lambda_min = min(eigen(H, symmetric = TRUE,
                                     only.values = TRUE)$values))
}

# The mode of a two-dimensional log posterior, found independently of the engine:
# BFGS from the prior mean, restarted once. A prior-predictive draw can leave one
# arm weakly identified, so the search is a guarded descent rather than a raw
# Newton.
coupled_occ_ref_mode <- function(log_post, z0 = c(0, 0)) {
  nll <- function(z) {
    v <- log_post(z[1], z[2])
    if (is.finite(v)) -v else 1e300
  }
  o <- stats::optim(z0, nll, method = "BFGS",
                    control = list(reltol = 1e-14, maxit = 2000L))
  stats::optim(o$par, nll, method = "BFGS",
               control = list(reltol = 1e-14, maxit = 2000L))$par
}

# Exact marginal mean / sd / skewness of each coordinate of a two-dimensional
# log posterior, by direct quadrature on a regular grid. The same construction
# `.exact_intercept_skew()` in test-inner-skew.R applies to a scalar posterior,
# one dimension up: normalise on the grid, sum out the other coordinate, take
# the central moments of what is left. No Laplace approximation anywhere.
coupled_occ_quadrature <- function(log_post, center, half = 12,
                                   n_grid = 1301L) {
  ga <- seq(center[1] - half, center[1] + half, length.out = n_grid)
  gb <- seq(center[2] - half, center[2] + half, length.out = n_grid)
  lp <- outer(ga, gb, log_post)
  w  <- exp(lp - max(lp))
  w  <- w / sum(w)
  moments <- function(g, wg) {
    mu <- sum(g * wg)
    sd <- sqrt(sum((g - mu)^2 * wg))
    c(mean = mu, sd = sd, skew = sum((g - mu)^3 * wg) / sd^3)
  }
  list(a = moments(ga, rowSums(w)), b = moments(gb, colSums(w)))
}

for (.nm in c("coupled_occ_register", "coupled_occ_data", "coupled_occ_arms",
              "coupled_occ_flat_prior", "coupled_occ_log_post",
              "coupled_occ_curvature", "coupled_occ_ref_mode",
              "coupled_occ_quadrature")) {
  assign(.nm, get(.nm), envir = globalenv())
}
rm(.nm)
