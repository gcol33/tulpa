# The reported interval of a bounded hyperparameter stays inside its support:
# an AR(1) autocorrelation on (-1, 1), a BYM2 mixing weight on (0, 1). The
# outer cells' mirrored edges are laid in the support's own unbounded
# coordinate (atanh for a correlation), so the 97.5% bound cannot pass 1 on a
# grid whose top node sits below it. A rw1 / rw2 temporal field carries no
# autocorrelation, so fit_st_nested() reports no rho axis for it.

.st_support_fixture <- function(seed = 1L) {
  set.seed(seed)
  nr <- 5L; nc <- 5L; S <- nr * nc; n_t <- 8L
  xy <- expand.grid(col = seq_len(nc), row = seq_len(nr))
  W <- 1 * (as.matrix(stats::dist(xy, method = "manhattan")) == 1)
  u <- as.numeric(scale(xy$col + 0.5 * xy$row)) * 0.6
  v <- as.numeric(scale(sin(2 * pi * seq_len(n_t) / n_t))) * 0.5
  d <- expand.grid(region = seq_len(S), time = seq_len(n_t))
  d$x <- stats::rnorm(nrow(d))
  d$y <- stats::rpois(nrow(d), exp(1.5 + 0.4 * d$x + u[d$region] + v[d$time]))
  list(d = d, W = W, n_t = n_t)
}

test_that("an ar1 autocorrelation axis is declared on the correlation support", {
  tg <- matrix(c(0.5, 1, 2, 0.1, 0.5, 0.9), ncol = 2L,
               dimnames = list(NULL, c("b1.tau", "b1.rho")))
  geo <- tulpa:::.joint_axis_geometry(list(
    theta_grid = tg, axis_offsets = c(0L, 2L),
    blocks = list(list(type = "ar1"))))
  expect_identical(geo$domain, c("positive", "correlation"))

  st <- matrix(1, 1L, 4L, dimnames = list(NULL, c(
    "sigma_spatial", "rho_spatial", "tau_temporal", "rho")))
  geo_st <- tulpa:::.joint_axis_geometry(list(theta_grid = st,
                                              prior = list(type = "st")))
  expect_identical(geo_st$domain, c("positive", "unit", "positive",
                                    "correlation"))

  # A proper-CAR `rho` has no support the registry names, and stays NA.
  car <- tulpa:::.joint_axis_geometry(list(
    theta_grid = matrix(1, 1L, 2L, dimnames = list(NULL, c("b1.tau", "b1.rho"))),
    axis_offsets = c(0L, 2L), blocks = list(list(type = "car_proper"))))
  expect_true(is.na(car$domain[2L]))
})

test_that("mirrored correlation edges stay inside (-1, 1)", {
  e <- tulpa:::.nl_cell_edges(c(0.1, 0.5, 0.9), "correlation")
  expect_true(all(abs(e) < 1))
  expect_equal(e[2L], tanh(atanh(0.9) + 0.5 * (atanh(0.9) - atanh(0.5))))
})

test_that("fit_st_nested ar1 and tulpa() bym2 + ar1 report rho inside (-1, 1)", {
  skip_if_fast()
  z <- .st_support_fixture()
  d <- z$d
  f <- fit_st_nested(d$y, cbind(1, d$x), d$region, z$W, d$time, z$n_t,
                     temporal_type = "ar1", family = "poisson",
                     control = list(n_threads = 1L))
  hp <- summary(f)$hyperparameters
  r <- hp[hp$parameter == "rho", ]
  expect_gt(r$ci_lo, -1)
  expect_lt(r$ci_hi, 1)
  expect_identical(unname(f$theta_cell_edge_coord[["rho"]]), "correlation")

  g <- tulpa(y ~ x + spatial(region), data = d, family = "poisson",
             spatial = list(type = "bym2", adjacency = z$W),
             temporal = temporal_ar1("time"), mode = "auto",
             control = list(n_threads = 1L, progress = FALSE))
  hg <- summary(g)$hyperparameters
  for (nm in c("b1.rho", "b2.rho")) {
    row <- hg[hg$parameter == nm, ]
    lo <- if (nm == "b1.rho") 0 else -1
    expect_gt(row$ci_lo, lo)
    expect_lt(row$ci_hi, 1)
  }
})

test_that("fit_st_nested rw1 / rw2 carry no temporal rho axis", {
  skip_if_fast()
  z <- .st_support_fixture()
  d <- z$d
  for (tt in c("rw1", "rw2")) {
    f <- fit_st_nested(d$y, cbind(1, d$x), d$region, z$W, d$time, z$n_t,
                       temporal_type = tt, family = "poisson",
                       control = list(n_threads = 1L))
    expect_false("rho" %in% colnames(f$theta_grid))
    expect_false("rho" %in% summary(f)$hyperparameters$parameter)
    expect_identical(sort(colnames(f$theta_grid)),
                     c("tau_spatial", "tau_temporal"))
  }
})

test_that("an ar1 rho carries the atanh proposal coordinate and its Jacobian", {
  tags <- tulpa:::.joint_pareto_block_tags("ar1", c("tau", "rho"))
  expect_identical(unname(tags), c("log", "atanh"))
  expect_true(is.na(tulpa:::.joint_pareto_block_tags("car_proper", "rho")))
  u <- c(-8, -1.3, 0, 0.4, 2.2, 9)
  iv <- tulpa:::.joint_pareto_inv("atanh", u)
  expect_equal(iv$theta, tanh(u))
  expect_equal(tulpa:::.joint_pareto_fwd("atanh", iv$theta[2:5]), u[2:5])
  # log |d rho / d u| = log(1 - tanh(u)^2), in a form that stays finite in the
  # tails where 1 - tanh(u)^2 rounds to 0.
  expect_equal(iv$log_jac[2:5], log(1 - tanh(u[2:5])^2), tolerance = 1e-12)
  expect_true(all(is.finite(iv$log_jac)))
  nodes <- tulpa:::.nl_recenter_axis_full("atanh", atanh(0.8), 0.3)$nodes
  expect_true(length(nodes) >= 2L && all(abs(nodes) < 1))
})
