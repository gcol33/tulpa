# spatial_range() / temporal_corr() must report the SAME interpretable quantity
# and row label on a nested-Laplace (Tier 2) fit as on a sampler (Tier 1) fit
# (#199). The nested path used to return the raw grid axes (tau, phi_gp, sigma2)
# with raw names; it now maps each axis to the sampler's interpretable quantity
# (sigma = 1/sqrt(tau) or sqrt(sigma2); range = 3 * lengthscale; rho identity),
# marginalizing the DERIVED quantity per grid cell.

rook_ct <- function(nr, nc) {
  n <- nr * nc; W <- matrix(0, n, n); id <- function(r, c) (c - 1) * nr + r
  for (r in seq_len(nr)) for (c in seq_len(nc)) {
    if (r < nr) { W[id(r, c), id(r + 1, c)] <- 1; W[id(r + 1, c), id(r, c)] <- 1 }
    if (c < nc) { W[id(r, c), id(r, c + 1)] <- 1; W[id(r, c + 1), id(r, c)] <- 1 }
  }
  W
}

nested_areal_fit <- function(type) {
  set.seed(9)
  nr <- nc <- 5L; S <- nr * nc; reps <- 4L; W <- rook_ct(nr, nc)
  unit <- rep(seq_len(S), each = reps); N <- length(unit)
  x <- rnorm(N); y <- rbinom(N, 3, plogis(-0.3 + 0.6 * x))
  idx <- tulpa:::.resolve_unit_index(factor(unit), "region", S)
  csr <- tulpa:::adjacency_to_csr_tulpa(W)
  prior <- list(type = type, spatial_idx = idx, n_spatial_units = S,
                adj_row_ptr = csr$row_ptr, adj_col_idx = csr$col_idx,
                n_neighbors = csr$n_neighbors)
  fit <- tulpa_nested_laplace(y = y, n_trials = rep(3L, N), X = cbind(1, x),
                              prior = prior, family = "binomial")
  fit$prior <- prior
  fit
}

test_that("nested spatial_range reports interpretable sigma / rho, marginalized (#199)", {
  skip_on_cran()
  # icar: single tau axis -> sigma = 1/sqrt(tau).
  f1 <- nested_areal_fit("icar")
  s1 <- spatial_range(f1)
  expect_identical(rownames(s1), "sigma")               # not the raw "tau"
  w1 <- f1$weights / sum(f1$weights); tau1 <- as.numeric(f1$theta_grid)
  expect_equal(s1["sigma", "mean"], sum(w1 * (1 / sqrt(tau1))))

  # car_proper: (tau, rho) -> (sigma, rho); rho passes through, sigma = 1/sqrt(tau).
  f2 <- nested_areal_fit("car_proper")
  s2 <- spatial_range(f2)
  expect_setequal(rownames(s2), c("sigma", "rho"))
  w2 <- f2$weights / sum(f2$weights)
  expect_equal(s2["sigma", "mean"], sum(w2 * (1 / sqrt(f2$theta_grid[, 1]))))
  expect_equal(s2["rho", "mean"],   sum(w2 * f2$theta_grid[, 2]))
})

test_that("the nested transform maps mirror the sampler's interpretable formulas (#199)", {
  sm <- tulpa:::.SPATIAL_HYPER_TRANSFORM
  expect_equal(sm$tau$fn(4), 0.5)         # 1/sqrt(4)  -> sigma
  expect_equal(sm$sigma2$fn(9), 3)        # sqrt(9)    -> sigma
  expect_equal(sm$phi_gp$fn(2), 6)        # 3 * 2      -> range
  expect_equal(sm$lengthscale$fn(2), 6)   # 3 * 2      -> range
  expect_identical(sm$tau$name, "sigma")
  expect_identical(sm$phi_gp$name, "range")

  tmn <- tulpa:::.TEMPORAL_HYPER_TRANSFORM
  expect_identical(tmn$tau$name, "precision")   # temporal reports tau as precision
})

test_that("nested temporal_corr maps its grid axes to the sampler labels (#199)", {
  ft <- structure(
    list(theta_grid = matrix(c(0.5, 1, 2), ncol = 1,
                             dimnames = list(NULL, "tau")),
         weights = c(0.2, 0.5, 0.3), theta_names = "tau",
         prior = list(type = "rw1")),
    class = c("tulpa_nested_laplace", "tulpa_fit"))
  tc <- temporal_corr(ft)
  expect_identical(rownames(tc), "precision")
  expect_equal(tc["precision", "mean"], sum(c(0.2, 0.5, 0.3) * c(0.5, 1, 2)))
})
