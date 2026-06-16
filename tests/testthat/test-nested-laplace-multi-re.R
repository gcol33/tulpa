# Multi-term random-effect re_list on a single nested-Laplace block
# (gcol33/tulpa#78). The first RE term rides the kernel's built-in iid slot; any
# further intercept terms become fixed-sigma (one-point grid) `iid` latent
# blocks, so [beta | RE_1 | <field> | RE_2 | ...] is fit jointly. Single-term
# models keep the built-in-only path unchanged; random slopes still decline.

make_block <- function(seed = 11) {
  set.seed(seed)
  n_units <- 20L
  W <- matrix(0, n_units, n_units)
  for (i in 1:(n_units - 1)) { W[i, i + 1] <- 1; W[i + 1, i] <- 1 }
  W[1, n_units] <- 1; W[n_units, 1] <- 1
  csr <- tulpa:::adjacency_to_csr_tulpa(W)
  n_site <- 12L; n_year <- 8L; N <- 600L
  spatial_idx <- sample.int(n_units, N, replace = TRUE)
  site <- sample.int(n_site, N, replace = TRUE)
  year <- sample.int(n_year, N, replace = TRUE)
  u_sp   <- as.numeric(scale(rnorm(n_units))) * 0.8
  b_site <- rnorm(n_site, 0, 0.7)
  b_year <- rnorm(n_year, 0, 0.5)
  eta <- 0.2 + u_sp[spatial_idx] + b_site[site] + b_year[year]
  y <- rbinom(N, 1, plogis(eta))
  list(
    block = list(
      y = as.numeric(y), n_trials = rep(1L, N), X = matrix(1, N, 1),
      family = "binomial",
      prior = list(type = "icar", spatial_idx = spatial_idx,
                   n_spatial_units = n_units, adj_row_ptr = csr$row_ptr,
                   adj_col_idx = csr$col_idx, n_neighbors = csr$n_neighbors),
      re_list = list(
        list(idx = site, n_groups = n_site, sigma = 0.7),
        list(idx = year, n_groups = n_year, sigma = 0.5)
      )
    ),
    truth = list(site = b_site, year = b_year, field = u_sp,
                 n_site = n_site, n_year = n_year, n_units = n_units)
  )
}

test_that("a two-term re_list on a nested block fits and recovers all effects", {
  skip_on_cran()
  s <- make_block()
  fit <- tulpa:::.fit_block_via_nested_laplace(s$block, n_threads = 1L)
  p <- 1L; n_site <- s$truth$n_site; n_units <- s$truth$n_units
  n_year <- s$truth$n_year
  # Layout: [beta | RE_1 (site) | field (icar) | RE_2 (year)].
  expect_equal(length(fit$mode), p + n_site + n_units + n_year)
  site_hat  <- fit$mode[p + seq_len(n_site)]
  field_hat <- fit$mode[p + n_site + seq_len(n_units)]
  year_hat  <- fit$mode[p + n_site + n_units + seq_len(n_year)]
  expect_gt(cor(s$truth$site, site_hat), 0.7)
  expect_gt(cor(s$truth$year, year_hat), 0.7)
  expect_gt(cor(s$truth$field, field_hat), 0.7)
})

test_that("single-term re_list is byte-identical to the built-in-only path", {
  skip_on_cran()
  s <- make_block()
  b1 <- s$block; b1$re_list <- s$block$re_list[1]
  fit_via_block <- tulpa:::.fit_block_via_nested_laplace(b1, n_threads = 1L)

  direct <- tulpa_nested_laplace(
    y = b1$y, n_trials = b1$n_trials, X = b1$X, prior = b1$prior,
    re_idx = b1$re_list[[1]]$idx, n_re_groups = b1$re_list[[1]]$n_groups,
    sigma_re = b1$re_list[[1]]$sigma, family = "binomial")
  w <- direct$weights %||% rep(1 / nrow(direct$modes), nrow(direct$modes))
  direct_mode <- as.numeric(crossprod(direct$modes, w))
  expect_equal(fit_via_block$mode, direct_mode, tolerance = 1e-10)
})

test_that("random slopes on a nested block still decline", {
  s <- make_block()
  b <- s$block
  b$re_list[[2]]$n_coefs <- 2L   # a slope term
  expect_error(
    tulpa:::.fit_block_via_nested_laplace(b, n_threads = 1L),
    "Random slopes"
  )
})
