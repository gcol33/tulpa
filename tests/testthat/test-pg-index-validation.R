# Rcpp vector and matrix subscripts are unchecked, so an R-supplied 1-based
# index converted to 0-based and used raw is a read or write past the end of an
# R allocation. Every Polya-Gamma entry point validates its index arguments once
# at entry; these are the checks, one per argument kind.

.pg_fixture <- function(N = 40L, J = 5L, G = 4L, seed = 3) {
  set.seed(seed)
  nb <- lapply(seq_len(J), function(j) {
    as.integer(setdiff(c(j - 1L, j + 1L), c(0L, J + 1L)))
  })
  list(
    y = rbinom(N, 1L, 0.4),
    n = rep(1L, N),
    X = cbind(1, rnorm(N)),
    re_group = as.integer(rep(seq_len(G), length.out = N)),
    n_re_groups = G,
    spatial_group = as.integer(rep(seq_len(J), length.out = N)),
    n_spatial_units = J,
    adj_list = nb,
    n_neighbors = as.integer(lengths(nb))
  )
}

.pg_spatial_call <- function(f, ...) {
  ov <- list(...)
  f[names(ov)] <- ov
  cpp_pg_binomial_gibbs_spatial(
    y = f$y, n = f$n, X = f$X,
    re_group = f$re_group, n_re_groups = f$n_re_groups,
    spatial_group = f$spatial_group, n_spatial_units = f$n_spatial_units,
    adj_list = f$adj_list, n_neighbors = f$n_neighbors,
    n_iter = 20L, n_warmup = 10L, thin = 1L, verbose = FALSE
  )
}

test_that("the fixture itself runs, so a rejection below is the guard talking", {
  expect_no_error(.pg_spatial_call(.pg_fixture()))
})

test_that("an out-of-range spatial index is rejected, not used as an offset", {
  f <- .pg_fixture()
  bad <- f$spatial_group; bad[7] <- f$n_spatial_units + 1L
  expect_error(.pg_spatial_call(f, spatial_group = bad),
               "spatial_group")

  zero <- f$spatial_group; zero[3] <- 0L
  expect_error(.pg_spatial_call(f, spatial_group = zero), "spatial_group")

  neg <- f$spatial_group; neg[1] <- -2L
  expect_error(.pg_spatial_call(f, spatial_group = neg), "spatial_group")

  na <- f$spatial_group; na[5] <- NA_integer_
  expect_error(.pg_spatial_call(f, spatial_group = na), "spatial_group")
})

test_that("a spatial index of the wrong length is rejected", {
  f <- .pg_fixture()
  expect_error(.pg_spatial_call(f, spatial_group = f$spatial_group[-1]),
               "length")
})

test_that("an out-of-range random-effect index is rejected", {
  f <- .pg_fixture()
  bad <- f$re_group; bad[2] <- f$n_re_groups + 3L
  expect_error(.pg_spatial_call(f, re_group = bad), "re_group")
  expect_error(.pg_spatial_call(f, re_group = f$re_group[-1]), "length")
})

test_that("a neighbour list that disagrees with n_neighbors is rejected", {
  # n_neighbors bounds the loop while adj_list supplies the entries, so a
  # disagreement reads past the end of the neighbour vector.
  f <- .pg_fixture()
  wrong <- f$n_neighbors; wrong[2] <- wrong[2] + 3L
  expect_error(.pg_spatial_call(f, n_neighbors = wrong), "neighbour|n_neighbors")

  short <- f$adj_list; short[[3]] <- integer(0)
  expect_error(.pg_spatial_call(f, adj_list = short), "neighbour|n_neighbors")
})

test_that("an out-of-range adjacency entry is rejected", {
  f <- .pg_fixture()
  bad <- f$adj_list
  bad[[1]] <- as.integer(f$n_spatial_units + 4L)
  expect_error(.pg_spatial_call(f, adj_list = bad, n_neighbors = as.integer(lengths(bad))),
               "adj_list")

  z <- f$adj_list
  z[[2]] <- c(0L, 3L)
  expect_error(.pg_spatial_call(f, adj_list = z, n_neighbors = as.integer(lengths(z))),
               "adj_list")
})

test_that("the adjacency length must match the declared unit count", {
  f <- .pg_fixture()
  expect_error(
    .pg_spatial_call(f, adj_list = f$adj_list[-1],
                     n_neighbors = f$n_neighbors[-1]),
    "adj_list|n_neighbors"
  )
})

test_that("a non-positive thinning or an inverted warmup is rejected", {
  f <- .pg_fixture()
  expect_error(
    cpp_pg_binomial_gibbs_spatial(
      y = f$y, n = f$n, X = f$X, re_group = f$re_group,
      n_re_groups = f$n_re_groups, spatial_group = f$spatial_group,
      n_spatial_units = f$n_spatial_units, adj_list = f$adj_list,
      n_neighbors = f$n_neighbors,
      n_iter = 20L, n_warmup = 10L, thin = 0L, verbose = FALSE),
    "thin")
  expect_error(
    cpp_pg_binomial_gibbs_spatial(
      y = f$y, n = f$n, X = f$X, re_group = f$re_group,
      n_re_groups = f$n_re_groups, spatial_group = f$spatial_group,
      n_spatial_units = f$n_spatial_units, adj_list = f$adj_list,
      n_neighbors = f$n_neighbors,
      n_iter = 5L, n_warmup = 10L, thin = 1L, verbose = FALSE),
    "n_iter")
})

test_that("the temporal kernel validates its time index", {
  f <- .pg_fixture()
  n_times <- 6L
  t_idx <- as.integer(rep(seq_len(n_times), length.out = length(f$y)))
  call_t <- function(idx) {
    cpp_pg_binomial_gibbs_temporal(
      y = f$y, n = f$n, X = f$X, re_group = f$re_group,
      n_re_groups = f$n_re_groups, time_idx = idx, n_times = n_times,
      seasonal_period = 0L, trend_type = 1L, short_type = 0L,
      n_iter = 20L, n_warmup = 10L, thin = 1L, verbose = FALSE)
  }
  expect_no_error(call_t(t_idx))
  bad <- t_idx; bad[4] <- n_times + 2L
  expect_error(call_t(bad), "time_idx")
  expect_error(call_t(t_idx[-1]), "length")
})

test_that("the negative-binomial spatial kernel validates the same arguments", {
  f <- .pg_fixture()
  y_ct <- rpois(length(f$y), 3)
  call_nb <- function(...) {
    ov <- list(...)
    a <- list(
      y = as.integer(y_ct), X = f$X,
      re_group = f$re_group, n_re_groups = f$n_re_groups,
      spatial_group = f$spatial_group, n_spatial_units = f$n_spatial_units,
      adj_list = f$adj_list, n_neighbors = f$n_neighbors,
      n_iter = 20L, n_warmup = 10L, thin = 1L,
      prior_beta_sd = 2.5, prior_sigma_re_scale = 2.5,
      prior_tau_shape = 1, prior_tau_rate = 1,
      prior_r_shape = 1, prior_r_rate = 1, r_init = 2,
      store_eta = FALSE, verbose = FALSE, n_threads = 1L)
    a[names(ov)] <- ov
    do.call(cpp_pg_negbin_gibbs_spatial, a)
  }
  expect_no_error(call_nb())
  bad <- f$spatial_group; bad[2] <- f$n_spatial_units + 1L
  expect_error(call_nb(spatial_group = bad), "spatial_group")
  bad_re <- f$re_group; bad_re[2] <- 0L
  expect_error(call_nb(re_group = bad_re), "re_group")
  wrong <- f$n_neighbors; wrong[1] <- wrong[1] + 2L
  expect_error(call_nb(n_neighbors = wrong), "neighbour|n_neighbors")
})

test_that("a design without an intercept column is rejected by the centring kernels", {
  # Centring a latent effect and adding the removed level to beta[0] leaves eta
  # unchanged only under an all-ones first column, so every kernel that centres
  # refuses a design without one rather than shifting the posterior silently.
  f <- .pg_fixture()
  N <- length(f$y)
  X_no_int <- cbind(rnorm(N), rnorm(N))
  X_empty <- matrix(numeric(0), nrow = N, ncol = 0)
  y_ct <- as.integer(rpois(N, 3))

  expect_error(.pg_spatial_call(f, X = X_no_int), "intercept")
  expect_error(.pg_spatial_call(f, X = X_empty), "intercept")

  call_nb_spatial <- function(X) {
    cpp_pg_negbin_gibbs_spatial(
      y = y_ct, X = X,
      re_group = f$re_group, n_re_groups = f$n_re_groups,
      spatial_group = f$spatial_group, n_spatial_units = f$n_spatial_units,
      adj_list = f$adj_list, n_neighbors = f$n_neighbors,
      n_iter = 20L, n_warmup = 10L, thin = 1L,
      prior_beta_sd = 2.5, prior_sigma_re_scale = 2.5,
      prior_tau_shape = 1, prior_tau_rate = 1,
      prior_r_shape = 1, prior_r_rate = 1, r_init = 2,
      store_eta = FALSE, verbose = FALSE, n_threads = 1L)
  }
  expect_error(call_nb_spatial(X_no_int), "intercept")
  expect_error(call_nb_spatial(X_empty), "intercept")

  call_nb <- function(X) {
    cpp_pg_negbin_gibbs(
      y = y_ct, X = X, group = f$re_group, n_groups = f$n_re_groups,
      n_iter = 20L, n_warmup = 10L, thin = 1L,
      prior_beta_sd = 2.5, prior_sigma_scale = 2.5,
      prior_r_shape = 1, prior_r_rate = 1, r_init = 2,
      store_eta = FALSE, verbose = FALSE, n_threads = 1L)
  }
  expect_no_error(call_nb(f$X))
  expect_error(call_nb(X_no_int), "intercept")
  expect_error(call_nb(X_empty), "intercept")
})
