library(tulpa)
spatial <- list(type = "icar", parameterization = "collapsed", n_spatial_units = 4L,
  spatial_idx = as.integer(rep_len(1:4, 16)), adj_row_ptr = c(0L, 1L, 3L, 5L, 6L),
  adj_col_idx = c(1L, 0L, 2L, 1L, 3L, 2L), n_neighbors = c(1L, 2L, 2L, 1L))
result <- tryCatch(tulpa:::tulpa_dispatch("hmc", fitter_args = list(
  y = rep(c(0, 1, 0, 2), 4), n_trials = rep(1L, 16), X = matrix(1, 16, 1),
  family = "poisson", backend = "hmc", spatial_spec = spatial,
  control = list(n_iter = 80L, warmup = 40L, n_chains = 1L,
                 seed = 1174L, max_treedepth = 3L))),
  error = function(e) conditionMessage(e))
stopifnot(identical(result,
  "Collapsed spatial inference requires double and arena likelihood weights and priors"))
print(result)
