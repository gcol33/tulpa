# The registry arm's inner solve at given outer cells (`.nl_solve_cells()`, the
# joint driver): `prior` is a registry block or a block list, `theta` the cells
# with one column per block axis in block order. The flat hyperprior leaves the
# inner Laplace log marginal as the kernel returns it. `phi_scale = "kernel"`
# reads `phi` on the compiled convention (a gaussian SD) rather than the
# registry's (a variance).
.nl_cells <- function(y, X, prior, theta, n_trials = rep(1L, length(y)),
                      family = "binomial", phi = 1, re_idx = NULL,
                      n_re_groups = 0L, sigma_re = 1, offset = NULL,
                      max_iter = 50L, tol = 1e-8, n_threads = 1L,
                      x_init = NULL, store_Q = FALSE, force_sparse = FALSE,
                      compute_fitted_var = TRUE, hyperprior = "flat",
                      beta_prior = NULL, phi_scale = c("registry", "kernel"),
                      compute_skew = FALSE, skew_idx = NULL, debias = NULL,
                      cila = NULL) {
  phi_scale <- match.arg(phi_scale)
  blocks <- if (!is.null(prior$type)) list(prior) else prior
  a <- tulpa:::.nl_kernel_args(
    y = y, n_trials = n_trials, X = X, offset = offset,
    re_idx = re_idx %||% rep(0L, length(y)), n_re_groups = n_re_groups,
    sigma_re = sigma_re, family = family, phi = phi, max_iter = max_iter,
    tol = tol, n_threads = n_threads, x_init = x_init, store_Q = store_Q,
    fitted_var = compute_fitted_var, beta_prior = beta_prior,
    hyperprior = tulpa:::.hp_choice(hyperprior))
  if (phi_scale == "kernel") a$phi <- as.numeric(phi)
  tulpa:::.nl_solve_cells(a, blocks, as.matrix(theta),
                          compute_skew = compute_skew, skew_idx = skew_idx,
                          debias = debias, cila = cila,
                          force_sparse = force_sparse)
}

# An ICAR registry block over the CSR adjacency `adj` (make_grid_adjacency())
# for the observations of `dat` (simulate_spatial_data()).
.icar_blk <- function(adj, dat, n_units) {
  list(type = "icar", n_spatial_units = as.integer(n_units),
       spatial_idx = dat$spatial_idx, adj_row_ptr = adj$adj_row_ptr,
       adj_col_idx = adj$adj_col_idx, n_neighbors = adj$n_neighbors)
}

# A field described by a kernel-style argument list -- the adjacency / basis /
# neighbour fields with their paired per-axis grids, and for a spatio-temporal
# field `temporal_type`, `temporal_idx`, `n_times` and the temporal grids --
# solved at its cells by `.nl_cells()`. `type` is the spatial family ("icar",
# "car_proper", "bym2", "nngp", "hsgp"), NULL for a temporal block alone.
.nl_args_blocks <- function(type, a) {
  g <- function(...) { for (nm in c(...)) if (!is.null(a[[nm]])) return(a[[nm]]); NULL }
  areal <- function(t, ...) c(list(type = t, n_spatial_units = a[["n_spatial_units"]],
                                   spatial_idx = a[["spatial_idx"]],
                                   adj_row_ptr = a[["adj_row_ptr"]],
                                   adj_col_idx = a[["adj_col_idx"]],
                                   n_neighbors = a[["n_neighbors"]]), list(...))
  kern <- switch(as.character(a[["kernel"]] %||% 1L), "0" = list(cov = "squared_exponential"),
                 "1" = list(cov = "matern", nu = 1.5), "2" = list(cov = "matern", nu = 2.5))
  sp <- switch(type %||% "none",
    none = NULL,
    icar = list(areal("icar"), cbind(tau = g("tau_grid", "tau_spatial_grid"))),
    car_proper = list(areal("car_proper"),
                      cbind(tau = g("tau_grid", "tau_spatial_grid"),
                            rho = g("rho_grid", "rho_spatial_grid"))),
    bym2 = list(c(areal("bym2", scale_factor = a[["scale_factor"]] %||% 1),
                  if (!is.null(a[["node_prec"]])) list(node_prec = a[["node_prec"]])),
                cbind(sigma = g("sigma_spatial_grid"),
                      rho = g("rho_grid", "rho_spatial_grid"))),
    nngp = list(list(type = "nngp", coords = a[["coords"]], nn_idx = a[["nn_idx"]],
                     nn_dist = a[["nn_dist"]], nn_order = as.integer(a[["nn_order"]]) + 1L,
                     n_spatial = a[["n_spatial"]], nn = a[["nn"]],
                     cov_type = a[["cov_type"]] %||% 0L,
                     spatial_idx = a[["spatial_idx"]] %||% seq_len(a[["n_spatial"]])),
                cbind(sigma2 = g("sigma2_grid", "sigma2_spatial_grid"),
                      phi_gp = g("phi_gp_grid", "phi_gp_spatial_grid"))),
    hsgp = list(c(list(type = "hsgp", phi_basis = a[["phi_basis"]],
                       lambda_eig = a[["lambda_eig"]]), kern),
                cbind(sigma2 = g("sigma2_grid", "sigma2_spatial_grid"),
                      lengthscale = g("lengthscale_grid", "lengthscale_spatial_grid"))))
  blocks <- if (is.null(sp)) list() else list(sp[[1L]])
  theta <- if (is.null(sp)) NULL else sp[[2L]]
  if (!is.null(a[["temporal_type"]])) {
    tt <- a[["temporal_type"]]
    tb <- list(type = tt, temporal_idx = a[["temporal_idx"]], n_times = a[["n_times"]])
    if (tt %in% c("rw1", "rw2")) tb$cyclic <- isTRUE(a[["cyclic"]])
    if (!is.null(a[["n_groups"]])) tb$n_groups <- a[["n_groups"]]
    tg <- cbind(tau = g("tau_temporal_grid", "tau_grid"))
    if (tt == "ar1") tg <- cbind(tg, rho = g("rho_temporal_grid", "rho_grid"))
    blocks <- c(blocks, list(tb))
    theta <- if (is.null(theta)) tg else cbind(theta, tg)
  }
  list(blocks = blocks, theta = theta)
}

.nl_args_cells <- function(type, a, ...) {
  bt <- .nl_args_blocks(type, a)
  ng <- isTRUE(a[["n_re_groups"]] > 0)
  .nl_cells(a[["y"]], a[["X"]], bt$blocks, bt$theta, n_trials = a[["n"]],
            family = a[["family"]] %||% "binomial", phi = a[["phi"]] %||% 1,
            phi_scale = "kernel",
            re_idx = if (ng) a[["re_idx"]], n_re_groups = a[["n_re_groups"]] %||% 0L,
            sigma_re = a[["sigma_re"]] %||% 1, offset = a[["offset_nullable"]],
            max_iter = a[["max_iter"]] %||% 50L, tol = a[["tol"]] %||% 1e-6,
            x_init = a[["x_init_nullable"]], store_Q = isTRUE(a[["store_Q"]]),
            force_sparse = isTRUE(a[["force_sparse"]]),
            compute_fitted_var = a[["compute_fitted_var"]] %||% TRUE,
            beta_prior = a[["beta_prior"]],
            compute_skew = isTRUE(a[["compute_skew"]]), skew_idx = a[["skew_idx"]],
            debias = a[["debias"]], cila = a[["cila"]], ...)
}
