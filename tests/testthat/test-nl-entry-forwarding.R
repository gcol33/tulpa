# Every cpp_nested_laplace_* outer-grid entry forwards the shared response and
# control arguments it declares to its driver.
#
# The eleven entries hand the same twenty-one arguments to one of three drivers.
# An argument an entry DECLARES but drops on the way to the driver reaches R as
# a silently ignored setting: the fit runs, and it runs at the default iteration
# budget, tolerance or checkpoint. Nothing downstream reads the argument back,
# so a fit at the wrong budget is indistinguishable from a fit at the right one
# unless something asserts the setting changed the result.
#
# Three observable channels cover the twenty-one between them:
#
#   - Eight change what the RESULT carries: max_iter caps n_iter, x_init moves
#     it, store_Q / compute_skew / debias / cila each add their own fields, and
#     a positive prune_tol turns on the cheap-screen report that screen_iters
#     then moves.
#   - Twelve enter the CHECKPOINT FINGERPRINT (make_nl_grid_checkpoint): y, n,
#     X, re_idx, n_re_groups, sigma_re, family, phi, max_iter, tol, the offset,
#     and the grid axes. Writing a checkpoint and then re-running against the
#     same file with one of them perturbed must be refused, which is what says
#     the perturbed value reached the fingerprint.
#   - One, compute_fitted_var, REMOVES a result field: the per-row predictive
#     variance is filled by the LatentBlock driver, so the switch drops
#     `fitted_eta_var` at the entries that report one and is inert at the
#     entries whose driver never reported one.
#
# n_threads is the one shared argument with no observable: the drivers are
# required to return the same numbers at any thread count.

.nlf_grid_adj <- function(side) {
  g <- expand.grid(r = seq_len(side), c = seq_len(side))
  n_s <- nrow(g)
  nbr <- lapply(seq_len(n_s), function(i)
    which(abs(g$r - g$r[i]) + abs(g$c - g$c[i]) == 1))
  list(adj_row_ptr = as.integer(c(0L, cumsum(lengths(nbr)))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(lengths(nbr)),
       n_spatial_units = n_s)
}

.nlf_chain_adj <- function(n_s) {
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  list(adj_row_ptr = as.integer(c(0L, cumsum(lengths(nbr)))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(lengths(nbr)),
       n_spatial_units = n_s)
}

.nlf_nn <- function(coords, k) {
  n <- nrow(coords)
  ord <- order(coords[, 1], coords[, 2])
  co <- coords[ord, , drop = FALSE]
  nn_idx <- matrix(0L, n, k)
  nn_dist <- matrix(0, n, k)
  for (i in 2:n) {
    d <- sqrt((co[1:(i - 1), 1] - co[i, 1])^2 + (co[1:(i - 1), 2] - co[i, 2])^2)
    nc <- min(length(d), k)
    o <- order(d)[1:nc]
    nn_idx[i, seq_len(nc)] <- o
    nn_dist[i, seq_len(nc)] <- d[o]
  }
  list(coords = co, nn_idx = nn_idx, nn_dist = nn_dist,
       nn_order = as.integer(ord - 1L), nn = k)
}

# One case per grid entry: the function, a complete argument list, and the name
# of the grid axis to perturb when checking that the axes reach the fingerprint.
.nlf_cases <- function() {
  set.seed(603L)

  # --- areal (icar / bym2 / car_proper) -------------------------------------
  a <- .nlf_grid_adj(4L)
  Na <- 4L * a$n_spatial_units
  areal <- list(
    y = as.numeric(rpois(Na, 2)), n = rep(1L, Na),
    X = cbind(1, rnorm(Na)), re_idx = rep(0, Na),
    n_re_groups = 0L, sigma_re = 1,
    spatial_idx = as.integer(rep(seq_len(a$n_spatial_units), each = 4L)),
    n_spatial_units = a$n_spatial_units,
    adj_row_ptr = a$adj_row_ptr, adj_col_idx = a$adj_col_idx,
    n_neighbors = a$n_neighbors,
    family = "poisson", phi = 1, max_iter = 50L, tol = 1e-9, n_threads = 1L)

  # --- temporal --------------------------------------------------------------
  n_t <- 8L
  Nt <- 96L
  temporal <- list(
    y = as.numeric(rpois(Nt, 2)), n = rep(1L, Nt),
    X = cbind(1, rnorm(Nt)), re_idx = rep(0, Nt),
    n_re_groups = 0L, sigma_re = 1,
    temporal_idx = as.integer(sample(seq_len(n_t), Nt, replace = TRUE)),
    n_times = n_t, temporal_type = "ar1",
    tau_grid = c(1, 2), rho_grid = c(0.4, 0.6), cyclic = FALSE,
    family = "poisson", phi = 1, max_iter = 50L, tol = 1e-9, n_threads = 1L)

  # --- nngp (one observation per spatial unit) -------------------------------
  n_g <- 40L
  co <- cbind(runif(n_g), runif(n_g))
  nb <- .nlf_nn(co, 8L)
  nngp <- list(
    y = as.numeric(rbinom(n_g, 1L, 0.5)), n = rep(1L, n_g),
    X = matrix(1, n_g, 1), re_idx = rep(0, n_g),
    n_re_groups = 0L, sigma_re = 1,
    spatial_idx = seq_len(n_g), coords = nb$coords,
    nn_idx = nb$nn_idx, nn_dist = nb$nn_dist, nn_order = nb$nn_order,
    n_spatial = n_g, nn = nb$nn,
    sigma2_grid = c(0.5, 1), phi_gp_grid = c(0.3, 0.5), cov_type = 0L,
    family = "binomial", phi = 1, max_iter = 50L, tol = 1e-9, n_threads = 1L)

  # --- hsgp ------------------------------------------------------------------
  Nh <- 60L
  Mh <- 10L
  hsgp <- list(
    y = as.numeric(rbinom(Nh, 1L, 0.5)), n = rep(1L, Nh),
    X = cbind(1, rnorm(Nh)), re_idx = rep(0, Nh),
    n_re_groups = 0L, sigma_re = 1,
    phi_basis = matrix(rnorm(Nh * Mh), Nh, Mh),
    lambda_eig = sort(abs(rnorm(Mh)) + 0.1, decreasing = TRUE),
    sigma2_grid = c(0.5, 1), lengthscale_grid = c(0.5, 1),
    family = "binomial", phi = 1, max_iter = 50L, tol = 1e-9, n_threads = 1L)

  # --- spatio-temporal areal -------------------------------------------------
  sa <- .nlf_chain_adj(12L)
  Ns <- 150L
  nts <- 6L
  st_common <- list(
    y = as.numeric(rbinom(Ns, 1L, 0.5)), n = rep(1L, Ns),
    X = cbind(1, rnorm(Ns)), re_idx = rep(0, Ns),
    n_re_groups = 0L, sigma_re = 1,
    spatial_idx = as.integer(sample(seq_len(sa$n_spatial_units), Ns, replace = TRUE)),
    n_spatial_units = sa$n_spatial_units,
    adj_row_ptr = sa$adj_row_ptr, adj_col_idx = sa$adj_col_idx,
    n_neighbors = sa$n_neighbors,
    temporal_idx = as.integer(sample(seq_len(nts), Ns, replace = TRUE)),
    n_times = nts, temporal_type = "ar1",
    tau_temporal_grid = c(1, 1.5), rho_temporal_grid = c(0.5, 0.6),
    cyclic = FALSE,
    family = "binomial", phi = 1, max_iter = 50L, tol = 1e-9, n_threads = 1L)

  # --- spatio-temporal hsgp / nngp -------------------------------------------
  st_hsgp <- list(
    y = as.numeric(rbinom(Nh, 1L, 0.5)), n = rep(1L, Nh),
    X = cbind(1, rnorm(Nh)), re_idx = rep(0, Nh),
    n_re_groups = 0L, sigma_re = 1,
    phi_basis = matrix(rnorm(Nh * Mh), Nh, Mh),
    lambda_eig = sort(abs(rnorm(Mh)) + 0.1, decreasing = TRUE),
    temporal_idx = as.integer(sample(seq_len(nts), Nh, replace = TRUE)),
    n_times = nts,
    sigma2_spatial_grid = c(0.5, 1), lengthscale_spatial_grid = c(0.5, 1),
    temporal_type = "ar1",
    tau_temporal_grid = c(1, 1.5), rho_temporal_grid = c(0.5, 0.6),
    cyclic = FALSE,
    family = "binomial", phi = 1, max_iter = 50L, tol = 1e-9, n_threads = 1L)

  st_nngp <- list(
    y = as.numeric(rbinom(n_g, 1L, 0.5)), n = rep(1L, n_g),
    X = matrix(1, n_g, 1), re_idx = rep(0, n_g),
    n_re_groups = 0L, sigma_re = 1,
    spatial_idx = seq_len(n_g), n_spatial = n_g,
    coords = nb$coords, nn_idx = nb$nn_idx, nn_dist = nb$nn_dist,
    nn_order = nb$nn_order, nn = nb$nn, cov_type = 0L,
    temporal_idx = as.integer(sample(seq_len(nts), n_g, replace = TRUE)),
    n_times = nts,
    sigma2_spatial_grid = c(0.5, 1), phi_gp_spatial_grid = c(0.3, 0.5),
    temporal_type = "rw1",
    tau_temporal_grid = c(1, 2), rho_temporal_grid = NULL, cyclic = FALSE,
    family = "binomial", phi = 1, max_iter = 50L, tol = 1e-9, n_threads = 1L)

  list(
    list(name = "icar", fn = cpp_nested_laplace_icar,
         args = c(areal, list(tau_grid = c(0.5, 1, 2))),
         axis = "tau_grid"),
    list(name = "bym2", fn = cpp_nested_laplace_bym2,
         args = c(areal, list(scale_factor = 1,
                              sigma_spatial_grid = c(0.5, 1),
                              rho_grid = c(0.3, 0.7))),
         axis = "sigma_spatial_grid"),
    list(name = "car_proper", fn = cpp_nested_laplace_car_proper,
         args = c(areal, list(tau_grid = c(0.5, 1),
                              rho_grid = c(0.3, 0.7))),
         axis = "tau_grid"),
    list(name = "nngp", fn = cpp_nested_laplace_nngp,
         args = nngp, axis = "sigma2_grid"),
    list(name = "hsgp", fn = cpp_nested_laplace_hsgp,
         args = hsgp, axis = "sigma2_grid"),
    list(name = "temporal", fn = cpp_nested_laplace_temporal,
         args = temporal, axis = "tau_grid"),
    list(name = "st_icar", fn = cpp_nested_laplace_st_icar,
         args = c(st_common, list(tau_spatial_grid = c(1, 2))),
         axis = "tau_spatial_grid"),
    list(name = "st_car_proper", fn = cpp_nested_laplace_st_car_proper,
         args = c(st_common, list(tau_spatial_grid = c(1, 2),
                                  rho_spatial_grid = c(0.4, 0.6))),
         axis = "tau_spatial_grid"),
    list(name = "st_bym2", fn = cpp_nested_laplace_st_bym2,
         args = c(st_common, list(scale_factor = 1,
                                  sigma_spatial_grid = c(0.5, 1),
                                  rho_spatial_grid = c(0.3, 0.7))),
         axis = "sigma_spatial_grid"),
    list(name = "st_hsgp", fn = cpp_nested_laplace_st_hsgp,
         args = st_hsgp, axis = "sigma2_spatial_grid"),
    list(name = "st_nngp", fn = cpp_nested_laplace_st_nngp,
         args = st_nngp, axis = "sigma2_spatial_grid")
  )
}

for (.case in .nlf_cases()) local({
  case <- .case

  test_that(paste(case$name, "forwards the arguments the result reports"), {
    skip_on_cran()
    base <- do.call(case$fn, case$args)
    expect_true(all(is.finite(base$log_marginal)))
    expect_gt(max(base$n_iter), 1L)

    capped <- do.call(case$fn, modifyList(case$args, list(max_iter = 1L)))
    expect_equal(max(capped$n_iter), 1L)

    # Two solves capped at one Newton step, one from the origin and one from a
    # shifted start: a converged pair would land on the same mode either way,
    # so the cap is what leaves the starting point visible in the result.
    n_lat <- ncol(base$modes)
    warm <- do.call(case$fn, modifyList(
      case$args, list(max_iter = 1L, x_init_nullable = rep(0.75, n_lat))))
    expect_false(isTRUE(all.equal(as.numeric(warm$modes),
                                  as.numeric(capped$modes))))

    withQ <- do.call(case$fn, modifyList(case$args, list(store_Q = TRUE)))
    expect_false("Q_csc_x_per_grid" %in% names(base))
    expect_true("Q_csc_x_per_grid" %in% names(withQ))

    skewed <- do.call(case$fn, modifyList(case$args, list(compute_skew = TRUE)))
    expect_false("inner_skew" %in% names(base))
    expect_true("inner_skew" %in% names(skewed))

    deb <- do.call(case$fn, modifyList(
      case$args, list(debias = list(idx = 1L, n_draws = 20L))))
    expect_false("debias_idx" %in% names(base))
    expect_true("debias_idx" %in% names(deb))

    cil <- do.call(case$fn, modifyList(
      case$args, list(cila = list(n_points = 8L))))
    expect_false("cila_log_marginal" %in% names(base))
    expect_true("cila_log_marginal" %in% names(cil))
  })

  test_that(paste(case$name, "forwards the cheap-screen arguments"), {
    skip_on_cran()
    base <- do.call(case$fn, case$args)
    expect_false("prune_mask" %in% names(base))

    # A tolerance this far below any cell's normalised screening weight prunes
    # nothing, so what the presence of the report says is that the screen RAN.
    screened <- do.call(case$fn, modifyList(
      case$args, list(prune_tol = 1e-12)))
    expect_true("prune_mask" %in% names(screened))
    expect_true("prune_cheap_log_marginal" %in% names(screened))
    expect_equal(screened$prune_tol, 1e-12)
    expect_equal(length(screened$prune_cheap_log_marginal),
                 length(base$log_marginal))

    # The screen ranks cells by the Laplace log-marginal at the quasi-mode its
    # own step budget reaches, so a one-step screen and a converged one do not
    # report the same cheap log-marginals.
    shallow <- do.call(case$fn, modifyList(
      case$args, list(prune_tol = 1e-12, screen_iters = 1L)))
    deep <- do.call(case$fn, modifyList(
      case$args, list(prune_tol = 1e-12, screen_iters = 30L)))
    expect_false(isTRUE(all.equal(shallow$prune_cheap_log_marginal,
                                  deep$prune_cheap_log_marginal)))

    # The per-row predictive variance is the LatentBlock driver's; where an
    # entry reports one, switching it off removes it and moves nothing else.
    novar <- do.call(case$fn, modifyList(
      case$args, list(compute_fitted_var = FALSE)))
    expect_false("fitted_eta_var" %in% names(novar))
    expect_equal(novar$log_marginal, base$log_marginal)
  })

  test_that(paste(case$name, "fingerprints every shared response argument"), {
    skip_on_cran()
    path <- tempfile(fileext = ".ckpt")
    on.exit(unlink(path), add = TRUE)
    args <- modifyList(case$args, list(checkpoint_path = path))
    invisible(do.call(case$fn, args))
    expect_true(file.exists(path))
    expect_gt(file.size(path), 0)

    # Same inputs, same fingerprint: the second run resumes rather than erroring.
    expect_silent(invisible(do.call(case$fn, args)))

    axis <- case$args[[case$axis]]
    perturbed <- list(
      y           = list(y = rev(case$args$y)),
      n           = list(n = case$args$n + 1L),
      X           = list(X = case$args$X + 0.5),
      re_idx      = list(re_idx = case$args$re_idx + 0.5),
      sigma_re    = list(sigma_re = 2),
      family      = list(family = "gaussian"),
      phi         = list(phi = 2),
      max_iter    = list(max_iter = case$args$max_iter - 1L),
      tol         = list(tol = case$args$tol * 10),
      grid_axis   = setNames(list(axis + 0.125), case$axis)
    )
    for (nm in names(perturbed)) {
      expect_error(
        do.call(case$fn, modifyList(args, perturbed[[nm]])),
        "checkpoint",
        info = paste(case$name, "did not fingerprint", nm)
      )
    }
  })
})

test_that("compute_fitted_var is the switch on the driver that fills it", {
  skip_on_cran()
  cases <- .nlf_cases()
  named <- setNames(cases, vapply(cases, function(x) x$name, character(1)))

  # icar reaches run_multi_block_nested_laplace, the one driver that reports a
  # per-row predictive variance: it is there by default and gone when asked for.
  icar_on <- do.call(named$icar$fn, named$icar$args)
  expect_true("fitted_eta_var" %in% names(icar_on))
  expect_equal(dim(icar_on$fitted_eta_var),
               c(length(icar_on$log_marginal), length(named$icar$args$y)))
  icar_off <- do.call(named$icar$fn, modifyList(
    named$icar$args, list(compute_fitted_var = FALSE)))
  expect_false("fitted_eta_var" %in% names(icar_off))
  expect_equal(icar_off$log_marginal, icar_on$log_marginal)

  # nngp reaches the joint-sparse driver, which reports no per-row variance at
  # all, so the switch has nothing to remove there.
  nngp_on <- do.call(named$nngp$fn, named$nngp$args)
  expect_false("fitted_eta_var" %in% names(nngp_on))
})
