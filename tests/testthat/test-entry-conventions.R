# Two grid entries fitting the same model report on one convention, and every
# grid entry reaches its driver through the shared bundle.
#
# gcol33/tulpa#698 -- the joint path's log-prior dropped the weak default
# fixed-effect prior while its own gradient and Hessian applied it, and the spec
# path included it. `log_marginal` therefore came back on two conventions
# depending on which entry family a field routed to -- icar / bym2 / car_proper
# / temporal take the spec path, nngp / hsgp / the five st_* take the joint one
# -- so a compare_models() or logLik() across them read a constant offset as
# evidence. It also left the joint objective missing a term its own gradient
# carried, which is the failure its informative-prior note describes.
# gcol33/tulpa#699 -- cpp_nested_laplace_spde hand-rolled its driver call, so
# control$prune / $prune_tol / $screen_iters / $fitted_var and the subspace
# debias were unreachable on the SPDE grid, and a knob added to the shared
# bundle would not have reached it either.

.ec_chain_adj <- function(S) {
  nbr <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
  nn <- vapply(nbr, length, integer(1))
  list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(nn))
}

test_that("the spec and joint entry families agree on log_marginal", {
  skip_on_cran()
  set.seed(11)
  S <- 12L
  adj <- .ec_chain_adj(S)
  n <- 120L
  site <- rep(seq_len(S), length.out = n)
  x <- rnorm(n)
  phi_true <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4))))
  y <- rpois(n, exp(0.3 + 0.5 * x + 0.4 * phi_true[site]))
  X <- cbind(`(Intercept)` = 1, x = x)

  sigma_grid <- c(0.5, 1.0, 1.5)

  # The same ICAR model at the same grid, through both entry families: the
  # single-block spec kernel and the one-arm joint driver.
  spec <- cpp_nested_laplace_icar(
    y = y, n = rep(1L, n), X = X,
    re_idx = rep(0, n), n_re_groups = 0L, sigma_re = 1,
    spatial_idx = site,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    n_spatial = S, tau_grid = 1 / sigma_grid^2,
    family = "poisson", phi = 1)

  joint <- tulpa_nested_laplace_joint(
    responses = list(a = list(y = y, X = X, family = "poisson",
                              spatial_idx = site)),
    prior = list(type = "icar", n_spatial_units = S,
                 adj_row_ptr = adj$adj_row_ptr,
                 adj_col_idx = adj$adj_col_idx,
                 n_neighbors = adj$n_neighbors,
                 sigma_grid = sigma_grid),
    control = list(diagnose_k = FALSE, diagnose_skew = FALSE))

  expect_true(all(is.finite(spec$log_marginal)))
  # Exactly, cell for cell: the joint path used to be offset by the density and
  # normalizer of the weak default beta prior at its own mode.
  expect_equal(joint$log_marginal, spec$log_marginal, tolerance = 1e-10)
})

test_that("every nested-Laplace grid entry takes the shared control knobs", {
  # The bundle exists so a knob added to a driver is added in one place. An
  # entry outside it silently ignores whatever the others gained.
  ns <- asNamespace("tulpa")
  entries <- grep("^cpp_nested_laplace_", ls(ns, all.names = TRUE), value = TRUE)
  # Not per-field grid entries: the two multi-block drivers and the batch one
  # take their control through their own list arguments, and the occupancy
  # likelihood is a coupled-fixture probe.
  entries <- setdiff(entries, c(
    "cpp_nested_laplace_multi", "cpp_nested_laplace_joint_multi",
    "cpp_nested_laplace_joint_multi_batch",
    "cpp_nested_laplace_test_occupancy_likelihood"))
  # icar / bym2 / car_proper / temporal / nngp / hsgp / the five st_* / spde.
  expect_identical(length(entries), 12L)

  for (e in entries) {
    fm <- names(formals(get(e, envir = ns)))
    for (k in c("prune_tol", "screen_iters", "debias", "cila")) {
      expect_true(k %in% fm, info = paste(e, "is missing", k))
    }
  }
})

test_that("fit_spde accepts the screening knobs its entry can now reach", {
  keys <- tulpa:::.CONTROL_KEYS$spde
  for (k in c("prune", "prune_tol", "prune_log_gap", "screen_iters",
              "fitted_var", "subspace_debias", "cila")) {
    expect_true(k %in% keys, info = k)
  }
  # And they are resolved the way tulpa_nested_laplace() resolves them, not
  # re-invented: an out-of-range tolerance is refused at the door.
  expect_error(tulpa:::.nl_check_prune_tol(1.5))
  expect_error(tulpa:::.nl_check_screen_iters(0L))
})
