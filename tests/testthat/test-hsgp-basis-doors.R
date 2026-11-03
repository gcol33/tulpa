# Every HSGP door sizes its basis by the one rule of R/hsgp_basis.R
# (Riutort-Mayol et al. 2023, Eqs. 19-24): a basis left unset is chosen from
# the lengthscale priors of the fields it carries, and refitted while the
# lengthscale posterior falls below what it represents. Every door's
# lengthscale prior is the one `.hsgp_lengthscale_prior()` states.

.bd_sites <- function(n = 60L, seed = 1L) {
  set.seed(seed)
  data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
}

test_that("spatial_svc(approx = 'hsgp') sizes its shared basis by the rule", {
  d <- .bd_sites()
  X <- model.matrix(~ x, d)
  v <- validate_svc(spatial_svc(~ lon + lat, terms = ~ x - 1, approx = "hsgp"),
                    d, X)
  S <- tulpa:::.hsgp_half_widths(v$coords_matrix)
  ell0 <- tulpa:::.hsgp_prior_lengthscale(v$coords_matrix)
  sz <- tulpa:::.hsgp_basis_size(S, ell0)
  expect_identical(v$m, sz$m)
  expect_equal(v$c_boundary, sz$c)
  expect_identical(v$basis_rule$history$source, "prior")
  expect_identical(v$basis_rule$c_key, "c_boundary")

  given <- validate_svc(spatial_svc(~ lon + lat, terms = ~ x - 1,
                                    approx = "hsgp", m = 7, c_boundary = 2),
                        d, X)
  expect_identical(given$m, 7L)
  expect_identical(given$c_boundary, 2)
  expect_false(any(given$basis_rule$auto))
  expect_error(spatial_svc(~ lon + lat, approx = "hsgp", m = 2), "between 3")
  expect_error(spatial_svc(~ lon + lat, approx = "hsgp", c_boundary = 0.5),
               ">= 1")
})

test_that("spatial_multiscale(approx = 'hsgp') sizes each scale's basis from its own prior", {
  d <- .bd_sites()
  v <- suppressMessages(validate_gp(
    spatial_multiscale(~ lon + lat, approx = "hsgp",
                       range_local = c(0.3, 1), range_regional = c(1, 10)), d))
  med <- function(U) tulpa:::.pc_range_quantile(0.5, U, tulpa:::.MSGP_RANGE_ALPHA, 2)
  S <- tulpa:::.hsgp_half_widths(v$coords_matrix)
  for (sc in c("local", "regional")) {
    U <- c(local = 0.3, regional = 1)[[sc]]
    sz <- tulpa:::.hsgp_basis_size(S, med(U))
    b <- v$scale_basis[[sc]]
    expect_identical(b$m, sz$m)
    expect_equal(b[["c"]], sz$c)
    expect_identical(tulpa:::.hsgp_block_basis(v, sc),
                     list(m = sz$m, c = sz$c, kernel = "matern32"))
  }
  expect_identical(v$scale_basis$local$basis_rule$axes, "b1.lengthscale")
  expect_identical(v$scale_basis$regional$basis_rule$axes, "b2.lengthscale")
  # The nested blocks are built on their own scale's basis.
  blks <- tulpa:::.multiscale_nl_blocks(v)
  expect_equal(ncol(blks[[1]]$phi_basis), v$scale_basis$local$m^2)
  expect_equal(ncol(blks[[2]]$phi_basis), v$scale_basis$regional$m^2)

  given <- suppressMessages(validate_gp(
    spatial_multiscale(~ lon + lat, approx = "hsgp", m = 7, c_boundary = 2,
                       range_local = c(0.3, 1), range_regional = c(1, 10)), d))
  expect_identical(tulpa:::.hsgp_block_basis(given, "regional"),
                   list(m = 7L, c = 2, kernel = "matern32"))
  expect_false(any(given$scale_basis$local$basis_rule$auto))
})

test_that("a refit step reads the smallest lengthscale of the fields on a basis", {
  d <- .bd_sites()
  sp <- validate_hsgp(spatial_gp(~ lon + lat, approx = "hsgp"), d)
  S <- sp$basis_rule$half_width
  win <- tulpa:::.hsgp_lengthscale_window(sp$m, sp[["c"]], S)
  local_mocked_bindings(
    .hsgp_lengthscale_posterior = function(fit, axes = NULL)
      rbind(c(mean = 0.8 * win[["lo"]], lo = 0, hi = 1),
            c(mean = 2 * win[["hi"]], lo = 0, hi = 1)),
    .package = "tulpa")
  step <- tulpa:::.hsgp_refit_spec(sp, list())
  expect_true(step$refit)
  expect_lte(tulpa:::.hsgp_lengthscale_window(step$spec$m, step$spec[["c"]],
                                              S)[["lo"]], 0.8 * win[["lo"]])
  expect_false(isTRUE(step$spec$basis_rule$capped))
})

test_that("the joint door's coordinate block sizes its basis and carries the lengthscale prior", {
  skip_on_cran()
  d <- .bd_sites(120L, 3L)
  d$y <- rpois(nrow(d), exp(0.3 + 0.4 * d$x + sin(5 * d$lon)))
  xy <- as.matrix(d[, c("lon", "lat")])
  fit <- suppressMessages(tulpa_nested_laplace_joint(
    responses = list(a = list(y = d$y, X = cbind(1, d$x), family = "poisson")),
    prior = list(list(type = "hsgp", coords = list(xy))),
    control = list(progress = FALSE)))
  hb <- fit$hsgp_basis
  expect_false(is.null(hb))
  expect_identical(hb$basis_rule$history$source[1L], "prior")
  expect_identical(fit$blocks[[1]]$basis$m, hb$m)
  expect_equal(fit$blocks[[1]]$basis$c, hb[["c"]])
  # The lengthscale axis carries the HSGP lengthscale prior rather than none.
  expect_false("b1.lengthscale" %in% names(fit$log_hyperprior_declined))
  expect_true("b1.lengthscale" %in% fit$log_hyperprior_axes)
  chk <- tulpa:::.hsgp_basis_check(fit)
  expect_true(chk$holds[["lower"]])
})

test_that("fit_st_nested() and the spec door size an unset HSGP basis by the rule", {
  skip_on_cran()
  set.seed(5)
  n_t <- 4L; N <- 160L
  xy <- cbind(runif(N), runif(N))
  tt <- sample(n_t, N, TRUE)
  x <- rnorm(N)
  y <- rpois(N, exp(0.2 + 0.4 * x + sin(4 * xy[, 1])))
  st <- suppressWarnings(fit_st_nested(y, cbind(1, x), NULL, NULL, tt, n_t,
                                       spatial_type = "hsgp",
                                       temporal_type = "rw1",
                                       family = "poisson", coords = xy))
  expect_false(is.null(st$hsgp_basis))
  S <- tulpa:::.hsgp_half_widths(xy)
  sz0 <- tulpa:::.hsgp_basis_size(S, tulpa:::.hsgp_prior_lengthscale(xy))
  expect_identical(st$hsgp_basis$basis_rule$history$m[1L], sz0$m)

  d <- data.frame(lon = xy[, 1], lat = xy[, 2], x = x, y = y)
  nl <- suppressWarnings(tulpa_nested_laplace(
    y = d$y, n_trials = rep(1L, N), X = cbind(1, d$x),
    spec = spatial_gp(~ lon + lat, approx = "hsgp"), data = d,
    family = "poisson"))
  expect_false(is.null(nl$hsgp_basis))
  expect_true(tulpa:::.hsgp_basis_check(nl)$holds[["lower"]])
})
