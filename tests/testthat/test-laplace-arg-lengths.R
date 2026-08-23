# The dense Laplace entry points check the length of every R argument they
# index by another argument's length (gcol33/tulpa#469).
#
# `Rcpp::Vector::operator[]`, `Matrix::operator()` and `List::operator[]` are
# unchecked. Both entries size their loops from `y` (or from `K`, the RE term
# count read off `re_ngroups`) and then index arguments the caller supplies
# separately, so a short one is read past the end of its allocation. Nothing
# crashes: the read lands inside the R heap and returns a finite double, the
# solve converges, and the result carries no sign of which numbers came from
# the data -- the same failure mode as the NNGP coordinate read in
# gcol33/tulpa#389.
#
# `cpp_laplace_fit` reaches its checks through `as_re_group_vec` and
# `build_spec_family_inputs`; `cpp_laplace_fit_multi_re` marshals ModelData by
# hand and carries its own. Both use the one `check_arg_length`, so the
# assertions below are on the message it raises.

skip_on_cran()

.lal_data <- function(N = 120L, n_re = c(12L, 8L), seed = 469L) {
  set.seed(seed)
  X <- cbind(1, rnorm(N))
  idx <- lapply(n_re, function(g) sample.int(g, N, replace = TRUE))
  eta <- as.numeric(X %*% c(0.3, -0.4))
  for (k in seq_along(idx)) eta <- eta + rnorm(n_re[k], sd = 0.4)[idx[[k]]]
  list(N = N, p = ncol(X), X = X, y = as.numeric(rpois(N, exp(eta))),
       n = rep(1L, N), idx = lapply(idx, as.integer), n_re = as.integer(n_re))
}

.lal_multi_args <- function(d) {
  list(y = d$y, n = d$n, X = d$X,
       re_idx_list = d$idx,
       re_ngroups = d$n_re,
       re_sigma_list = as.list(rep(0.5, length(d$n_re))),
       family = "poisson", phi = 1.0,
       max_iter = 20L, tol = 1e-6, n_threads = 1L)
}

test_that("cpp_laplace_fit_multi_re rejects every short argument it indexes", {
  d <- .lal_data()
  args <- .lal_multi_args(d)
  base <- do.call(tulpa:::cpp_laplace_fit_multi_re, args)
  expect_true(all(is.finite(base$mode)))

  # The warm start is [beta | per-term RE effects].
  n_lat <- d$p + sum(d$n_re)

  short <- list(
    X             = d$X[seq_len(d$N - 1L), , drop = FALSE],
    n             = d$n[seq_len(d$N - 1L)],
    re_sigma_list = args$re_sigma_list[1],
    weights       = rep(1, d$N - 1L),
    re_ncoefs     = 1L,
    offset        = rep(0, d$N - 1L),
    x_init        = rep(0, n_lat - 1L)
  )
  # Replace outright rather than modifyList(): a list-valued argument
  # (re_sigma_list) is MERGED element-wise by that, which leaves it its
  # original length and tests nothing.
  with_arg <- function(nm, value) { a <- args; a[[nm]] <- value; a }

  for (nm in names(short)) {
    expect_error(
      do.call(tulpa:::cpp_laplace_fit_multi_re, with_arg(nm, short[[nm]])),
      "must equal",
      info = paste("short", nm, "was not refused")
    )
  }

  # A long argument is refused on the same check: it is as much a sign the
  # caller built the wrong vector as a short one, and silently ignoring the
  # tail would fit a model the caller did not ask for.
  expect_error(
    do.call(tulpa:::cpp_laplace_fit_multi_re,
            with_arg("offset", rep(0, d$N + 1L))),
    "must equal")
})

test_that("cpp_laplace_fit_multi_re still takes every argument at its own length", {
  d <- .lal_data()
  args <- .lal_multi_args(d)
  n_lat <- d$p + sum(d$n_re)
  ok <- list(
    weights   = rep(1, d$N),
    re_ncoefs = rep(1L, length(d$n_re)),
    offset    = rep(0, d$N),
    x_init    = rep(0, n_lat)
  )
  for (nm in names(ok)) {
    a <- args; a[[nm]] <- ok[[nm]]
    fit <- do.call(tulpa:::cpp_laplace_fit_multi_re, a)
    expect_true(all(is.finite(fit$mode)),
                info = paste(nm, "at its own length"))
  }
  # All four together, and the fit is the one the base call produces (weights
  # of 1 and a zero offset are the defaults; a zero warm start is where the
  # solve already begins).
  full <- do.call(tulpa:::cpp_laplace_fit_multi_re, c(args, ok))
  base <- do.call(tulpa:::cpp_laplace_fit_multi_re, args)
  expect_equal(full$mode, base$mode, tolerance = 1e-10)
})

test_that("cpp_laplace_fit rejects a short re_idx, n or X", {
  d <- .lal_data()
  args <- list(y = d$y, n = d$n, X = d$X,
               re_idx = as.numeric(d$idx[[1]]),
               n_re_groups = d$n_re[1], sigma_re = 0.5,
               family = "poisson", phi = 1.0,
               max_iter = 20L, tol = 1e-6, n_threads = 1L)
  expect_true(all(is.finite(do.call(tulpa:::cpp_laplace_fit, args)$mode)))

  short <- list(
    re_idx = as.numeric(d$idx[[1]])[seq_len(d$N - 1L)],
    n      = d$n[seq_len(d$N - 1L)],
    X      = d$X[seq_len(d$N - 1L), , drop = FALSE]
  )
  for (nm in names(short)) {
    a <- args; a[[nm]] <- short[[nm]]
    expect_error(do.call(tulpa:::cpp_laplace_fit, a),
                 "must equal",
                 info = paste("short", nm, "was not refused"))
  }
})
