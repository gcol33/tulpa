# tulpa(hyperprior =) reaches every route that integrates or maximizes over
# hyperparameters, and is refused where the backend carries priors of its own.

set.seed(21)
n  <- 160L
x  <- rnorm(n)
g  <- rep(1:16, each = 10L)
u  <- rnorm(16, 0, 0.6)
d  <- data.frame(x = x, g = factor(g), time = rep(1:16, each = 10L),
                 yb = rbinom(n, 1, plogis(0.2 + 0.6 * x + u[g])))

test_that("the backends that read hyperprior are the registry's declaration", {
  expect_setequal(tulpa:::.hyperprior_backends(),
                  c("re_cov_nested", "eb", "nested_laplace",
                    "nested_laplace_joint", "spde"))
  for (b in tulpa:::.hyperprior_backends()) {
    fitter <- match.fun(tulpa:::BACKEND_REGISTRY[[b]]$fitter)
    expect_true("hyperprior" %in% names(formals(fitter)), info = b)
    expect_identical(eval(formals(fitter)$hyperprior), c("proper", "flat"),
                     info = b)
  }
})

test_that("a backend that does not read hyperprior refuses \"flat\"", {
  expect_error(tulpa(yb ~ x + (1 | g), data = d, family = "binomial",
                     mode = "laplace", sigma_re = 1, hyperprior = "flat"),
               "not read by backend 'laplace'")
  expect_error(tulpa(yb ~ x + (1 + x | g), data = d, family = "binomial",
                     mode = "laplace", hyperprior = "flat",
                     control = list(re_cov = "gibbs")),
               "not read by backend 're_cov_gibbs'")
})

test_that("the nested route folds the choice into the fit", {
  skip_on_cran()
  fp <- tulpa(yb ~ x, data = d, family = "binomial",
              temporal = temporal_rw1("time"), mode = "auto")
  ff <- tulpa(yb ~ x, data = d, family = "binomial",
              temporal = temporal_rw1("time"), mode = "auto",
              hyperprior = "flat")
  expect_identical(ff$backend, "nested_laplace")
  expect_true(is.finite(fp$log_evidence))
  expect_true(is.na(ff$log_evidence))
  expect_identical(ff$log_evidence_declined, "improper_hyperprior")
  expect_true(all(unlist(ff$log_hyperprior_declined) == "flat_hyperprior"))
  expect_true(length(ff$log_hyperprior_declined) > 0L)
})

test_that("the RE-covariance redirect and EB take the same choice", {
  skip_on_cran()
  ff <- tulpa(yb ~ x + (1 + x | g), data = d, family = "binomial",
              mode = "laplace", hyperprior = "flat")
  expect_identical(ff$backend, "re_cov_nested")
  fp <- tulpa(yb ~ x + (1 + x | g), data = d, family = "binomial",
              mode = "laplace")
  expect_true(all(ff$log_hyperprior == 0))
  expect_true(all(is.finite(fp$log_hyperprior) & fp$log_hyperprior != 0))

  terms <- list(idx = as.integer(d$g), n_groups = 16L, n_coefs = 1L,
                correlated = FALSE)
  X <- cbind(1, d$x)
  ep <- tulpa_eb(d$yb, rep(1L, n), X, "binomial", re_terms = terms)
  ef <- tulpa_eb(d$yb, rep(1L, n), X, "binomial", re_terms = terms,
                 hyperprior = "flat")
  gp <- tulpa(yb ~ x + (1 | g), data = d, family = "binomial", mode = "eb")
  gf <- tulpa(yb ~ x + (1 | g), data = d, family = "binomial", mode = "eb",
              hyperprior = "flat")
  expect_equal(gp$theta_hat, ep$theta_hat, tolerance = 1e-6)
  expect_equal(gf$theta_hat, ef$theta_hat, tolerance = 1e-6)
  expect_false(isTRUE(all.equal(ep$theta_hat, ef$theta_hat)))
})

test_that("the SPDE route keeps a stated anchor and drops a defaulted one", {
  skip_on_cran()
  set.seed(5)
  m <- 120L
  coords <- cbind(runif(m), runif(m))
  w <- as.numeric(t(chol(exp(-as.matrix(dist(coords)) / 0.3) + diag(1e-6, m))) %*%
                    rnorm(m))
  y <- rbinom(m, 1, plogis(0.2 + 0.8 * w))
  X <- matrix(1, m, 1L)
  mesh <- tulpaMesh::tulpa_mesh(coords, max_edge = c(0.15, 0.4), cutoff = 0.05)
  sp_default <- spatial_spde(coords, mesh = mesh)
  sp_range   <- spatial_spde(coords, mesh = mesh, prior_range = c(0.3, 0.5))
  expect_identical(sp_default$prior_stated, c(range = FALSE, sigma = FALSE))
  expect_identical(sp_range$prior_stated, c(range = TRUE, sigma = FALSE))

  r <- c(0.2, 0.6); s <- c(0.5, 1.5)
  expect_identical(tulpa:::.spde_log_hyperprior(r, s, sp_default),
                   tulpa:::.spde_log_hyperprior(r, s, sp_default, "proper"))
  rec <- tulpa:::.spde_hyperprior_record(r, s, sp_range, "flat")
  expect_identical(rec$axes, "range")
  expect_identical(names(rec$declined), "sigma")

  ctrl <- list(method = "grid", n_grid = 4L, diagnose_k = FALSE)
  fp <- fit_spde(y, X, sp_range, control = ctrl)
  ff <- fit_spde(y, X, sp_range, control = ctrl, hyperprior = "flat")
  expect_true(is.finite(fp$log_evidence))
  expect_null(fp$log_hyperprior_declined)
  expect_identical(names(ff$log_hyperprior_declined), "sigma")
  expect_identical(ff$log_evidence_declined, "improper_hyperprior")

  df <- data.frame(y = y, sx = coords[, 1], sy = coords[, 2])
  tf <- tulpa(y ~ 1, data = df, family = "binomial", spatial = sp_range,
              hyperprior = "flat", control = ctrl)
  expect_identical(tf$backend, "spde")
  expect_identical(names(tf$log_hyperprior_declined), "sigma")
})
