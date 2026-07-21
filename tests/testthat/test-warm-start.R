# Warm-starting NUTS from a Laplace / EB fit.
#
# The property that matters is that a warm start changes only where the sampler
# BEGINS, never what it converges to. Everything below is built around that:
# the placement tests check the starting vector lands the estimate in the slots
# the engine reserved for it, and the equivalence tests check the posterior is
# the one a cold chain finds.

ws_data <- function(seed = 1L, G = 25L, per = 12L) {
  set.seed(seed)
  n <- G * per
  g <- rep(seq_len(G), each = per)
  x <- rnorm(n)
  b <- rnorm(G, 0, 0.7)
  data.frame(y = rpois(n, exp(0.3 + 0.5 * x + b[g])),
             x = x, g = factor(g))
}

ws_layout <- function(d, zi = NULL) {
  X <- stats::model.matrix(~ x, d)
  gidx <- as.integer(d$g)
  cpp_tulpa_glmm_layout(
    y = as.numeric(d$y), n_trials = rep(1L, nrow(d)), X = X,
    family = "poisson", phi = 1.0, sigma_beta = 10.0,
    re_spec = list(idx = list(gidx), ngroups = nlevels(d$g),
                   ncoefs = 1L, correlated = FALSE),
    fixed_names = colnames(X),
    zi_spec = zi)
}


test_that("the layout probe reports the sampler's own parameter vector", {
  d <- ws_data()
  lay <- ws_layout(d)

  # 2 fixed + 25 group deviations + 1 log_sigma_re.
  expect_equal(lay$total_params, 2L + nlevels(d$g) + 1L)
  expect_length(lay$param_names, lay$total_params)
  expect_length(lay$re_terms, 1L)
  expect_equal(lay$re_terms[[1]]$n_groups, nlevels(d$g))
  expect_equal(lay$re_terms[[1]]$n_coefs, 1L)
  expect_length(lay$unsupported, 0L)

  # The reported spans must agree with the names the sampler labels columns
  # with -- that is the check that the probe reads the same layout the kernel
  # samples, rather than a parallel reconstruction of it.
  beta_idx <- seq.int(lay$beta[[1]][1], lay$beta[[1]][2])
  expect_equal(lay$param_names[beta_idx], colnames(stats::model.matrix(~ x, d)))
  re_idx <- seq.int(lay$re_terms[[1]]$re[1], lay$re_terms[[1]]$re[2])
  expect_true(all(grepl("^re\\[", lay$param_names[re_idx])))
  expect_true(grepl("^log_sigma_re",
                    lay$param_names[[lay$re_terms[[1]]$log_sigma_re]]))
})


test_that("the warm start places the source fit's estimates in the right slots", {
  d <- ws_data()
  lay <- ws_layout(d)
  re_terms <- list(list(idx = as.integer(d$g), n_groups = nlevels(d$g),
                        n_coefs = 1L))
  X <- stats::model.matrix(~ x, d)

  src <- tulpa_eb(y = d$y, n_trials = NULL, X = X, re_terms = re_terms,
                  family = "poisson")

  ws <- .build_warm_start(src, lay, re_terms = re_terms, n_chains = 1L,
                          jitter = 0)
  init <- ws$init[1, ]

  # Fixed effects and random effects land where the layout says they do.
  beta_idx <- seq.int(lay$beta[[1]][1], lay$beta[[1]][2])
  expect_equal(unname(init[beta_idx]), unname(src$mode[seq_len(src$n_fixed)]))

  re_idx <- seq.int(lay$re_terms[[1]]$re[1], lay$re_terms[[1]]$re[2])
  expect_equal(unname(init[re_idx]),
               unname(src$mode[-seq_len(src$n_fixed)]))

  # The variance component is handed over on the sampler's scale, which is log.
  expect_equal(init[[lay$re_terms[[1]]$log_sigma_re]],
               log(src$map$sigma), tolerance = 1e-10)

  # The metric is withheld by default (see below); the position is the part
  # that is handed over.
  expect_null(ws$inv_metric_diag)
})


test_that("the inverse-mass diagonal is withheld unless asked for", {
  d <- ws_data()
  lay <- ws_layout(d)
  re_terms <- list(list(idx = as.integer(d$g), n_groups = nlevels(d$g),
                        n_coefs = 1L))
  X <- stats::model.matrix(~ x, d)
  src <- tulpa_eb(y = d$y, n_trials = NULL, X = X, re_terms = re_terms,
                  family = "poisson")

  # Default: position only. A metric that is right for the fixed effects and
  # wrong for the random effects and variance components is worse than none,
  # because NUTS takes its step size from the worst-conditioned direction.
  plain <- .build_warm_start(src, lay, re_terms = re_terms, n_chains = 1L)
  expect_null(plain$inv_metric_diag)

  # Opted in, it must still be a valid metric: the kernel rejects any entry
  # that is not finite and positive, since it reads them as variances.
  with_m <- .build_warm_start(src, lay, re_terms = re_terms, n_chains = 1L,
                              metric = TRUE)
  expect_length(with_m$inv_metric_diag, lay$total_params)
  expect_true(all(is.finite(with_m$inv_metric_diag)))
  expect_true(all(with_m$inv_metric_diag > 0))
})


test_that("chains are dispersed rather than stacked on the mode", {
  d <- ws_data()
  lay <- ws_layout(d)
  re_terms <- list(list(idx = as.integer(d$g), n_groups = nlevels(d$g),
                        n_coefs = 1L))
  X <- stats::model.matrix(~ x, d)
  src <- tulpa_eb(y = d$y, n_trials = NULL, X = X, re_terms = re_terms,
                  family = "poisson")

  set.seed(4)
  ws <- .build_warm_start(src, lay, re_terms = re_terms, n_chains = 4L)
  expect_equal(dim(ws$init), c(4L, lay$total_params))

  # Chain 1 is the mode itself; the rest are drawn around it. Identical starts
  # would leave Rhat comparing chains that began at the same point.
  expect_gt(max(abs(ws$init[2, ] - ws$init[1, ])), 0)
  expect_false(isTRUE(all.equal(ws$init[2, ], ws$init[3, ])))

  # jitter = 0 is the opt-out, and must genuinely stack them.
  ws0 <- .build_warm_start(src, lay, re_terms = re_terms, n_chains = 3L,
                           jitter = 0)
  expect_equal(ws0$init[1, ], ws0$init[2, ])
  expect_equal(ws0$init[1, ], ws0$init[3, ])
})


test_that("a warm-started chain finds the same posterior as a cold one", {
  skip_on_cran()
  # The contract: warm-starting moves where sampling begins, not where it ends.
  d <- ws_data(seed = 2L)
  ctl <- list(n_iter = 1500L, n_warmup = 750L, n_chains = 4L, seed = 9L)

  cold <- tulpa(y ~ x + (1 | g), data = d, family = "poisson",
                mode = "hmc", control = ctl)
  warm <- tulpa(y ~ x + (1 | g), data = d, family = "poisson",
                mode = "hmc", warm_start = "eb", control = ctl)

  # Loose enough for Monte Carlo error at this chain length, tight enough that
  # a mis-placed init (a wrong slot, a wrong scale) would fail it.
  expect_equal(unname(coef(cold)), unname(coef(warm)), tolerance = 0.05)

  # A warm start must not buy agreement by breaking the geometry.
  expect_equal(sum(warm$divergent), 0L)
  dw <- as.data.frame(diagnostics(warm))
  expect_true(all(dw$rhat < 1.05))
})


test_that("a warm start is refused where it cannot be honoured", {
  d <- ws_data()

  # Backends other than NUTS take neither an init nor an inverse mass.
  expect_error(
    tulpa(y ~ x + (1 | g), data = d, family = "poisson", mode = "ess",
          warm_start = "eb", control = list(n_iter = 60L, n_warmup = 30L)),
    "only carried by the NUTS/HMC kernel")

  expect_error(
    tulpa(y ~ x + (1 | g), data = d, family = "poisson", mode = "hmc",
          warm_start = "pathfinder",
          control = list(n_iter = 60L, n_warmup = 30L)),
    "must be NULL")

  expect_error(
    tulpa(y ~ x + (1 | g), data = d, family = "poisson", mode = "hmc",
          warm_start = 42, control = list(n_iter = 60L, n_warmup = 30L)),
    "must be NULL, a mode name, or a fitted")
})


test_that("a fit from a backend with no mode is refused as a warm-start source", {
  d <- ws_data()
  lay <- ws_layout(d)
  re_terms <- list(list(idx = as.integer(d$g), n_groups = nlevels(d$g),
                        n_coefs = 1L))

  fake <- structure(list(backend = "smc", mode = c(0, 0), n_fixed = 2L,
                         fixed_names = c("(Intercept)", "x")),
                    class = c("tulpa_fit", "list"))
  # `args = list()` is deliberately empty: the source fit must be rejected on
  # its own terms, before anything is built for it. Reaching the engine with
  # empty arguments would report this as a C++ error about a model the caller
  # never described.
  expect_error(
    .resolve_warm_start(fake, args = list(), re_terms = re_terms,
                        sigma_re = NULL, beta_prior = NULL, n_chains = 1L),
    "carries no mode and curvature")
})


test_that("a source fit for a different model is refused", {
  d <- ws_data()
  lay <- ws_layout(d)
  X <- stats::model.matrix(~ x, d)
  re_terms <- list(list(idx = as.integer(d$g), n_groups = nlevels(d$g),
                        n_coefs = 1L))
  src <- tulpa_eb(y = d$y, n_trials = NULL, X = X, re_terms = re_terms,
                  family = "poisson")

  # Two RE terms claimed against a one-term layout: the counts disagree, and
  # silently filling one term would seed the sampler from the wrong model.
  two_terms <- c(re_terms, re_terms)
  expect_error(
    .build_warm_start(src, lay, re_terms = two_terms, n_chains = 1L),
    "must come from the same model")
})


test_that("the warm start carries zero-inflation coefficients across", {
  skip_on_cran()
  # The one block whose position genuinely differs between the two layouts: the
  # Laplace path appends the ZI predictor to the fixed effects as a second
  # process, the sampler gives it its own block. This is the test that the move
  # is explicit rather than assumed.
  set.seed(6)
  n <- 600L
  x <- rnorm(n)
  y <- rpois(n, exp(0.5 + 0.4 * x))
  y[runif(n) < 0.3] <- 0L
  d <- data.frame(y = y, x = x)

  ctl <- list(n_iter = 1200L, n_warmup = 600L, n_chains = 2L, seed = 8L)
  cold <- tulpa(y ~ x, data = d, family = "poisson", ziformula = ~1,
                mode = "hmc", control = ctl)
  warm <- tulpa(y ~ x, data = d, family = "poisson", ziformula = ~1,
                mode = "hmc", warm_start = "laplace", control = ctl)

  expect_equal(unname(coef(cold)), unname(coef(warm)), tolerance = 0.1)
})
