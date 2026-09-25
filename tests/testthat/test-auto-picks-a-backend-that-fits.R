# mode = "auto" never picks a backend that errors at dispatch.
#
# The selector answered from the model's SHAPE -- family, size, which latent
# fields are present -- and could not see the per-call features a backend
# refuses: an offset, weights, a ziformula, a second dispersion, more
# random-effect terms than its sweep updates. So it routed calls to backends
# that then refused the very call that selected them.
#
# gcol33/tulpa#666 -- every RE-bearing formula went to re_cov_gibbs.
# gcol33/tulpa#681 -- a binomial areal model with two random intercepts went to
#   the Polya-Gamma spatial Gibbs sampler, which updates one RE block.
# gcol33/tulpa#672 -- a temporal GP had no auto branch at all and was redirected
#   to nested Laplace, the one door that refuses it.

set.seed(2)
n  <- 200L
g  <- rep(1:20, each = 10L)
h  <- rep(1:10, times = 20L)
x  <- rnorm(n)
u  <- rnorm(20, 0, 0.5)
d  <- data.frame(
  x = x, g = factor(g), h = factor(h), s = factor(g),
  off = log(runif(n, 1, 3)), w = runif(n, 0.5, 1.5),
  t = sort(runif(n))
)
d$yp <- rpois(n, exp(0.3 + 0.4 * x + u[g]))
d$yb <- rbinom(n, 1, plogis(0.2 + 0.5 * x + u[g]))
d$yn <- 0.5 + 0.4 * x + u[g] + rnorm(n, 0, 0.5)

W <- matrix(0, 20, 20)
for (i in 1:19) { W[i, i + 1L] <- 1; W[i + 1L, i] <- 1 }

test_that("the acceptance predicate reads the registry, not a restated list", {
  fam <- list(name = "poisson", distribution = "poisson")
  expect_true(tulpa:::.auto_backend_ok("re_cov_gibbs", fam, list()))
  expect_false(tulpa:::.auto_backend_ok("re_cov_gibbs", fam, list(offset = TRUE)))
  expect_false(tulpa:::.auto_backend_ok("re_cov_gibbs", fam, list(ziformula = TRUE)))
  expect_false(tulpa:::.auto_backend_ok("re_cov_gibbs", fam, list(phi2 = TRUE)))
  expect_false(tulpa:::.auto_backend_ok("re_cov_gibbs", fam, list(weights = TRUE)))
  expect_false(tulpa:::.auto_backend_ok(
    "re_cov_gibbs", list(name = "gamma", distribution = "gamma"), list()))

  # The PG spatial sweep updates one RE block.
  fb <- list(name = "binomial", distribution = "binomial")
  expect_true(tulpa:::.auto_backend_ok("gibbs", fb, list(n_re_terms = 1L)))
  expect_false(tulpa:::.auto_backend_ok("gibbs", fb, list(n_re_terms = 2L)))
})

test_that("auto fits an RE model whose call the Gibbs debias cannot take", {
  skip_on_cran()
  # Each of these used to select re_cov_gibbs and then be refused by it.
  f1 <- tulpa(yp ~ x + offset(off) + (1 | g), data = d, family = "poisson",
              mode = "auto")
  expect_s3_class(f1, "tulpa_fit")
  expect_false(identical(f1$backend, "re_cov_gibbs"))

  f2 <- tulpa(yn ~ x + (1 | g), data = d, family = "gaussian", weights = d$w,
              mode = "auto")
  expect_s3_class(f2, "tulpa_fit")

  # The plain case still takes the exact debias, which is the designed default.
  f3 <- tulpa(yp ~ x + (1 | g), data = d, family = "poisson", mode = "auto")
  expect_identical(f3$backend, "re_cov_gibbs")
})

test_that("auto does not send a two-RE binomial areal model to the PG sweep", {
  skip_on_cran()
  fit <- tulpa(yb ~ x + (1 | g) + (1 | h) + spatial(s), data = d,
               family = "binomial",
               spatial = list(type = "icar", adjacency = W), mode = "auto")
  expect_s3_class(fit, "tulpa_fit")
  expect_false(identical(fit$backend, "gibbs"))

  # One RE term alongside the field is what the sweep supports, and auto still
  # picks it there.
  sel <- tulpa:::select_inference_mode(
    "auto", family = list(name = "binomial", distribution = "binomial"),
    n_obs = n, has_spatial = TRUE, spatial_type = "icar", has_re = TRUE,
    feat = list(n_re_terms = 1L))
  expect_identical(sel$backend, "gibbs")
})

test_that("auto routes a temporal GP to the sampler that fits it", {
  sel <- tulpa:::select_inference_mode(
    "auto", family = list(name = "binomial", distribution = "binomial"),
    n_obs = n, has_temporal = TRUE, temporal = list(type = "gp"))
  expect_identical(sel$backend, "hmc")

  sel2 <- tulpa:::select_inference_mode(
    "auto", family = list(name = "gaussian", distribution = "gaussian"),
    n_obs = n, has_temporal = TRUE, temporal = list(type = "multiscale"))
  expect_identical(sel2$backend, "hmc")

  # The discrete temporal fields are unchanged: they keep the nested path.
  sel3 <- tulpa:::select_inference_mode(
    "auto", family = list(name = "poisson", distribution = "poisson"),
    n_obs = n, has_temporal = TRUE, temporal = list(type = "rw1"))
  expect_false(identical(sel3$backend, "hmc"))
})

# gcol33/tulpa#769 -- auto still picked backends that then refused the very
# call that selected them: the default MALA arm, the latent and spatial
# nested arms, the multiscale arm, and the slope redirect applied after
# selection did not consult .auto_backend_ok(), and there was no temporal arm
# at all (a comment referred to one "below" that did not exist).

test_that("auto's default arm routes ziformula off MALA instead of erroring", {
  fam <- list(name = "poisson", distribution = "poisson")
  sel <- tulpa:::select_inference_mode(
    "auto", family = fam, n_obs = n, feat = list(ziformula = TRUE))
  expect_identical(sel$backend, "hmc")

  # Unaffected: no zi, still the plain default.
  sel2 <- tulpa:::select_inference_mode("auto", family = fam, n_obs = n)
  expect_identical(sel2$backend, "mala")
})

test_that("auto fits a ziformula model that used to error at the front door", {
  skip_on_cran()
  set.seed(2)
  n2 <- 160
  dq <- data.frame(x = rnorm(n2)); dq$y <- rpois(n2, exp(0.5 + 0.5 * dq$x))
  dq$y[runif(n2) < .3] <- 0
  f <- tulpa(y ~ x, data = dq, family = "poisson", ziformula = ~ 1,
             control = list(n_iter = 100L, warmup = 50L, seed = 1L))
  expect_identical(f$backend, "hmc")
})

test_that("auto's spatial-nested arm falls back to hmc when nested_laplace refuses the call", {
  fam <- list(name = "poisson", distribution = "poisson")
  sel <- tulpa:::select_inference_mode(
    "auto", family = fam, n_obs = n, has_spatial = TRUE, spatial_type = "icar",
    feat = list(ziformula = TRUE))
  expect_identical(sel$backend, "hmc")

  # SPDE has no hmc route (dispatch_glmm_modeldata() has no SPDE spec
  # builder), so it keeps returning nested_laplace even though that backend
  # also refuses ziformula -- no fallback is offered for a call nothing carries.
  sel2 <- tulpa:::select_inference_mode(
    "auto", family = fam, n_obs = n, has_spatial = TRUE, spatial_type = "spde",
    feat = list(ziformula = TRUE))
  expect_identical(sel2$backend, "nested_laplace")

  # Unaffected: no zi, still the designed nested route.
  sel3 <- tulpa:::select_inference_mode(
    "auto", family = fam, n_obs = n, has_spatial = TRUE, spatial_type = "icar")
  expect_identical(sel3$backend, "nested_laplace")
})

test_that("auto adds a temporal arm that sees ziformula before picking a backend", {
  fam <- list(name = "poisson", distribution = "poisson")
  sel <- tulpa:::select_inference_mode(
    "auto", family = fam, n_obs = n, has_temporal = TRUE,
    temporal = list(type = "rw1"), feat = list(ziformula = TRUE))
  expect_identical(sel$backend, "hmc")

  # Unaffected: no zi, still nested Laplace (matches the pre-existing
  # downstream-redirect outcome, reached one step earlier now).
  sel2 <- tulpa:::select_inference_mode(
    "auto", family = fam, n_obs = n, has_temporal = TRUE,
    temporal = list(type = "rw1"))
  expect_identical(sel2$backend, "nested_laplace")
})

test_that("the slope redirect re-checks features instead of blindly redirecting", {
  skip_on_cran()
  set.seed(3)
  dg <- data.frame(x = rnorm(n), g = factor(sample(1:12, n, TRUE)))
  dg$y <- 1 + 0.5 * dg$x + rnorm(12)[dg$g] + rnorm(n)
  w <- rep(1:2, n / 2)
  # re_cov_gibbs does not carry weights, so this used to be redirected to it
  # blindly and surface ITS refusal. re_cov_nested carries them
  # (gcol33/tulpa#874), so the slope's covariance is integrated there instead.
  fit <- tulpa(y ~ x + (1 + x | g), data = dg, weights = w, phi = 1)
  expect_identical(fit$backend, "re_cov_nested")
})

test_that("auto honours a named control$re_cov integrator", {
  skip_on_cran()
  # auto's RE arm picks the integrator itself, so the redirect that reads
  # control$re_cov never ran there: re_cov = "nested" still gave re_cov_gibbs.
  f1 <- tulpa(yp ~ x + (1 | g), data = d, family = "poisson", mode = "auto",
              control = list(re_cov = "nested"))
  expect_identical(f1$backend, "re_cov_nested")
  # ... and naming the integrator auto would refuse for this call errors.
  expect_error(
    tulpa(yp ~ x + offset(off) + (1 | g), data = d, family = "poisson",
          mode = "auto", control = list(re_cov = "gibbs")),
    "does not carry")
})
