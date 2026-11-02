# Posterior-predictive draws on a nested fit over a random-effect covariance.
# Each fixed-effect draw comes from one integration node, and the node's inner
# Laplace solve is a joint Gaussian over [beta | b]; a row's random effects are
# drawn from that Gaussian's conditional given the row's beta, so the replicates
# carry the group effects and the criteria score the conditional model.

test_that("the conditional draw paired with the Schur marginal is the joint Gaussian", {
  set.seed(11)
  nf <- 2L; nr <- 3L
  A <- matrix(rnorm(25), 5)
  H <- crossprod(A) + diag(5)
  mode <- rnorm(5)
  V <- solve(H)
  S <- 40000L
  beta <- sweep(matrix(rnorm(S * nf), S) %*% chol(V[1:nf, 1:nf]), 2, mode[1:nf], "+")
  b <- tulpa:::.gaussian_re_conditional_draws(Matrix::Matrix(H, sparse = TRUE),
                                              mode, nf, beta, nr)
  xb <- cbind(beta, b)
  expect_equal(colMeans(xb), mode, tolerance = 0.03)
  expect_lt(max(abs(stats::cov(xb) - V)) / max(abs(V)), 0.03)
})

test_that("node-mixture draws follow each row's node", {
  set.seed(12)
  nodes <- lapply(1:2, function(k) {
    A <- matrix(rnorm(9), 3)
    list(mode = c(0, 5 * k, -5 * k), H = crossprod(A) + diag(3) * 50)
  })
  obj <- list(latent_nodes = nodes, draw_nodes = rep(1:2, each = 50))
  beta <- matrix(0, 100, 1)
  b <- tulpa:::.node_re_conditional_draws(obj, beta, seq_len(100), 2L)
  expect_lt(abs(mean(b[1:50, 1]) - 5), 0.2)
  expect_lt(abs(mean(b[51:100, 1]) - 10), 0.2)
  # A row subset reads the nodes of the rows it kept.
  b2 <- tulpa:::.node_re_conditional_draws(obj, beta[1:10, , drop = FALSE],
                                           91:100, 2L)
  expect_lt(abs(mean(b2[, 2]) + 10), 0.3)
})

skip_on_cran()

.rcp_data <- fixture_once(function() {
  set.seed(20260716)
  G <- 40L; npg <- 12L; N <- G * npg
  grp <- rep(seq_len(G), each = npg)
  x <- rnorm(N)
  S <- matrix(c(0.49, 0.14, 0.14, 0.25), 2)
  u <- t(t(chol(S)) %*% matrix(rnorm(2 * G), 2))
  y <- rpois(N, exp(0.2 + 0.5 * x + u[grp, 1] + u[grp, 2] * x))
  data.frame(y = y, x = x, g = factor(grp))
})

test_that("posterior_predict() on a nested random-slope fit carries the group effects", {
  d <- .rcp_data()
  fit <- tulpa(y ~ x + (1 + x | g), data = d, family = "poisson",
               mode = "laplace", control = list(diagnose_k = FALSE))
  expect_identical(fit$backend, "re_cov_nested")
  expect_silent(yrep <- posterior_predict(fit, ndraws = 300, seed = 1))
  expect_gt(stats::cor(colMeans(yrep), d$y), 0.9)
  # Per observation, the replicate mean against the observed count: a
  # population-level replicate cannot follow the groups and over-disperses.
  pdf(NULL); on.exit(grDevices::dev.off(), add = TRUE)
  cm <- check_model(fit)
  expect_gt(cm$ks_p, 0.01)
  expect_gt(cm$disp_ratio, 0.8)
  expect_lt(cm$disp_ratio, 1.25)
})

test_that("WAIC on nested fits scores the conditional model and agrees with the exact sampler", {
  d <- .rcp_data()
  slp <- tulpa(y ~ x + (1 + x | g), data = d, family = "poisson",
               mode = "laplace", control = list(diagnose_k = FALSE))
  int <- tulpa(y ~ x + (1 | g), data = d, family = "poisson",
               mode = "laplace", control = list(re_cov = "nested",
                                                diagnose_k = FALSE))
  gib <- suppressWarnings(tulpa(y ~ x + (1 + x | g), data = d, family = "poisson",
               mode = "laplace", control = list(re_cov = "gibbs", n_iter = 12000L,
                                                warmup = 2000L, seed = 3L)))
  cmp <- compare_models(intercept = int, slopes = slp, criterion = "waic")
  expect_identical(cmp$model[1], "slopes")
  e <- function(f) cmp_elpd(compare_models(intercept = int, m = f,
                                            criterion = "waic"))
  cmp_elpd <- function(tab) tab$elpd[tab$model == "m"]
  # Nodes centred at the leading-order latent mean: 0.7 nats from the exact
  # sampler's elpd. Centred at the joint mode the mixture read 4.8 nats low,
  # every eta shifted up by 0.063 on average.
  expect_lt(abs(e(slp) - e(gib)), 1.5)
})
