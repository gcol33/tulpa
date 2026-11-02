# tulpa_re_cov_nested(n_quad > 1) integrates the fixed effects out of the AGHQ
# marginal at every covariance point it visits: a Newton profile of
# sum_g log M_g + log prior(beta), its curvature a central difference of the
# analytic beta-gradient, compiled and batched across threads
# (cpp_aghq_profile_batch).

skip_on_cran()

.rap_data <- fixture_once(function() {
  set.seed(20260716)
  G <- 30L; npg <- 8L; N <- G * npg
  g <- rep(seq_len(G), each = npg); x <- rnorm(N)
  S <- matrix(c(0.49, 0.14, 0.14, 0.25), 2)
  u <- t(t(chol(S)) %*% matrix(rnorm(2 * G), 2))
  list(y = rpois(N, exp(0.2 + 0.5 * x + u[g, 1] + u[g, 2] * x)), x = x, g = g,
       G = G)
})

test_that("the compiled profile is the BFGS + optimHess read of the same objective", {
  d <- .rap_data()
  X <- cbind(1, d$x)
  orc <- cpp_glmm_oracle_make("poisson", 1, as.numeric(d$y), rep(1, length(d$y)),
                              X, X, as.integer(d$g), d$G)
  nq <- 5L
  bfgs <- function(sc) {
    eval_at <- .aghq_grad_cache(orc, 2L, TRUE, nq, 1.0)
    lp <- function(b) sum(stats::dnorm(b, 0, 2.5, log = TRUE))
    negf <- function(b) -(eval_at(c(b, sc))$f + lp(b))
    negg <- function(b) -(eval_at(c(b, sc))$grad[1:2] - b / 2.5^2)
    o <- stats::optim(c(0, 0), negf, negg, method = "BFGS", hessian = TRUE,
                      control = list(reltol = 1e-12, maxit = 500L))
    c(-o$value + log(2 * pi) -
        0.5 * as.numeric(determinant(o$hessian)$modulus), o$par)
  }
  set.seed(1)
  pts <- cbind(log(0.7) + rnorm(4, 0, 0.2), 0.2 + rnorm(4, 0, 0.2),
               log(0.45) + rnorm(4, 0, 0.2))
  ref <- t(apply(pts, 1, bfgs))
  new <- cpp_aghq_profile_batch(pts, orc, 2L, TRUE, nq, c(0, 0), rep(0, 2),
                                rep(2.5, 2), n_threads = 1L)
  # The analytic gradient omits the node-placement derivatives, so the two
  # reads stop where it and the objective part, a few 1e-4 nats apart at most.
  expect_lt(max(abs(new$log_marginal - ref[, 1])), 1e-3)
  expect_lt(max(abs(new$beta - ref[, 2:3])), 1e-2)
  # Each point starts from the same beta, so the batch reads the same at any
  # thread count.
  two <- cpp_aghq_profile_batch(pts, orc, 2L, TRUE, nq, c(0, 0), rep(0, 2),
                                rep(2.5, 2), n_threads = 2L)
  expect_identical(two$log_marginal, new$log_marginal)
})

test_that("the n_quad > 1 nested fit runs its outer proposal through the batch", {
  d <- .rap_data()
  dd <- data.frame(y = d$y, x = d$x, g = factor(d$g))
  fit <- tulpa(y ~ x + (1 + x | g), data = dd, family = "poisson",
               mode = "laplace",
               control = list(re_cov = "aghq", n_quad = 3L, n_threads = 2L))
  expect_true(is.finite(fit$pareto_k))
  hp <- summary(fit)$hyperparameters
  expect_true(all(is.finite(hp$mean)))
  expect_gt(hp$mean[hp$parameter == "sigma_1"], 0.4)
  expect_lt(hp$mean[hp$parameter == "sigma_1"], 1.1)
})
