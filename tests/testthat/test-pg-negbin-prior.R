skip_on_cran()

nb_prior_data <- function() {
  list(y = c(0L, 2L, 1L, 3L, 0L, 1L, 1L, 2L),
       X = cbind(1, rep(c(-1, 1), 4)), unit = rep(1:2, each = 4))
}

# Gaussian quadrature for Z ~ N(0,1) or U ~ Exp(1), from their orthogonal
# polynomials' Jacobi matrices. A two-node ICAR field is (f,-f), f = sigma Z/2,
# and its PC scale prior is sigma = U / lambda.
nb_prior_rule <- function(n, exponential = FALSE) {
  J <- diag(if (exponential) 2 * seq_len(n) - 1 else rep(0, n))
  off <- if (exponential) seq_len(n - 1L) else sqrt(seq_len(n - 1L))
  J[cbind(seq_len(n - 1L), 2:n)] <- off
  J[cbind(2:n, seq_len(n - 1L))] <- off
  e <- eigen(J, symmetric = TRUE)
  list(x = e$values, w = e$vectors[1, ]^2)
}

nb_prior_quantiles <- function(x, w) {
  w <- w / sum(w)
  cdf <- cumsum(w) - w / 2
  keep <- !duplicated(cdf)
  approx(cdf[keep], x[keep], c(.025, .5, .975), rule = 2)$y
}

nb_prior_log_quantiles <- function(x, w) {
  # The density is supplied with respect to d log(x). Interpolate its smooth
  # log-density before integrating the CDF on a fine logarithmic grid.
  fine <- spline(log(x), log(w), n = 10001L, method = "natural")
  exp(nb_prior_quantiles(fine$x, exp(fine$y - max(fine$y))))
}

nb_prior_convolve <- function(a, b) {
  n <- length(a) + length(b) - 1L
  size <- 2^ceiling(log2(n))
  fa <- fft(c(a, numeric(size - length(a))))
  fb <- fft(c(b, numeric(size - length(b))))
  pmax(Re(fft(fa * fb, inverse = TRUE))[seq_len(n)] / size, 0)
}

nb_prior_reference <- function(sd, spatial = FALSE, n = 101L, nf = 12L, nr = 61L) {
  d <- nb_prior_data()
  # theta_minus = beta0 - beta1 and theta_plus = beta0 + beta1 have independent
  # N(0, 2 sd^2) priors. Given (r,f), each covariate class reads only its theta.
  # Convolutions give beta0 = (theta_plus + theta_minus)/2 and
  # beta1 = (theta_plus - theta_minus)/2 after integrating both theta axes.
  theta <- seq(-min(8, 8 * sqrt(2) * sd), min(8, 8 * sqrt(2) * sd),
               length.out = 4L * n + 1L)
  nt <- length(theta)
  b <- seq(theta[1], tail(theta, 1), length.out = 2L * nt - 1L)
  tw <- dnorm(theta, 0, sqrt(2) * sd)
  # The omitted r tails have Gamma(100,50) prior probability below 1e-22.
  u <- seq(log(.6), log(5), length.out = nr)
  r <- exp(u)
  rw <- dgamma(r, 100, 50) * r
  if (spatial) {
    z <- nb_prior_rule(nf)
    s <- nb_prior_rule(nf, exponential = TRUE)
    f <- as.vector(outer(z$x, s$x, "*") / (2 * (-log(.1) / .8)))
    fw <- as.vector(outer(z$w, s$w))
  } else {
    f <- 0
    fw <- 1
  }
  rows <- list(c(1L, 3L), c(2L, 4L), c(5L, 7L), c(6L, 8L))
  mu <- lapply(rows, function(ids) {
    i <- ids[1L]
    exp(outer(theta, if (d$unit[i] == 1L) f else -f, "+"))
  })
  marginal <- matrix(0, length(b), 2)
  r_mass <- numeric(length(r))
  f_mass <- numeric(length(f))
  for (k in seq_along(r)) {
    ll <- lapply(seq_along(rows), function(g) {
      ids <- rows[[g]]
      matrix(dnbinom(d$y[ids[1]], size = r[k], mu = mu[[g]], log = TRUE) +
               dnbinom(d$y[ids[2]], size = r[k], mu = mu[[g]], log = TRUE), nt)
    })
    minus <- exp(ll[[1]] + ll[[3]]) * tw
    plus <- exp(ll[[2]] + ll[[4]]) * tw
    for (j in seq_along(f)) {
      w <- rw[k] * fw[j]
      marginal[, 1] <- marginal[, 1] + w * nb_prior_convolve(minus[, j], plus[, j])
      marginal[, 2] <- marginal[, 2] + w * nb_prior_convolve(rev(minus[, j]), plus[, j])
      node_mass <- w * sum(minus[, j]) * sum(plus[, j])
      r_mass[k] <- r_mass[k] + node_mass
      f_mass[j] <- f_mass[j] + node_mass
    }
  }
  marginal <- sweep(marginal, 2L, colSums(marginal), "/")
  means <- colSums(b * marginal)
  sds <- sqrt(colSums(b^2 * marginal) - means^2)
  list(mean = means, sd = sds,
       quantiles = apply(marginal, 2L, function(w) nb_prior_quantiles(b, w)),
       r_quantiles = nb_prior_log_quantiles(r, r_mass),
       field_mean = sum(f * f_mass) / sum(f_mass),
       field_sd = sqrt(sum(f^2 * f_mass) / sum(f_mass) -
                         (sum(f * f_mass) / sum(f_mass))^2))
}

nb_prior_sample <- function(sd, spatial = FALSE, n_iter = 33000L) {
  d <- nb_prior_data()
  common <- list(y = d$y, X = d$X, n_iter = n_iter, n_warmup = 3000L,
                 thin = 1L, prior_beta_sd = sd, prior_r_shape = 100,
                 prior_r_rate = 50, r_init = 2, store_eta = TRUE,
                 verbose = FALSE, n_threads = 1L)
  if (spatial) {
    do.call(tulpa:::cpp_pg_negbin_gibbs_spatial, c(common, list(
      re_group = rep(1L, 8), n_re_groups = 0L, spatial_group = d$unit,
      n_spatial_units = 2L, adj_list = list(2L, 1L), n_neighbors = c(1L, 1L),
      prior_sigma_re_scale = 1, prior_sigma_spatial_U = .8,
      prior_sigma_spatial_alpha = .1)))
  } else {
    do.call(tulpa:::cpp_pg_negbin_gibbs, c(common, list(
      group = rep(1L, 8), n_groups = 0L, prior_sigma_scale = 1)))
  }
}

test_that("the marginal convolution sums both coefficient axes", {
  skip_on_cran()
  expect_equal(nb_prior_convolve(1:3, 4:6), c(4, 13, 28, 27, 18), tolerance = 1e-13)
})

test_that("the log-axis CDF resolves Gamma dispersion quantiles", {
  skip_on_cran()
  u <- seq(log(.6), log(5), length.out = 61L)
  r <- exp(u)
  expect_equal(nb_prior_log_quantiles(r, dgamma(r, 100, 50) * r),
               qgamma(c(.025, .5, .975), 100, 50), tolerance = 1e-5)
})

test_that("both NB C API hooks preserve the engine's NB2 draws", {
  skip_on_cran()
  load_cpp_fixture("pg_negbin_prior_api")
  d <- nb_prior_data()
  for (spatial in c(FALSE, TRUE)) for (ng in 0:1) {
    set.seed(1163)
    api <- cpp_test_pg_negbin_prior_api(d$y, d$X, spatial, ng)
    common <- list(y = d$y, X = d$X, n_iter = 80L, n_warmup = 20L,
                   thin = 1L, prior_beta_sd = .1, prior_r_shape = 100,
                   prior_r_rate = 50, r_init = 2, store_eta = TRUE,
                   verbose = FALSE, n_threads = 1L)
    set.seed(1163)
    if (spatial) {
      fit <- do.call(tulpa:::cpp_pg_negbin_gibbs_spatial, c(common, list(
        re_group = rep(1L, 8), n_re_groups = ng, spatial_group = d$unit,
        n_spatial_units = 2L, adj_list = list(2L, 1L), n_neighbors = c(1L, 1L),
        prior_sigma_re_scale = 1, prior_sigma_spatial_U = .8,
        prior_sigma_spatial_alpha = .1)))
    } else {
      fit <- do.call(tulpa:::cpp_pg_negbin_gibbs, c(common, list(
        group = rep(1L, 8), n_groups = ng, prior_sigma_scale = 1)))
    }
    expect_identical(api, fit[names(api)])
  }
})

for (spatial in c(FALSE, TRUE)) {
  for (sd in c(.1, 2.5)) {
    local({
      spatial <- spatial
      sd <- sd
      test_that(sprintf("NB2 coefficient priors match quadrature: sd=%g, ICAR=%s",
                        sd, spatial), {
        skip_if_not_slow()
        ref <- nb_prior_reference(sd, spatial)
        fine <- nb_prior_reference(sd, spatial, n = 151L, nf = 16L)
        expect_lt(max(abs(ref$quantiles - fine$quantiles) /
                        rep(fine$sd, each = 3)), .02)
        expect_lt(max(abs(ref$sd / fine$sd - 1)), .002)
        set.seed(1163L)
        fit <- nb_prior_sample(sd, spatial)
        q <- apply(fit$beta, 2, quantile, c(.025, .5, .975))
        expect_lt(max(abs(q - fine$quantiles) / rep(fine$sd, each = 3)), .08)
        expect_lt(max(abs(colMeans(fit$beta) - fine$mean) / fine$sd), .04)
        expect_lt(max(abs(apply(fit$beta, 2, stats::sd) / fine$sd - 1)), .04)
        expect_lt(max(abs(quantile(fit$r, c(.025, .5, .975)) -
                            fine$r_quantiles)), .025)
        eta <- fit$beta %*% t(nb_prior_data()$X)
        if (spatial) {
          eta <- eta + fit$spatial[, nb_prior_data()$unit]
          expect_lt(max(abs(rowSums(fit$spatial))), 1e-12)
          expect_lt(abs(mean(fit$spatial[, 1]) - fine$field_mean) / fine$field_sd, .08)
          expect_lt(abs(stats::sd(fit$spatial[, 1]) / fine$field_sd - 1), .08)
        }
        expect_equal(fit$eta, unname(eta), tolerance = 1e-12)
      })
    })
  }
}

nb_iid_reference <- function(step = .02) {
  y <- c(0L, 1L, 2L, 1L, 1L, 0L, 1L, 2L)
  b <- seq(-.6, .6, length.out = 401L)
  edges <- seq(-5, 5, by = step)
  f <- (head(edges, -1L) + tail(edges, -1L)) / 2
  # Integrate the N(0, sigma^2) bin probabilities against a half-Cauchy(.25)
  # scale using its uniform quantile coordinate. Bin probabilities include
  # the marginal density's integrable singularity at f = 0.
  fw <- vapply(seq_along(f), function(j) {
    integrate(function(u) {
      sigma <- .25 * tan(pi * u / 2)
      pnorm(edges[j + 1L] / sigma) - pnorm(edges[j] / sigma)
    }, 0, 1, rel.tol = 1e-9)$value
  }, 0)
  u <- seq(log(.6), log(5), length.out = 61L)
  mass <- numeric(length(b))
  fm <- numeric(length(f))
  eta <- outer(b, f, "+")
  for (lr in u) {
    r <- exp(lr)
    e <- eta - lr
    ll <- sum(lgamma(y + r) - lgamma(r) - lgamma(y + 1)) +
      sum(y) * e - (sum(y) + length(y) * r) *
      (pmax(e, 0) + log1p(exp(-abs(e))))
    w <- exp(ll) * dnorm(b, 0, .1) * dgamma(r, 100, 50) * r
    mass <- mass + as.numeric(w %*% fw)
    fm <- fm + colSums(w) * fw
  }
  mass <- mass / sum(mass)
  fm <- fm / sum(fm)
  list(quantiles = nb_prior_quantiles(b, mass),
       sd = sqrt(sum(b^2 * mass) - sum(b * mass)^2),
       field_mean = sum(f * fm), field_sd = sqrt(sum(f^2 * fm) - sum(f * fm)^2))
}

test_that("the NB2 intercept and iid level match their declared joint prior", {
  skip_if_not_slow()
  ref <- nb_iid_reference()
  fine <- nb_iid_reference(.01)
  expect_lt(max(abs(ref$quantiles - fine$quantiles)) / fine$sd, .005)
  expect_lt(abs(ref$field_sd / fine$field_sd - 1), .005)
  set.seed(1163)
  fit <- tulpa:::cpp_pg_negbin_gibbs(
    c(0L, 1L, 2L, 1L, 1L, 0L, 1L, 2L), matrix(1, 8, 1), rep(1L, 8), 1L,
    33000L, 3000L, 1L, .1, .25, 100, 50, 2, TRUE, FALSE, 1L)
  expect_lt(max(abs(quantile(fit$beta[, 1], c(.025, .5, .975)) -
                      fine$quantiles)) / fine$sd, .08)
  expect_lt(abs(stats::sd(fit$beta[, 1]) / fine$sd - 1), .04)
  expect_lt(abs(mean(fit$re[, 1]) - fine$field_mean) / fine$field_sd, .08)
  expect_lt(abs(stats::sd(fit$re[, 1]) / fine$field_sd - 1), .08)
  expect_equal(fit$eta[, 1], fit$beta[, 1] + fit$re[, 1], tolerance = 1e-12)
})
