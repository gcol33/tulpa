# field_effects(): a spatial field's posterior per unit, read where each
# backend keeps it (gcol33/tulpa#960).

.fe_ring <- function(K) {
  W <- matrix(0, K, K)
  for (i in 1:K) { j <- if (i < K) i + 1 else 1; W[i, j] <- W[j, i] <- 1 }
  W
}

.fe_ring_data <- function(K = 20L, n = 300L, seed = 1L) {
  set.seed(seed)
  region <- sample(seq_len(K), n, replace = TRUE)
  field  <- as.numeric(scale(sin(2 * pi * seq_len(K) / K)))[region]
  x <- rnorm(n)
  data.frame(x = x, region = region,
             yb = rbinom(n, 1, plogis(-0.2 + 0.7 * x + field)),
             yc = rpois(n, exp(0.2 + 0.5 * x + field)))
}

.fe_chain <- function(n) {
  A <- matrix(0, n, n)
  for (i in seq_len(n - 1L)) A[i, i + 1L] <- A[i + 1L, i] <- 1
  A
}

.fe_island_data <- function() {
  A <- matrix(0, 16, 16)
  A[1:15, 1:15] <- .fe_chain(15L)
  set.seed(1)
  d <- data.frame(region = rep(1:16, each = 4), x = rnorm(64))
  d$y <- rpois(64, exp(0.3 + 0.5 * d$x))
  list(A = A, d = d)
}

.fe_cols <- c("unit", "label", "estimate", "sd", "conf.low", "conf.high",
              "source")

skip_on_cran_budget()

test_that("field_effects() refuses a fit with no spatial field", {
  skip_on_cran_budget()
  expect_error(field_effects(list()), "tulpa_fit")
  expect_error(field_effects(structure(list(backend = "laplace"),
                                       class = "tulpa_fit")),
               "carries no spatial field")
  expect_error(field_effects(structure(list(temporal = list(type = "rw1")),
                                       class = "tulpa_fit")),
               "temporal\\(\\)")
  expect_error(field_effects(structure(list(spatial = list(type = "svc")),
                                       class = "tulpa_fit")),
               "svc\\(\\)")
})

skip_on_cran()

test_that("field_effects() errors on a fitted model without a field", {
  skip_on_cran()
  set.seed(2)
  d <- data.frame(y = rpois(80, 3), x = rnorm(80))
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
  expect_error(field_effects(fit), "carries no spatial field")
})

test_that("laplace ICAR: the field at its mode, one row per region", {
  skip_on_cran()
  K <- 20L
  d <- .fe_ring_data(K)
  fit <- tulpa(yb ~ x + spatial(region), data = d, family = "binomial",
               spatial = list(type = "icar", adjacency = .fe_ring(K)),
               mode = "laplace")
  fe <- field_effects(fit)
  expect_named(fe, .fe_cols)
  expect_equal(nrow(fe), K)
  expect_equal(fe$unit, seq_len(K))
  expect_equal(fe$estimate, unname(tail(fit$mode, K)))
  expect_true(all(fe$source == "mode"))
  expect_true(all(is.na(fe$sd)))
  expect_equal(attr(fe, "field"), "icar")
})

test_that("hmc BYM2: the combined field per draw, as the engine assembles it", {
  skip_on_cran()
  dd <- .fe_island_data()
  sp <- suppressWarnings(spatial_bym2(dd$A, group_var = "region"))
  fit <- suppressWarnings(tulpa(
    y ~ x + spatial(region), dd$d, family = "poisson", mode = "hmc",
    spatial = sp,
    control = list(n_chains = 1, n_iter = 400, warmup = 200, seed = 1)))
  fe <- field_effects(fit)
  expect_equal(nrow(fe), 16L)
  expect_true(all(fe$source == "sampled"))
  expect_true(all(is.finite(fe$estimate)))
  expect_true(all(fe$sd > 0))
  expect_true(all(fe$conf.low < fe$estimate & fe$estimate < fe$conf.high))

  # sigma * (sqrt(rho) * s * phi_c + sqrt(1 - rho) * theta), phi centred over
  # the whole graph: the contribution the BYM2 density puts into eta.
  D <- fit$draws
  phi <- D[, sprintf("phi_spatial[%d]", 1:16)]
  theta <- D[, sprintf("theta_spatial[%d]", 1:16)]
  sigma <- exp(D[, "log_sigma_spatial"])
  rho <- stats::plogis(D[, "logit_rho_bym2"])
  phi_c <- phi - rowMeans(phi)
  ref <- sigma * (sqrt(rho) * sp$scale_factor * phi_c) + sigma * sqrt(1 - rho) * theta
  expect_equal(fe$estimate, unname(colMeans(ref)), tolerance = 1e-8)
})

test_that("gibbs binomial ICAR: the sampled field columns per unit", {
  skip_on_cran()
  K <- 20L
  d <- .fe_ring_data(K)
  fit <- suppressWarnings(tulpa(
    yb ~ x + spatial(region), data = d, family = "binomial",
    spatial = list(type = "icar", adjacency = .fe_ring(K)),
    mode = "gibbs", control = list(n_iter = 400, warmup = 200, seed = 1)))
  fe <- field_effects(fit)
  expect_equal(nrow(fe), K)
  expect_true(all(fe$source == "sampled"))
  expect_true(all(fe$sd > 0))
  ph <- fit$draws[, sprintf("phi_spatial[%d]", seq_len(K)), drop = FALSE]
  expect_equal(fe$estimate, unname(colMeans(ph)))
})

test_that("nested_laplace ICAR: the grid-weighted field per unit", {
  skip_on_cran()
  K <- 20L
  d <- .fe_ring_data(K)
  fit <- suppressWarnings(tulpa(
    yc ~ x + spatial(region), data = d, family = "poisson",
    spatial = list(type = "icar", adjacency = .fe_ring(K)),
    mode = "nested_laplace", control = list(n_threads = 1L)))
  fe <- field_effects(fit)
  expect_equal(nrow(fe), K)
  expect_true(all(fe$source %in% c("mode", "mixture")))
  M <- fit$modes
  w <- as.numeric(fit$weights)
  keep <- is.finite(w) & w > 0 & is.finite(rowSums(M))
  ref <- crossprod(w[keep] / sum(w[keep]),
                   M[keep, (ncol(M) - K + 1L):ncol(M), drop = FALSE])
  expect_equal(fe$estimate, as.numeric(ref))
  # The field tracks the generating wave around the ring.
  truth <- as.numeric(scale(sin(2 * pi * seq_len(K) / K)))
  expect_gt(stats::cor(fe$estimate, truth), 0.7)
})
