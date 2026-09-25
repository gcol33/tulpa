# temporal() reads a field back off a fit: the right block (gcol33/tulpa#903),
# with its within-cell variance (gcol33/tulpa#904), as the field and not its
# whitened coordinates, labelled by its own time values (gcol33/tulpa#905).

lattice_adj_rb <- function(nr) {
  S <- nr * nr; A <- matrix(0, S, S); id <- function(i, j) (j - 1L) * nr + i
  for (i in 1:nr) for (j in 1:nr) {
    if (i < nr) A[id(i, j), id(i + 1L, j)] <- A[id(i + 1L, j), id(i, j)] <- 1
    if (j < nr) A[id(i, j), id(i, j + 1L)] <- A[id(i, j + 1L), id(i, j)] <- 1
  }
  A
}

sim_ar1_rb <- function(Tn, rho, sig, seed) {
  set.seed(seed)
  f <- numeric(Tn); f[1] <- rnorm(1, 0, sig / sqrt(1 - rho^2))
  for (t in 2:Tn) f[t] <- rho * f[t - 1] + rnorm(1, 0, sig)
  f
}

test_that("a single-block field is the tail of the latent vector", {
  blk <- list(type = "ar1", n_times = 10L, n_groups = 1L, role = "temporal")
  expect_identical(tulpa:::.nl_role_cols(blk, NULL, 13L, "temporal"), 4:13)
  expect_null(tulpa:::.nl_role_cols(blk, NULL, 13L, "spatial"))
  # A multi-block layout is read at the solve's own offsets, by role.
  blocks <- list(list(type = "icar", role = "spatial"),
                 list(type = "ar1", role = "temporal"),
                 list(type = "rw2"))                       # an s(x) smoother
  expect_identical(tulpa:::.nl_role_cols(blocks, c(2L, 38L, 48L, 60L), 60L,
                                         "temporal"), 39:48)
  expect_identical(tulpa:::.nl_block_roles(blocks),
                   c("spatial", "temporal", "other"))
  # Untagged (a hand-built prior): classified by type, and iid is not temporal.
  expect_identical(tulpa:::.nl_block_roles(list(list(type = "rw1"),
                                                list(type = "iid"))),
                   c("temporal", "other"))
})

test_that("temporal() reads the temporal block behind a spatial one", {
  skip_on_cran()
  A <- lattice_adj_rb(6L); S <- nrow(A)
  set.seed(41); phi <- rnorm(S, 0, 0.7); Tn <- 10L; f <- 1.2 * sin(1:Tn)
  d <- expand.grid(region = 1:S, time = 1:Tn); set.seed(43)
  d$x <- rnorm(nrow(d))
  d$y <- rpois(nrow(d), exp(0.3 + 0.5 * d$x + phi[d$region] + f[d$time]))
  ft <- suppressWarnings(tulpa(
    y ~ x + spatial(region), data = d, family = "poisson",
    mode = "nested_laplace", spatial = spatial_car(A, group_var = "region"),
    temporal = temporal_ar1("time"),
    control = list(n_threads = 1, progress = FALSE)))
  expect_identical(tulpa:::.nl_temporal_field_cols(ft), 2L + S + seq_len(Tn))
  tp <- temporal(ft, summary = TRUE)
  expect_gt(cor(tp$mean, f), 0.95)
  expect_identical(ft$grid_field_var_cols, 2L + S + seq_len(Tn))
  # temporal_corr() reports the AR1 block and nothing of the spatial one.
  tc <- temporal_corr(ft)
  expect_setequal(rownames(tc), c("precision", "rho_ar1"))
  expect_setequal(rownames(spatial_range(ft)), "sigma")
})

test_that("nested temporal() intervals carry the within-cell variance", {
  skip_on_cran()
  Tn <- 30L
  f <- sim_ar1_rb(Tn, 0.8, 0.6, 21)
  set.seed(22)
  d <- data.frame(time = rep(1:Tn, each = 5), x = rnorm(5 * Tn))
  d$y <- 0.4 + 0.7 * d$x + f[d$time] + rnorm(nrow(d), sd = 0.3)
  ft <- tulpa(y ~ x, data = d, phi = 0.09, mode = "nested_laplace",
              temporal = temporal_ar1("time"), control = list(n_threads = 1))
  tp <- temporal(ft, summary = TRUE)
  # The summary is the exact mixture of the per-cell Gaussians.
  cols <- tulpa:::.nl_temporal_field_cols(ft)
  w <- ft$weights / sum(ft$weights)
  mu <- ft$modes[, cols]; v <- ft$grid_field_var
  m <- colSums(w * mu)
  expect_equal(tp$mean, m, tolerance = 1e-10)
  expect_equal(tp$sd, sqrt(colSums(w * (v + mu^2)) - m^2), tolerance = 1e-10)
  # Well above the between-cell spread alone (the old read, sd ~0.05).
  between <- sqrt(colSums(w * mu^2) - m^2)
  expect_true(all(tp$sd > 2 * between))
  expect_gte(mean(f >= tp$q2.5 & f <= tp$q97.5), 0.8)
})

test_that("nested temporal() intervals cover at the nominal rate", {
  skip_if_not_slow()
  Tn <- 30L
  cover <- vapply(1:12, function(s) {
    f <- sim_ar1_rb(Tn, 0.8, 0.6, 100 + s)
    set.seed(200 + s)
    d <- data.frame(time = rep(1:Tn, each = 5), x = rnorm(5 * Tn))
    d$y <- 0.4 + 0.7 * d$x + f[d$time] + rnorm(nrow(d), sd = 0.3)
    ft <- tulpa(y ~ x, data = d, phi = 0.09, mode = "nested_laplace",
                temporal = temporal_ar1("time"), control = list(n_threads = 1))
    tp <- temporal(ft, summary = TRUE)
    mean(f >= tp$q2.5 & f <= tp$q97.5)
  }, numeric(1))
  expect_gt(mean(cover), 0.88)
})

test_that("a temporal GP's summary is labelled by its own time values", {
  set.seed(1); tt <- sort(round(runif(12, 0, 100), 1))
  d <- data.frame(t = rep(tt, each = 2))
  spec <- validate_temporal_gp(temporal_gp("t"), d)
  expect_equal(spec$time_levels, tt)
})

test_that("temporal() on a non-centered temporal GP returns the field", {
  skip_if_not_slow()
  set.seed(1); Tn <- 30L; tt <- sort(round(runif(Tn, 0, 100), 1)); set.seed(61)
  f <- as.numeric(t(chol(0.64 * exp(-abs(outer(tt, tt, "-")) / 15) +
                           diag(1e-8, Tn))) %*% rnorm(Tn))
  d <- data.frame(t = rep(tt, each = 4)); d$x <- rnorm(nrow(d))
  d$y <- 0.5 + 0.5 * d$x + f[match(d$t, tt)] + rnorm(nrow(d), sd = 0.3)
  ft <- tulpa(y ~ x, data = d, phi = 0.09, mode = "hmc",
              temporal = temporal_gp("t"),
              control = list(n_iter = 1000, warmup = 500, n_chains = 1, seed = 1))
  tp <- temporal(ft, summary = TRUE)
  expect_gt(cor(tp$mean, f), 0.9)
  expect_equal(tp$time, tt)
})
