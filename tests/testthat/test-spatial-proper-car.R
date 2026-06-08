# Inline proper-CAR varying-coefficient fields: spatial(graph, ~ ... || cell,
# proper = TRUE) (gcol33/tulpa#90). Each field gains its own (sigma, rho_car):
# Q = D - rho_car * W with rho_car estimated, instead of the intrinsic rho = 1.

# 2D lattice adjacency (rook contiguity) -- rho_car is far better identified on a
# grid than on a chain.
.grid_adj <- function(nx, ny) {
  n <- nx * ny
  W <- matrix(0L, n, n)
  id <- function(i, j) (j - 1L) * nx + i
  for (i in seq_len(nx)) for (j in seq_len(ny)) {
    if (i < nx) { a <- id(i, j); b <- id(i + 1L, j); W[a, b] <- W[b, a] <- 1L }
    if (j < ny) { a <- id(i, j); b <- id(i, j + 1L); W[a, b] <- W[b, a] <- 1L }
  }
  W
}

# Draw z ~ N(0, (tau (D - rho W))^-1) (proper CAR), then centre.
.sim_proper_car <- function(W, rho, tau) {
  n <- nrow(W)
  Q <- tau * (diag(rowSums(W)) - rho * W)
  U <- chol(Q)                       # Q = U'U
  z <- backsolve(U, stats::rnorm(n)) # cov = (U'U)^-1 = Q^-1
  z - mean(z)
}

# --------------------------------------------------------------------------- #
# (1) Construction: proper = TRUE builds car_proper blocks carrying rho_bounds #
# --------------------------------------------------------------------------- #

test_that("proper = TRUE builds car_proper blocks with rho_bounds", {
  adj <- .grid_adj(4L, 4L)
  f <- spatial(graph = adj, formula = ~ 1 + time || cell, proper = TRUE)
  expect_true(f$proper)
  d <- data.frame(cell = rep(seq_len(16L), each = 2L),
                  time = rnorm(32L))
  blocks <- tulpa:::.spatial_field_blocks(f, d)
  expect_length(blocks, 2L)                       # intercept + time slope
  expect_true(all(vapply(blocks, function(b) b$type, "") == "car_proper"))
  expect_true(all(vapply(blocks, function(b) !is.null(b$rho_bounds), NA)))
  # Slope column carries the per-row svc_weight; the intercept does not.
  expect_null(blocks[[1]]$svc_weight)
  expect_false(is.null(blocks[[2]]$svc_weight))

  # proper = FALSE stays intrinsic icar.
  f0 <- spatial(graph = adj, formula = ~ 1 || cell, proper = FALSE)
  b0 <- tulpa:::.spatial_field_blocks(f0, d)
  expect_identical(b0[[1]]$type, "icar")
  expect_null(b0[[1]]$rho_bounds)
})

test_that("proper-CAR field spec prints its structure", {
  adj <- .grid_adj(4L, 4L)
  f <- spatial(graph = adj, formula = ~ 1 || cell, proper = TRUE)
  out <- paste(capture.output(print(f)), collapse = "\n")
  expect_match(out, "proper CAR")
})

# --------------------------------------------------------------------------- #
# (2) Recovery: the field and rho_car vs simulated truth                      #
# --------------------------------------------------------------------------- #

test_that("spatial(proper = TRUE) recovers the field and a high rho_car", {
  skip_on_cran()
  skip_if_fast()

  nx <- ny <- 10L
  adj <- .grid_adj(nx, ny)
  n_s <- nx * ny
  rho_true <- 0.9

  cors <- numeric(0)
  rho_meds <- numeric(0)
  for (seed in 1:3) {
    set.seed(seed)
    z <- .sim_proper_car(adj, rho = rho_true, tau = 1.0)
    n_per <- 30L
    cell <- rep(seq_len(n_s), each = n_per)
    N <- length(cell)
    b0 <- 0.4
    y <- stats::rnorm(N, b0 + z[cell], 0.5)
    d <- data.frame(y = y, cell = cell)

    fit <- suppressWarnings(tulpa(
      y ~ spatial(graph = adj, formula = ~ 1 || cell, proper = TRUE),
      data = d, family = "gaussian", mode = "laplace",
      control = list(n_draws = 300L)))

    expect_s3_class(fit, "tulpa_spatial_field_fit")
    z_hat <- fit$spatial_fields[["cell.Intercept"]]$mean
    cors <- c(cors, abs(cor(z_hat, z)))

    h <- fit$spatial_field_hypers[["cell.Intercept"]]
    expect_false(is.null(h$rho_car))           # rho_car is exposed per field
    expect_length(h$rho_car, 3L)               # (2.5%, 50%, 97.5%)
    rho_meds <- c(rho_meds, h$rho_car[2L])
  }

  # Field recovered every seed.
  expect_true(all(cors > 0.7))
  # rho_car recovers substantial positive autocorrelation (true rho = 0.9);
  # the proper-CAR rho is weakly identified, so the bar is "clearly positive",
  # not a tight point match.
  expect_gt(median(rho_meds), 0.5)
})

test_that("proper-CAR fit print reports rho_car per field", {
  skip_on_cran()
  skip_if_fast()
  set.seed(11)
  nx <- ny <- 8L
  adj <- .grid_adj(nx, ny)
  n_s <- nx * ny
  z <- .sim_proper_car(adj, rho = 0.85, tau = 1.0)
  cell <- rep(seq_len(n_s), each = 25L)
  y <- stats::rnorm(length(cell), 0.2 + z[cell], 0.5)
  fit <- suppressWarnings(tulpa(
    y ~ spatial(graph = adj, formula = ~ 1 || cell, proper = TRUE),
    data = data.frame(y = y, cell = cell), family = "gaussian",
    mode = "laplace", control = list(n_draws = 200L)))
  out <- paste(capture.output(print(fit)), collapse = "\n")
  expect_match(out, "rho_car")
  expect_match(out, "proper CAR")
})
