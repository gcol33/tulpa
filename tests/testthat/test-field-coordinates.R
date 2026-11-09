coordinate_reference <- function(r) {
  p <- length(r$q)
  J <- diag(p)
  log_det <- 0
  jacobian_score <- numeric(p)
  x <- r$q
  for (e in r$blocks) {
    index <- seq.int(e$begin + 1L, e$end)
    scale <- e$scale_index + 1L
    width <- e$group_width
    a <- sqrt((e$scaled_information + e$fixed_information) /
      (e$scaled_information * exp(e$scale_sign * r$q[scale]) + e$fixed_information))
    derivative <- -.5 * e$scale_sign * e$scaled_information *
      exp(e$scale_sign * r$q[scale]) /
      (e$scaled_information * exp(e$scale_sign * r$q[scale]) + e$fixed_information)
    for (start in seq.int(1L, length(index), by = width)) {
      ii <- index[seq.int(start, length.out = width)]
      P <- matrix(0, width, width)
      rank <- width
      if (e$preserve_constant) {
        P <- P + matrix(1 / width, width, width)
        rank <- rank - 1L
      }
      if (e$preserve_linear && width > 1L) {
        time <- seq_len(width) - (width + 1) / 2
        P <- P + tcrossprod(time) / sum(time^2)
        rank <- rank - 1L
      }
      A <- P + a * (diag(width) - P)
      J[ii, ii] <- A
      x[ii] <- drop(A %*% r$q[ii])
      J[ii, scale] <- derivative * drop((diag(width) - P) %*% x[ii])
      log_det <- log_det + rank * log(a)
      jacobian_score[scale] <- jacobian_score[scale] + rank * derivative
    }
  }
  list(x = x, log_det = log_det, dense_log_det = as.numeric(determinant(J)$modulus),
       gradient = drop(crossprod(J, r$raw_gradient)) + jacobian_score)
}

test_that("field transports match dense Jacobians and the engine gradient check", {
  set.seed(1801)
  n <- 40L
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  X <- cbind("(Intercept)" = 1, x = d$x)
  y <- .2 + .5 * d$x + rnorm(n)
  time_idx <- as.integer(rep_len(1:5, n))
  spatial <- list(type = "icar", parameterization = "auto", n_spatial_units = 4L,
    spatial_idx = as.integer(rep_len(1:4, n)), adj_row_ptr = c(0L, 1L, 3L, 5L, 6L),
    adj_col_idx = c(1L, 0L, 2L, 1L, 3L, 2L), n_neighbors = c(1L, 2L, 2L, 1L))
  temporal <- list(type = "rw1", parameterization = "auto", time_idx = time_idx,
    n_times = 5L, n_groups = 1L, cyclic = FALSE)
  gp <- .gp_sampler_spec(validate_gp(spatial_gp(~ lon + lat, nn = 4L), d))
  svc <- .svc_sampler_spec(validate_svc(spatial_svc(~ lon + lat, terms = ~ x,
    nn = 4L), d, X), X)
  tvc <- list(n_times = 5L, n_groups = 1L, n_tvc = 1L, time_index = time_idx,
    group_index = rep(1L, n), tvc_indices = 2L, X_tvc = d$x, structure = "rw1")
  ms <- list(type = "multiscale", parameterization = "auto", time_index = time_idx,
    n_times = 5L, n_groups = 1L, trend = "rw1", seasonal = 3L, short_term = "ar1")
  tgp <- c(temporal, list(time_values = c(0, .1, .4, 1, 2), cov = "exponential",
    phi_prior_lower = .01, phi_prior_upper = 10))
  tgp$type <- "gp"
  cases <- list(list(spatial_spec = spatial), list(spatial_spec = gp),
    list(svc_spec = svc), list(temporal_spec = temporal),
    list(temporal_spec = ms), list(temporal_spec = tgp), list(tvc_spec = tvc))
  car <- spatial; car$type <- "car_proper"
  W <- matrix(0, 4, 4)
  for (s in 1:3) W[s, s + 1L] <- W[s + 1L, s] <- 1
  car$adj_eigenvalues <- eigen(W / sqrt(outer(rowSums(W), rowSums(W))),
    symmetric = TRUE, only.values = TRUE)$values
  car$rho_lower <- 0; car$rho_upper <- 1
  cases <- c(cases, list(list(spatial_spec = car)))
  dense_tgp <- tgp; dense_tgp$cov <- "matern"; dense_tgp$nu <- 1.5
  cases <- c(cases, list(list(temporal_spec = dense_tgp)))
  ms_rw2 <- ms; ms_rw2$trend <- "rw2"
  cases <- c(cases, list(list(temporal_spec = ms_rw2)))
  for (type in c("rw2", "ar1")) {
    v <- tvc; v$structure <- type
    cases <- c(cases, list(list(tvc_spec = v)))
  }
  for (type in c("rw2", "ar1")) {
    t <- temporal; t$type <- type
    cases <- c(cases, list(list(temporal_spec = t)))
  }
  for (type in c("rw1", "rw2")) {
    t <- temporal; t$type <- type; t$cyclic <- TRUE
    cases <- c(cases, list(list(temporal_spec = t)))
    v <- tvc; v$structure <- type; v$cyclic <- TRUE
    cases <- c(cases, list(list(tvc_spec = v)))
  }
  for (case in cases) {
    initial_blocks <- NULL
    for (shift in c(.1, .8)) {
      r <- do.call(tulpa:::cpp_test_field_coordinates,
        c(list(y = y, X = X, shift = shift), case))
      expect_gt(length(r$blocks), 0L)
      if (is.null(initial_blocks)) initial_blocks <- r$blocks
      else expect_identical(r$blocks, initial_blocks)
      reference <- coordinate_reference(r)
      expect_equal(r$x, reference$x, tolerance = 1e-11)
      expect_equal(r$jacobian, reference$log_det, tolerance = 1e-11)
      expect_equal(r$jacobian, reference$dense_log_det, tolerance = 1e-11)
      expect_equal(r$gradient, reference$gradient, tolerance = 1e-9)
      expect_equal(r$lp, r$raw_lp + r$jacobian, tolerance = 1e-11)
      expect_equal(r$inverse, r$q, tolerance = 1e-11)
      expect_lt(max(abs(r$gradient - r$numerical) / pmax(1, abs(r$gradient))), 1e-4)
      expect_lt(max(abs(r$prior_gradient - r$numerical_prior) /
        pmax(1, abs(r$prior_gradient))), 1e-4)
      expect_true(r$verified)
      expect_true(r$numerical_dispatch)
      if (!is.null(case$temporal_spec) && case$temporal_spec$type %in% c("rw1", "rw2")) {
        e <- r$blocks[[1]]
        expect_equal(mean(r$stored[seq.int(e$begin + 1L, e$end)]), 0,
                     tolerance = 1e-12)
      }
    }
  }
})

test_that("field constructors select automatic sampler coordinates", {
  expect_identical(temporal_rw1("time")$parameterization, "auto")
  expect_identical(temporal_rw2("time")$parameterization, "auto")
  expect_identical(temporal_ar1("time")$parameterization, "auto")
  expect_identical(temporal_gp("time")$parameterization, "auto")
  expect_identical(temporal_multiscale("time")$parameterization, "auto")
  expect_identical(spatial_gp(~ lon + lat)$parameterization, "auto")
  expect_identical(spatial_svc(~ lon + lat)$parameterization, "auto")
})

test_that("reported density uses native field coordinates", {
  skip_if_not_slow()
  y <- seq(-1, 1, length.out = 40)
  tp <- list(type = "rw1", parameterization = "auto", n_times = 5L,
    n_groups = 1L, time_idx = as.integer(rep_len(1:5, length(y))))
  r <- tulpa:::cpp_test_field_coordinates(y, matrix(1, length(y), 1),
    temporal_spec = tp, n_iter = 400L)
  expect_equal(r$last_reported_lp, r$last_native_lp, tolerance = 1e-10)
})
