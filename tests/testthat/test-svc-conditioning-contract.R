test_that("SVC conditioning policies describe the same dense prior in both coordinates", {
  set.seed(197)
  n <- 8L
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  X <- cbind(1, x = d$x); y <- rnorm(n)
  specification <- validate_svc(spatial_svc(~ lon + lat, terms = ~ x, nn = n - 1L), d, X)
  base <- .svc_sampler_spec(specification, X)
  base$range_prior <- "uniform"; base$range_lower <- .02; base$range_upper <- 2
  policies <- list(native = c(1e-4, 1e-4, 1e-4, 1),
    ridge = c(1e-8, 0, 1e-10, 0), floored = c(.01, .002, .3, 1))
  for (policy in policies) for (coordinate in c("centered", "noncentered")) {
    spec <- base; spec$parameterization <- coordinate
    spec$neighbor_jitter <- policy[1L]; spec$marginal_nugget <- policy[2L]
    spec$variance_floor <- policy[3L]; spec$blend_variance_floor <- policy[4L] == 1
    q <- .1 * sin(seq_len(4L + n))
    q0 <- q; q0[3L] <- -.7; q0[4L] <- .5
    evaluate <- function(point) {
      r <- tulpa:::cpp_test_field_coordinates(y, X, svc_spec = spec, point = point)
      sigma2 <- exp(point[3L]); u <- plogis(point[4L]); phi <- .02 + 1.98 * u
      C <- sigma2 * exp(-as.matrix(dist(cbind(d$lon, d$lat))) / phi)
      order <- as.integer(spec$nn_order) + 1L
      nn <- spec$nn
      indices <- matrix(spec$nn_idx, n, nn, byrow = TRUE)
      A <- diag(n); variance <- rep(sigma2 + policy[2L], n)
      for (i in 2:n) {
        neighbors <- indices[i, ]; neighbors <- neighbors[neighbors > 0L]
        locations <- order[neighbors]
        at <- order[i]
        weight <- solve(C[locations, locations, drop = FALSE] + diag(policy[1L], length(locations)), C[locations, at])
        A[at, locations] <- -weight
        raw <- sigma2 + policy[2L] - sum(C[locations, at] * weight)
        variance[at] <- if (raw < policy[3L]) {
          if (policy[4L] == 1) .99 * policy[3L] + .01 * max(raw, 0) else policy[3L]
        } else raw
      }
      inverse <- solve(A)
      Sigma <- inverse %*% diag(variance) %*% t(inverse)
      z <- tail(point, n)
      field <- if (coordinate == "noncentered") drop(inverse %*% (sqrt(variance) * z)) else z
      density <- if (coordinate == "noncentered") sum(dnorm(z, log = TRUE)) else
        -.5 * (n * log(2 * pi) + as.numeric(determinant(Sigma)$modulus) + drop(crossprod(field, solve(Sigma, field))))
      rate <- -log(.01)
      prior <- -.5 * sum(point[1:2]^2) / 2.5^2 + log(rate / 2) + .5 * point[3L] - rate * sqrt(sigma2) + log(u) + log1p(-u)
      expected <- prior + density + sum(dnorm(y, drop(X %*% point[1:2]) + d$x * (field - mean(field)), 1, log = TRUE))
      expect_equal(tail(r$stored, n), field - mean(field), tolerance = 1e-10)
      expect_true(r$verified)
      expect_equal(r$gradient, r$numerical, tolerance = 5e-5)
      list(actual = r$raw_lp, expected = expected)
    }
    r <- evaluate(q); zero <- evaluate(q0)
    expect_equal(r$actual - zero$actual, r$expected - zero$expected, tolerance = 1e-9)
  }
})