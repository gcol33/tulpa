test_that("areal constraints and island priors match dense Gaussian references", {
  S <- 6L
  components <- list(c(1L, 3L, 5L), c(2L, 4L), 6L)
  adjacency <- matrix(0, S, S)
  for (edge in list(c(1L, 3L), c(3L, 5L), c(2L, 4L)))
    adjacency[edge[1], edge[2]] <- adjacency[edge[2], edge[1]] <- 1
  Q <- diag(rowSums(adjacency)) - adjacency
  group <- rep(seq_len(S), each = 3L)
  X <- cbind(1, seq(-1, 1, length.out = length(group)))
  y <- sin(seq_along(group))
  phi <- c(.3, -.2, -.1, .4, .6, -.3)
  theta <- c(-.2, .1, .4, -.5, .3, .2)
  weights <- c(1, 2, 1, 2, 1, 3)
  for (type in c("icar", "bym2")) for (component_centering in c(FALSE, TRUE)) {
    spec <- list(type = type, parameterization = "auto", spatial_idx = group,
      n_spatial_units = S,
      adj_row_ptr = c(0L, 1L, 2L, 4L, 5L, 6L, 6L),
      adj_col_idx = c(2L, 3L, 0L, 4L, 1L, 2L),
      n_neighbors = as.integer(rowSums(adjacency)), scale_factor = 1.7,
      node_prec = weights, component_centering = component_centering)
    w <- if (type == "bym2") weights else rep(1, S)
    H <- diag(w) %*% Q
    P <- diag(S)
    for (cc in components) {
      coefficient <- if (component_centering && length(cc) > 1L) 1 else w[cc[1]]
      H[cc, cc] <- H[cc, cc] + coefficient / length(cc)
      if (component_centering && length(cc) > 1L)
        P[cc, cc] <- P[cc, cc] - 1 / length(cc)
    }
    if (!component_centering) P <- P - 1 / S
    p <- ncol(X) + if (type == "bym2") 2L + 2L * S else 1L + S
    zero <- numeric(p)
    point <- zero
    start <- ncol(X) + if (type == "bym2") 2L else 1L
    point[start + seq_len(S)] <- phi
    effect <- drop(P %*% phi)
    quadratic <- drop(crossprod(phi, H %*% phi))
    if (type == "bym2") {
      point[start + S + seq_len(S)] <- theta
      effect <- (1.7 * effect + theta) / sqrt(2)
      quadratic <- quadratic + sum(theta^2)
    }
    base <- tulpa:::cpp_test_field_coordinates(y, X, spatial_spec = spec, point = zero)
    r <- tulpa:::cpp_test_field_coordinates(y, X, spatial_spec = spec, point = point)
    expected <- sum(dnorm(y, effect[group], log = TRUE) - dnorm(y, log = TRUE)) -
      .5 * quadratic
    expect_equal(r$lp - base$lp, expected, tolerance = 1e-10)
    expect_equal(r$spatial_log_normalizer,
      .5 * as.numeric(determinant(H)$modulus) -
        .5 * S * log(2 * pi) * if (type == "bym2") 2 else 1, tolerance = 1e-10)
    expect_true(r$verified)
    expect_lt(max(abs(r$gradient - r$numerical) / pmax(1, abs(r$gradient))), 1e-4)
  }
})
