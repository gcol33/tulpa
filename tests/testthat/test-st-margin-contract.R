test_that("spatiotemporal margin coordinates match dense Gaussian operators", {
  set.seed(195)
  graphs <- list(list(S = 3L, edges = matrix(c(1L, 2L, 2L, 3L), ncol = 2L, byrow = TRUE),
    components = list(1:3)), list(S = 5L,
    edges = matrix(c(1L, 3L, 2L, 4L), ncol = 2L, byrow = TRUE),
    components = list(c(1L, 3L), c(2L, 4L), 5L)))
  for (graph in graphs) {
  S <- graph$S
  nt <- 4L
  nfield <- S * nt
  index <- rep(seq_len(nfield), each = 2L)
  N <- length(index)
  X <- cbind(1, rnorm(N))
  y <- rnorm(N)
  adjacency <- matrix(0L, S, S)
  adjacency[graph$edges] <- 1L
  adjacency <- adjacency + t(adjacency)
  spatial <- diag(rowSums(adjacency)) - adjacency
  row_ptr <- c(0L, cumsum(rowSums(adjacency)))
  col_idx <- unlist(lapply(seq_len(S), function(s) which(adjacency[s, ] != 0L)))
  Pt <- diag(nt) - matrix(1 / nt, nt, nt)
  specifications <- list(c(1L, 1L, 0L), c(2L, 1L, 0L), c(2L, 2L, 0L),
    c(2L, 2L, 1L), c(2L, 3L, 0L), c(3L, 1L, 0L),
    c(4L, 1L, 0L), c(4L, 2L, 0L), c(4L, 2L, 1L), c(4L, 3L, 0L))
  for (case in specifications) for (coordinate in 0:1) {
    type <- case[1L]
    time_type <- case[2L]
    cyclic <- case[3L] == 1L
    components <- if (type %in% c(3L, 4L)) graph$components else list(seq_len(S))
    Ps <- diag(S)
    for (component in components) Ps[component, component] <-
      Ps[component, component] - 1 / length(component)
    projection <- kronecker(Ps, Pt)
    contrasts <- S - length(components)
    spec <- list(type = type, temporal_type = time_type, cyclic = cyclic,
      parameterization = coordinate, n_spatial = S, n_times = nt,
      st_flat = as.integer(index), adj_row_ptr = as.integer(row_ptr),
      adj_col_idx = as.integer(col_idx))
    r <- tulpa:::cpp_test_field_coordinates(y, X, st_spec = spec)
    q <- r$x
    log_tau <- q[3L]
    tau <- exp(log_tau)
    ar1 <- type %in% c(2L, 4L) && time_type == 3L
    u <- if (ar1) plogis(q[4L]) else .5
    rho <- 2 * u - 1
    D <- diff(diag(nt), differences = if (time_type == 2L) 2L else 1L)
    if (cyclic) {
      D <- diag(nt) - diag(nt)[c(nt, seq_len(nt - 1L)), ]
      if (time_type == 2L) D <- D %*% D
    }
    if (ar1) {
      D <- diag(nt)
      D[cbind(2:nt, 1:(nt - 1L))] <- -rho
      D[1L, 1L] <- sqrt(max(1 - rho^2, 1e-10))
    }
    temporal <- crossprod(D)
    base <- switch(type, diag(nfield), kronecker(diag(S), temporal),
      kronecker(spatial, diag(nt)), kronecker(spatial, temporal))
    H <- projection %*% base %*% projection + diag(nfield) - projection
    field <- tail(q, nfield)
    centered <- drop(projection %*% field)
    effect <- centered / if (coordinate == 1L) sqrt(tau) else 1
    rank <- nfield - if (type %in% c(2L, 4L) && time_type == 2L && !cyclic) contrasts else 0L
    expect_equal(sum(eigen(H, symmetric = TRUE, only.values = TRUE)$values > 1e-8), rank)
    normalizer <- .5 * (if (coordinate == 1L) rank - nfield else rank) * log_tau
    if (ar1) {
      U <- qr.Q(qr(cbind(1, diag(nt)[, -nt])))[, -1L, drop = FALSE]
      normalizer <- normalizer + .5 * contrasts * as.numeric(determinant(crossprod(U, temporal %*% U))$modulus)
    }
    rate <- -log(.01)
    pc <- log(rate) - log(2) - .5 * log_tau - rate * exp(-.5 * log_tau)
    prior <- -.5 * sum(q[1:2]^2) / 2.5^2 + pc + normalizer -
      .5 * (if (coordinate == 1L) 1 else tau) * drop(crossprod(field, H %*% field))
    if (ar1) prior <- prior + log(u) + log1p(-u)
    eta <- drop(X %*% q[1:2]) + effect[index]
    expected <- prior + sum(dnorm(y, eta, 1, log = TRUE))
    expect_equal(r$raw_lp, expected, tolerance = 1e-10)
    expect_equal(tail(r$stored, nfield), effect, tolerance = 1e-12)
    expect_true(r$verified)
    expect_equal(r$gradient, r$numerical, tolerance = 3e-5)
    expect_equal(rowSums(matrix(effect, nt)), rep(0, nt), tolerance = 1e-12)
    expect_equal(colSums(matrix(effect, nt)), rep(0, S), tolerance = 1e-12)
  }
  }
})
