# What a reported median / interval means depends on WHAT KIND of node set the
# outer integrator left behind (gcol33/tulpa#308, #309, #310).
#
# A tensor grid's uniform cells discretize the posterior density, so the
# cumulative node weights are a CDF and the discrete weighted quantile is the
# summary. A central-composite design is a moment rule: its nodes sit where they
# reproduce the integrand's first two moments and carry no probability mass, so
# the cumulative design weight is not a CDF and a discrete quantile over it can
# only ever return the design's own extent -- `theta_hat +/- 1.1 sqrt(d)`
# posterior SDs, a fixed multiple that has nothing to do with 0.95.
#
# These tests pin the dispatcher, the per-domain coordinate, the joint axis
# registry that names each axis's domain, and the two front doors that read an
# interval off a CCD: the joint multi-block per-axis summary and the inline
# spatial() bar field / MCAR summary.

# ---------------------------------------------------------------- dispatcher

test_that(".nl_summary_quantile reads a density support as a weighted quantile", {
  v <- c(1, 2, 3, 4, 5)
  w <- rep(0.2, 5)
  # Named, because this test is about the SUPPORT dispatch and the within-cell
  # default moved to `box_uniform` at 0.0.188 (gcol33/tulpa#357).
  expect_identical(
    .nl_summary_quantile(v, w, c(0.025, 0.5, 0.975), "positive", "density",
                         "chord"),
    .nl_wtd_quantile(v, w, c(0.025, 0.5, 0.975), outside = "extend"))
  # The default read is still a CDF read of the same weights on the same axis,
  # sorted and bracketed by the same outer edges.
  q <- .nl_summary_quantile(v, w, c(0.025, 0.5, 0.975), "positive", "density")
  expect_false(is.unsorted(q))
  expect_gt(q[1L], .nl_cell_edges(v, "positive")[1L])
})

test_that(".nl_summary_quantile reads a moment rule from its moments", {
  set.seed(1)
  v <- exp(rnorm(9, 0.3, 0.4))
  w <- runif(9); w <- w / sum(w)
  q <- .nl_summary_quantile(v, w, c(0.025, 0.5, 0.975), "positive", "moment_rule")
  m <- sum(w * log(v))
  s <- sqrt(sum(w * log(v)^2) - m^2)
  expect_equal(q, exp(m + qnorm(c(0.025, 0.5, 0.975)) * s))
})

test_that("a moment interval leaves the node range where the design confines it", {
  # The scalar CCD of gcol33/tulpa#308: three nodes at 0 and +/- 1.1 whitened SDs
  # with the matching design weights, on a flat integrand. The design's own
  # second moment sets the interval half-width at 1.96 of it, which reaches past
  # the 1.1 the node positions span.
  ccd <- ccd_grid(1L, f_0 = 1.1)
  w   <- ccd_weights(ccd)
  s   <- 0.4
  v   <- exp(as.numeric(ccd$z) * s)
  q <- .nl_summary_quantile(v, w, c(0.025, 0.5, 0.975), "positive", "moment_rule")
  su <- sqrt(sum(w * log(v)^2) - sum(w * log(v))^2)
  expect_equal(unname(q[3] / q[2]), exp(qnorm(0.975) * su))
  expect_lt(q[1], min(v))
  expect_gt(q[3], max(v))
  # The same nodes read as a density are confined by the design's own GEOMETRY
  # and say nothing about its moments, which is the defect. The confinement is
  # the node range widened by the outer cells' half-spacing and no further
  # (gcol33/tulpa#353); before that it was the node range exactly.
  qd <- .nl_summary_quantile(v, w, c(0.025, 0.5, 0.975), "positive", "density")
  e <- .nl_cell_edges(sort(v))
  expect_lt(qd[1], min(v)); expect_gt(qd[1], e[1])
  expect_gt(qd[3], max(v)); expect_lt(qd[3], e[2])
  expect_false(isTRUE(all.equal(unname(qd), unname(q))))
})

test_that("a moment rule on an axis with no known domain withholds the number", {
  v <- c(0.2, 0.4, 0.6)
  w <- c(0.3, 0.4, 0.3)
  q <- .nl_summary_quantile(v, w, c(0.025, 0.5, 0.975), NA_character_,
                            "moment_rule")
  expect_true(is.na(q[1]))
  expect_true(is.na(q[3]))
  # ... rather than returning a number the design's own geometry fixes, which is
  # what the same weights read as a density do.
  qd <- .nl_summary_quantile(v, w, c(0.025, 0.5, 0.975), NA_character_, "density")
  e <- .nl_cell_edges(sort(v))
  expect_true(all(is.finite(qd)))
  expect_lt(qd[1], min(v)); expect_gt(qd[1], e[1])
  expect_gt(qd[3], max(v)); expect_lt(qd[3], e[2])
})

test_that("the unit domain keeps a mixing weight inside (0, 1)", {
  v <- c(0.55, 0.7, 0.85)
  w <- c(0.2, 0.6, 0.2)
  q <- .nl_summary_quantile(v, w, c(0.025, 0.5, 0.975), "unit", "moment_rule")
  expect_true(all(q > 0 & q < 1))
  u <- qlogis(v)
  m <- sum(w * u); s <- sqrt(sum(w * u^2) - m^2)
  expect_equal(q, plogis(m + qnorm(c(0.025, 0.5, 0.975)) * s))
  # A node on the boundary is outside the map's support, so the summary is
  # withheld rather than reaching qlogis() as an infinity.
  expect_true(all(is.na(
    .nl_summary_quantile(c(0.5, 1.0), c(0.5, 0.5), 0.5, "unit", "moment_rule"))))
})

# ------------------------------------------------------------ axis registry

test_that(".joint_axis_domains names each joint axis's coordinate", {
  mk <- function(types, axes) {
    cn <- unlist(lapply(seq_along(types), function(b)
      paste0("b", b, ".", axes[[b]])))
    list(theta_grid = matrix(0, 1L, length(cn), dimnames = list(NULL, cn)),
         axis_offsets = as.integer(c(0, cumsum(lengths(axes)))),
         blocks = lapply(types, function(t) list(type = t)))
  }
  expect_identical(
    .joint_axis_domains(mk(c("iid", "iid"), list("sigma", "sigma"))),
    c("positive", "positive"))
  expect_identical(
    .joint_axis_domains(mk("bym2", list(c("tau", "rho")))),
    c("positive", "unit"))
  expect_identical(
    .joint_axis_domains(mk("mcar", list(c("L11", "L21", "L22")))),
    rep("unbounded", 3L))
  # car_proper's rho_car lives on the adjacency's eigenvalue interval, which the
  # registry will not guess; that axis reports no domain.
  expect_identical(
    .joint_axis_domains(mk("car_proper", list(c("tau", "rho_car")))),
    c("positive", NA_character_))
})

test_that(".joint_pareto_axis_tags still declines a fit carrying an unguessable axis", {
  res <- list(
    theta_grid = matrix(0, 1L, 2L,
                        dimnames = list(NULL, c("b1.tau", "b1.rho_car"))),
    axis_offsets = c(0L, 2L),
    blocks = list(list(type = "car_proper")))
  tags <- .joint_pareto_axis_tags(res)
  expect_true(.k_is_decline(tags))
})

# ------------------------------------------- the joint multi-block front door

test_that("a CCD-integrated joint fit reports an interval off the design's extent", {
  skip_on_cran()
  set.seed(4242)
  G <- 30L; N <- 600L
  grp <- lapply(1:3, function(k) sample.int(G, N, replace = TRUE))
  X <- cbind(1, rnorm(N))
  eta <- as.numeric(X %*% c(0.2, 0.6))
  for (k in 1:3) eta <- eta + rnorm(G, 0, c(0.8, 0.5, 0.3)[k])[grp[[k]]]
  y <- eta + rnorm(N, 0, 0.5)
  sg <- exp(seq(log(0.1), log(2), length.out = 7))
  fit <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = y, n_trials = rep(1L, N), X = X,
                              family = "gaussian", phi = 0.25)),
    prior = lapply(grp, function(g)
      list(type = "iid", obs_idx = list(g), n_units = G, sigma_grid = sg)),
    control = list(integration = "ccd", n_threads = 1L, diagnose_k = FALSE,
                   max_iter = 100L, tol = 1e-8)))
  expect_identical(fit$integration, "ccd")

  for (j in seq_len(ncol(fit$theta_grid))) {
    ext <- range(fit$theta_grid[, j])
    lo <- fit$theta_ci_lo[[j]]; hi <- fit$theta_ci_hi[[j]]
    expect_true(is.finite(lo) && is.finite(hi))
    # Not the node extent -- the defect was exact equality to it.
    expect_false(isTRUE(all.equal(lo, ext[1], tolerance = 0)))
    expect_false(isTRUE(all.equal(hi, ext[2], tolerance = 0)))
    # A positive-scale axis is summarized on log, so the interval is positive
    # and symmetric about its median there.
    expect_gt(lo, 0)
    med <- fit$theta_median[[j]]
    expect_equal(log(hi) - log(med), log(med) - log(lo), tolerance = 1e-8)
  }
})

test_that("the tensor grid keeps the weighted quantile", {
  skip_on_cran()
  set.seed(99)
  G <- 30L; N <- 400L
  g1 <- sample.int(G, N, replace = TRUE)
  X <- cbind(1, rnorm(N))
  y <- as.numeric(X %*% c(0.2, 0.6)) + rnorm(G, 0, 0.7)[g1] + rnorm(N, 0, 0.5)
  sg <- exp(seq(log(0.1), log(2), length.out = 9))
  fit <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = y, n_trials = rep(1L, N), X = X,
                              family = "gaussian", phi = 0.25)),
    prior = list(list(type = "iid", obs_idx = list(g1), n_units = G,
                      sigma_grid = sg)),
    control = list(integration = "grid", n_threads = 1L, diagnose_k = FALSE)))
  expect_false(identical(fit$integration, "ccd"))
  # The grid's cells discretize the density, so the summary stays inside them.
  ext <- range(fit$theta_grid[, 1])
  expect_gte(fit$theta_ci_lo[[1]], ext[1])
  expect_lte(fit$theta_ci_hi[[1]], ext[2])
})

# --------------------------------------------- the inline spatial front door

test_that("the MCAR field's Sigma summary comes off the CCD's moments", {
  skip_on_cran()
  nx <- ny <- 8L; n_s <- nx * ny
  adj <- matrix(0L, n_s, n_s)
  id <- function(i, j) (j - 1L) * nx + i
  for (i in seq_len(nx)) for (j in seq_len(ny)) {
    if (i < nx) { a <- id(i, j); b <- id(i + 1L, j); adj[a, b] <- adj[b, a] <- 1L }
    if (j < ny) { a <- id(i, j); b <- id(i, j + 1L); adj[a, b] <- adj[b, a] <- 1L }
  }
  Sigma <- matrix(c(1, 0.48, 0.48, 0.64), 2, 2)
  set.seed(31)
  Qp <- diag(rowSums(adj)) - 0.99 * adj
  U <- chol(Qp)
  z1 <- backsolve(U, rnorm(n_s)); z1 <- z1 - mean(z1)
  z2 <- backsolve(U, rnorm(n_s)); z2 <- z2 - mean(z2)
  L <- t(chol(Sigma))
  fu <- L[1, 1] * z1; fs <- L[2, 1] * z1 + L[2, 2] * z2
  cell <- rep(seq_len(n_s), each = 20L); N <- length(cell); x <- rnorm(N)
  d <- data.frame(y = rnorm(N, 0.3 + fu[cell] + x * fs[cell], 0.4),
                  x = x, cell = cell)
  fit <- suppressWarnings(tulpa(
    y ~ spatial(graph = adj, formula = ~ 1 + x | cell), data = d,
    family = "gaussian", mode = "laplace", control = list(n_draws = 100L)))
  expect_identical(fit$integration, "ccd")

  tg <- fit$theta_grid
  Sig_list <- lapply(seq_len(nrow(tg)), function(k) {
    Lk <- .re_logchol_to_L(as.numeric(tg[k, c("b1.L11", "b1.L21", "b1.L22")]), 2L)
    Lk %*% t(Lk)
  })
  D <- .re_cov_derived_matrix(Sig_list, 2L, full = TRUE)
  for (cn in colnames(D)) {
    q <- fit$mcar_summary[[cn]]$q
    ext <- range(D[, cn])
    expect_false(isTRUE(all.equal(q[1], ext[1], tolerance = 0)),
                 label = paste(cn, "lower endpoint is the node minimum"))
    expect_false(isTRUE(all.equal(q[3], ext[2], tolerance = 0)),
                 label = paste(cn, "upper endpoint is the node maximum"))
  }
  # Scales stay positive and the correlation stays inside (-1, 1), which is what
  # naming each derived quantity's domain buys.
  expect_gt(fit$mcar_summary$sigma_1$q[1], 0)
  expect_gt(fit$mcar_summary$sigma_2$q[1], 0)
  expect_true(all(abs(fit$mcar_summary$rho_12$q) < 1))
})

test_that("a three-column bar field's per-block sigma comes off the CCD's moments", {
  skip_on_cran()
  n_s <- 30L
  adj <- matrix(0, n_s, n_s)
  for (i in seq_len(n_s - 1L)) adj[i, i + 1L] <- adj[i + 1L, i] <- 1
  set.seed(17)
  f <- lapply(1:3, function(k) { v <- rnorm(n_s); v - mean(v) })
  cell <- rep(seq_len(n_s), each = 25L); N <- length(cell)
  a <- rnorm(N); b <- rnorm(N)
  d <- data.frame(
    y = rnorm(N, 0.2 + 0.9 * f[[1]][cell] + a * 0.6 * f[[2]][cell] +
                b * 0.4 * f[[3]][cell], 0.4),
    a = a, b = b, cell = cell)
  fit <- suppressWarnings(tulpa(
    y ~ spatial(graph = adj, formula = ~ 1 + a + b || cell), data = d,
    family = "gaussian", mode = "laplace", control = list(n_draws = 100L)))
  # Three independent ICAR blocks are three transformable axes, which is where
  # the CCD engages.
  expect_identical(fit$integration, "ccd")
  expect_length(fit$spatial_field_hypers, 3L)

  for (k in seq_along(fit$spatial_field_hypers)) {
    h  <- fit$spatial_field_hypers[[k]]
    sg <- 1 / sqrt(fit$theta_grid[, paste0("b", k, ".tau")])
    ext <- range(sg)
    expect_false(isTRUE(all.equal(h$sigma[1], ext[1], tolerance = 0)))
    expect_false(isTRUE(all.equal(h$sigma[3], ext[2], tolerance = 0)))
    expect_gt(h$sigma[1], 0)
    expect_equal(log(h$sigma[3]) - log(h$sigma[2]),
                 log(h$sigma[2]) - log(h$sigma[1]), tolerance = 1e-8)
  }
})
