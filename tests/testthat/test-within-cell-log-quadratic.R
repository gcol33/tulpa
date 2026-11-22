# The log-quadratic within-cell read (gcol33/tulpa#932): node densities joined
# by the overlapping quadratics through adjacent log densities, continued past
# the outer node by the end quadratic, per row of the grid, and gated on the
# axis's resolution.

lq_gauss_nodes <- function(mu, s, r, span = 2.5, offset = 0) {
  k <- seq(-ceiling(span / r) - 1L, ceiling(span / r) + 1L)
  u <- mu + (offset + k) * r * s
  u[abs(u - mu) <= span * s + 1e-12]
}

test_that("an unresolved axis declines to the box read and says why", {
  mu <- log(2); s <- 0.15
  u <- lq_gauss_nodes(mu, s, 3, span = 6, offset = 0.3)
  w <- stats::dnorm(u, mu, s); w <- w / sum(w)
  probs <- c(0.025, 0.5, 0.975)
  rd <- .nl_summary_quantile_read(exp(u), w, probs, "positive", "density",
                                  "log_quadratic", row_id = rep(1L, length(u)),
                                  log_density = log(w))
  expect_identical(rd$within, "box_uniform")
  expect_identical(rd$declined, "unresolved")
  expect_equal(rd$q, .nl_summary_quantile(exp(u), w, probs, "positive",
                                          "density", "box_uniform"))
  rc <- .nl_lq_reconstruct(exp(u), w, "positive", NULL, rep(1L, length(u)),
                           log(w))
  expect_gt(rc$h_over_sd, .nl_diag("lq_max_h_over_sd"))
})

skip_on_cran_budget()

test_that("a Gaussian marginal is read exactly at a spacing the box read misses", {
  skip_on_cran_budget()
  mu <- log(2); s <- 0.15
  probs <- c(0.025, 0.5, 0.975)
  truth <- exp(stats::qnorm(probs, mu, s))
  for (off in c(0, 0.37)) {
    u <- lq_gauss_nodes(mu, s, 1.25, offset = off)
    # Equal spacing on the axis's own coordinate, so a cell's mass is its node
    # density times one common width.
    w <- stats::dnorm(u, mu, s); w <- w / sum(w)
    rd <- .nl_summary_quantile_read(exp(u), w, probs, "positive", "density",
                                    "log_quadratic", row_id = rep(1L, length(u)),
                                    log_density = log(w))
    expect_identical(rd$within, "log_quadratic")
    expect_true(is.na(rd$declined))
    expect_equal(rd$q, truth, tolerance = 1e-4)
    bx <- .nl_summary_quantile(exp(u), w, probs, "positive", "density",
                               "box_uniform")
    # The box read truncates at the outer box edge and errs by O(h^2) inside it.
    expect_gt(abs(diff(log(bx[c(1L, 3L)])) / diff(log(truth[c(1L, 3L)])) - 1),
              0.02)
  }
})

test_that("a quantity with no rows or no node density declines to the box read", {
  skip_on_cran_budget()
  u <- lq_gauss_nodes(0, 1, 1)
  w <- stats::dnorm(u); w <- w / sum(w)
  rd <- .nl_summary_quantile_read(u, w, c(0.025, 0.975), "unbounded", "density",
                                  "log_quadratic", log_density = log(w))
  expect_identical(rd$within, "box_uniform")
  expect_identical(rd$declined, "not_a_grid_axis")
  # Nor one with no node density: a summary taken off weights alone.
  rd <- .nl_summary_quantile_read(u, w, c(0.025, 0.975), "unbounded", "density",
                                  "log_quadratic", row_id = rep(1L, length(u)))
  expect_identical(rd$within, "box_uniform")
  expect_identical(rd$declined, "no_node_density")
})

test_that("each row is read on its own conditional, mass kept per row", {
  skip_on_cran_budget()
  # A correlated Gaussian on a tensor grid: every row's conditional along the
  # first axis is exactly Gaussian, so the read is the mixture of those
  # conditionals at the rows' masses, which is computable in closed form.
  rho <- 0.6; s1 <- 0.2; s2 <- 1
  a <- log(1.5) + seq(-2.5, 2.5, by = 1.25) * s1
  b <- seq(-2.5, 2.5, by = 0.625) * s2
  tg <- as.matrix(expand.grid(sigma = exp(a), x = b))
  z1 <- (log(tg[, "sigma"]) - log(1.5)) / s1; z2 <- tg[, "x"] / s2
  lm <- -0.5 * (z1^2 - 2 * rho * z1 * z2 + z2^2) / (1 - rho^2)
  qs <- .nl_axis_quantiles(tg, lm, domains = list("positive", "unbounded"),
                           within = "log_quadratic")
  expect_identical(unname(qs$within[["sigma"]]), "log_quadratic")

  rid <- .nl_axis_row_id(tg, 1L)
  wr <- tapply(exp(lm - max(lm)), rid, sum); wr <- wr / sum(wr)
  bv <- tapply(tg[, "x"], rid, `[`, 1L)
  cm <- log(1.5) + rho * s1 * bv / s2
  cs <- s1 * sqrt(1 - rho^2)
  cdf <- function(q) sum(wr * stats::pnorm(log(q), cm, cs))
  for (p in c(0.025, 0.5, 0.975)) {
    q <- switch(as.character(p), "0.025" = qs$ci_lo[["sigma"]],
                "0.5" = qs$median[["sigma"]], "0.975" = qs$ci_hi[["sigma"]])
    expect_equal(cdf(q), p, tolerance = 2e-4)
  }
})

test_that("a row's density is its log marginal times each cell's slab", {
  skip_on_cran_budget()
  # On a refined grid the cell the slices pass through has its other-axis box
  # cut by them: it holds 1/30 of its row-mates' slab, and the slice row that
  # crosses it holds the rest of its box. The read has to interpolate the row's
  # log marginal (no dip at the hub) and still put only the hub's own mass in
  # the hub's box (the slice row supplies the remainder).
  mu <- log(2); s <- 0.15
  u <- lq_gauss_nodes(mu, s, 1)
  ld <- stats::dnorm(u, mu, s, log = TRUE)
  hub <- which.min(abs(u - mu))
  w <- exp(ld)
  w_hub <- w[hub]
  w[hub] <- w_hub / 30
  v <- c(exp(u), exp(u[hub]))
  ww <- c(w, w_hub * 29 / 30); ww <- ww / sum(ww)
  lds <- c(ld, ld[hub])
  rid <- c(rep(1L, length(u)), 2L)
  probs <- c(0.025, 0.5, 0.975)
  truth <- exp(stats::qnorm(probs, mu, s))
  rd <- .nl_summary_quantile_read(v, ww, probs, "positive", "density",
                                  "log_quadratic", row_id = rid,
                                  log_density = lds)
  expect_identical(rd$within, "log_quadratic")
  bx <- .nl_summary_quantile(v, ww, probs, "positive", "density", "box_uniform")
  err <- function(q) abs(log(q[3L] / q[1L]) / log(truth[3L] / truth[1L]) - 1)
  expect_lt(err(rd$q), 0.01)
  expect_lt(err(rd$q), err(bx))
})

test_that("a declared point mass is split off before the continuum is read", {
  skip_on_cran_budget()
  mu <- log(0.5); s <- 0.3
  u <- lq_gauss_nodes(mu, s, 1)
  wc <- stats::dnorm(u, mu, s); wc <- 0.7 * wc / sum(wc)
  v <- c(0, exp(u)); w <- c(0.3, wc)
  probs <- c(0.1, 0.3, 0.5, 0.975)
  rd <- .nl_summary_quantile_read(v, w, probs, "positive", "density",
                                  "log_quadratic", atom = 0,
                                  row_id = rep(1L, length(v)),
                                  log_density = log(w))
  expect_identical(rd$within, "log_quadratic")
  expect_identical(rd$q[1:2], c(0, 0))
  expect_equal(rd$q[3:4],
               exp(stats::qnorm((probs[3:4] - 0.3) / 0.7, mu, s)),
               tolerance = 1e-4)
})

test_that("a row's tail stops at a level laid beyond it that holds no mass", {
  skip_on_cran_budget()
  # One row of a Gaussian on the unbounded coordinate, nodes 1.25 SDs apart, and
  # a level one node step below the lowest massive one whose cell carries no
  # mass (pruned by the screen: weight 0, log marginal -Inf). The tail
  # continued past the lowest massive node would put mass on that level's box,
  # which the grid measured to hold none.
  s <- 1; h <- 1.25 * s
  u_mass <- seq(-2, 2) * h
  u <- c(-3 * h, u_mass)
  ld <- c(-Inf, stats::dnorm(u_mass, 0, s, log = TRUE))
  w <- c(0, exp(ld[-1L])); w <- w / sum(w)
  probs <- c(5e-4, 0.025, 0.5, 0.975)
  rd <- .nl_summary_quantile_read(u, w, probs, "unbounded", "density",
                                  "log_quadratic", row_id = rep(1L, length(u)),
                                  log_density = ld)
  expect_identical(rd$within, "log_quadratic")
  # The lowest massive node's box ends half way to the laid level below it.
  edge <- (u[1L] + u_mass[1L]) / 2
  expect_gte(min(rd$q), edge)
  # Without the laid level the same row's tail runs past that edge.
  free <- .nl_summary_quantile_read(u_mass, w[-1L], probs, "unbounded",
                                    "density", "log_quadratic",
                                    row_id = rep(1L, length(u_mass)),
                                    log_density = ld[-1L])
  expect_lt(free$q[1L], edge)
})

test_that("a correlation measured on its own scale is read by its density on atanh", {
  skip_on_cran_budget()
  # An AR(1)-like rho grid running to tanh(+/- 4.5), integrated on the natural
  # scale, with a likelihood that barely moves over it: the posterior is the
  # uniform prior on (-1, 1). Per unit of rho that density is flat, so a read
  # interpolating it on atanh took the rows' sd far past their nodes
  # (gcol33/tulpa#1274); per unit of atanh it is 1 - rho^2, sd near 0.9.
  rho <- tanh(seq(-4.5, 4.5, length.out = 22))
  sig <- exp(c(-0.5, 0, 0.5))
  tg <- as.matrix(expand.grid(sigma = sig, rho = rho))
  specs <- list(hyper_axis_spec("sigma", grid = sig, log_scale = TRUE),
                hyper_axis_spec("rho", grid = rho, bounds = c(-1, 1)))
  lm <- -0.5 * log(tg[, "sigma"])^2 - 1e-10 * tg[, "rho"]^2
  lq <- .hyper_log_quad_weights(tg, specs)
  qs <- .nl_axis_quantiles(tg, lm, log_quad = lq,
                           domains = list("positive", "correlation"),
                           within = "log_quadratic", specs = specs)
  expect_identical(unname(qs$within[["rho"]]), "log_quadratic")
  got <- c(qs$ci_lo[["rho"]], qs$median[["rho"]], qs$ci_hi[["rho"]])
  # Within the grid measure's own quadrature error at this spacing (its box
  # read gives +/- 0.954).
  expect_lt(max(abs(got - c(-0.95, 0, 0.95))), 0.01)
  # The rows' resolution is read off the density on atanh: spacing 0.43 over
  # an sd near 0.9, not over one of thousands.
  ld <- .nl_read_log_density(tg, lm, specs, list("positive", "correlation"))
  rc <- .nl_lq_reconstruct(tg[, "rho"], exp(lm + lq) / sum(exp(lm + lq)),
                           "correlation", NULL, .nl_axis_row_id(tg, 2L), ld)
  expect_gt(rc$h_over_sd, 0.3)
  expect_lt(rc$h_over_sd, 1)
})

test_that("a row flat on its own coordinate ends at its outer boxes", {
  skip_on_cran_budget()
  # Two rows over the same 22 levels: one Gaussian with sd 0.5 holding 0.99 of
  # the mass, one whose log density falls by 2e-8 nats over its nodes. The
  # flat row's nodes never saw its density fall, so its curvature says nothing
  # about a tail: before gcol33/tulpa#1278 it reached +/- 1.9e5 and put almost
  # all of that row's mass outside the grid. It now ends at its outer box
  # edges, half a spacing past the outer levels, and holds its mass as the
  # uniform its nodes describe.
  u <- seq(-4.5, 4.5, length.out = 22)
  ldA <- -0.5 * (u / 0.5)^2
  ldB <- -1e-9 * u^2
  v <- c(u, u); row <- rep(1:2, each = 22)
  w <- c(exp(ldA) / sum(exp(ldA)) * 0.99, exp(ldB) / sum(exp(ldB)) * 0.01)
  rc <- .nl_lq_reconstruct(v, w, "unbounded", NULL, row, c(ldA, ldB))
  edge <- 4.5 + 0.5 * diff(u)[1L]
  expect_equal(c(rc$pieces[[2L]]$lo, rc$pieces[[2L]]$hi), c(-edge, edge))
  expect_equal(range(rc$x), c(-edge, edge))
  q <- .nl_lq_cdf_quantile(rc, c(0.025, 0.5, 0.975))
  F <- function(x) 0.99 * stats::pnorm(x / 0.5) + 0.01 * (x + edge) / (2 * edge)
  hi <- stats::uniroot(function(x) F(x) - 0.975, c(0, edge), tol = 1e-12)$root
  expect_equal(q, c(-hi, 0, hi), tolerance = 2e-3)
})

test_that("a side whose nodes saw the density fall keeps its tail", {
  skip_on_cran_budget()
  # One Gaussian row laid from its mode to 1.5 SDs above and 0.3 SDs below:
  # the upper end node stands 1.125 nats under the top, past the
  # qnorm(3/4)^2 / 2 = 0.227 at which an extrapolated tail would carry more
  # mass than the stretch the nodes measured, so that side continues; the
  # lower end stands 0.045 nats under it and ends at its box edge.
  u <- seq(-0.3, 1.5, by = 0.3)
  ld <- -0.5 * u^2
  rc <- .nl_lq_reconstruct(u, exp(ld) / sum(exp(ld)), "unbounded", NULL,
                           rep(1L, length(u)), ld)
  p <- rc$pieces[[1L]]
  expect_identical(p$kind, "lq")
  expect_equal(p$lo, -0.45)
  expect_gt(p$hi, 1.65)
})
