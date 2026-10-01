# The log-quadratic within-cell read (gcol33/tulpa#932): node densities joined
# by the overlapping quadratics through adjacent log densities, continued past
# the outer node by the end quadratic, per row of the grid, and gated on the
# axis's resolution.

lq_gauss_nodes <- function(mu, s, r, span = 2.5, offset = 0) {
  k <- seq(-ceiling(span / r) - 1L, ceiling(span / r) + 1L)
  u <- mu + (offset + k) * r * s
  u[abs(u - mu) <= span * s + 1e-12]
}

test_that("a Gaussian marginal is read exactly at a spacing the box read misses", {
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

test_that("a quantity with no rows or no node density declines to the box read", {
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
