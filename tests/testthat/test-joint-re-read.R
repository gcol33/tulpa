# The random effects of a joint nested-Laplace fit are read as one Gaussian per
# outer cell -- the coordinate times its scale, with the cell's precision read
# along that loading -- and mixed over the outer weights: on the one-arm route
# tulpa() takes, on an inline spatial() / temporal() field fit, and on a fit
# made through tulpa_nested_laplace_joint() directly.

.jrr_lattice <- function(side = 4L) {
  K <- side^2
  W <- matrix(0, K, K)
  for (i in 1:K) for (j in 1:K) {
    ri <- (i - 1) %/% side; ci <- (i - 1) %% side
    rj <- (j - 1) %/% side; cj <- (j - 1) %% side
    if (abs(ri - rj) + abs(ci - cj) == 1) W[i, j] <- 1
  }
  W
}

test_that("an inline field fit reports its random intercepts", {
  skip_on_cran()
  set.seed(11)
  W <- .jrr_lattice()
  d <- data.frame(cell = sample(1:16, 400, TRUE), x = stats::rnorm(400),
                  g = factor(sample(1:12, 400, TRUE)))
  u <- stats::rnorm(16) * 0.5
  b <- stats::rnorm(12) * 0.8
  d$y <- stats::rpois(400, exp(0.2 + 0.4 * d$x + u[d$cell] + b[d$g]))
  fit <- suppressWarnings(tulpa(
    y ~ x + spatial(graph = W, formula = ~ 1 | cell) + (1 | g), data = d,
    family = "poisson"))
  r <- ranef(fit)
  expect_identical(nrow(r), 12L)
  expect_true(all(r$source == "mixture") && all(is.finite(r$sd)))
  expect_gt(stats::cor(r$estimate, b), 0.9)
  # The table is the latent posterior the predictions are drawn from.
  src <- tulpa:::.field_source(fit)
  L <- tulpa:::.field_latent_draws(fit, src, 4000L)
  # The (1 | g) term is an `iid` block, non-centred: each effect is its
  # coordinate times the SD of the cell the row was drawn from.
  re <- src$layout$re
  S <- t(vapply(attr(L, "cells"), re$scale, numeric(length(re$cols))))
  B <- L[, re$cols, drop = FALSE] * S
  expect_lt(max(abs(apply(B, 2, stats::sd) / r$sd - 1)), 0.08)
})

test_that("a direct joint fit reports its iid, miid and arm random effects", {
  skip_on_cran()
  set.seed(21)
  G <- 25L
  N <- 1000L
  grp <- sample(seq_len(G), N, TRUE)
  x <- stats::rnorm(N)
  L <- matrix(c(0.7, 0, 0.2, sqrt(0.75) * 0.4), 2, byrow = TRUE)
  B <- matrix(stats::rnorm(2 * G), G) %*% t(L)
  arm <- list(y = stats::rnorm(N, 0.3 + 0.5 * x + B[grp, 1] + B[grp, 2] * x,
                               0.5),
              X = cbind(1, x), family = "gaussian", n_trials = rep(1L, N),
              phi = 0.25)
  fm <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(arm = arm),
    prior = list(list(type = "miid", n_groups = G, n_fields = 2L,
                      obs_idx = list(grp), name = "g",
                      field_weight = list(list(rep(1, N)), list(x)))),
    control = list(diagnose_k = FALSE, store_Q = TRUE)))
  r <- ranef(fm)
  expect_identical(r$term[c(1, G + 1)], c("g.f1[1]", "g.f2[1]"))
  # The coefficients are read field-major, intercepts then slopes.
  expect_gt(stats::cor(r$estimate[seq_len(G)], B[, 1]), 0.95)
  expect_gt(stats::cor(r$estimate[G + seq_len(G)], B[, 2]), 0.9)
  # The table is the fit's latent posterior: its mixture draws give the same
  # moments (measured 0.98-1.02 on the SD ratio at 4000 draws).
  dr <- tulpa_posterior_draws(fm, idx = NULL, n = 4000L)
  st <- fm$arm_layout$block_start[1L]
  D <- dr[, st + seq_len(2L * G)]
  expect_lt(max(abs(apply(D, 2, stats::sd) / r$sd - 1)), 0.08)
  expect_lt(max(abs(colMeans(D) - r$estimate) / r$sd), 0.1)

  # An iid block is non-centred: the effect is the coordinate times the
  # cell's sigma.
  b <- B[, 1]
  ap <- list(y = stats::rpois(N, exp(0.3 + 0.5 * x + b[grp])), X = cbind(1, x),
             family = "poisson", n_trials = rep(1L, N))
  fi <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(arm = ap),
    prior = list(list(type = "iid", n_units = G, obs_idx = list(grp))),
    control = list(diagnose_k = FALSE, store_Q = TRUE)))
  ri <- ranef(fi)
  expect_identical(ri$term[1], "b1[1]")
  expect_gt(stats::cor(ri$estimate, b), 0.95)
  expect_lt(abs(stats::sd(ri$estimate) / stats::sd(b) - 1), 0.2)
  # Against draws of each cell's Gaussian (the latent draw kernel, cells taken
  # by weight), scaled by the cell's sigma.
  set.seed(1)
  w <- fi$weights / sum(fi$weights)
  cells <- sample.int(length(w), 4000L, replace = TRUE, prob = w)
  n_x <- ncol(fi$modes)
  st <- fi$arm_layout$block_start[1L]
  E <- do.call(rbind, lapply(unique(cells), function(k) {
    n_k <- sum(cells == k)
    Z <- tulpa:::.joint_draw_cell(fi$Q_csc_p_per_grid[[k]],
                                  fi$Q_csc_i_per_grid[[k]],
                                  fi$Q_csc_x_per_grid[[k]], n_x,
                                  tulpa:::.joint_constraint_cols(
                                    fi$arm_layout, n_x), n_k)
    t(Z[st + seq_len(G), , drop = FALSE] + fi$modes[k, st + seq_len(G)]) *
      fi$theta_grid[k, "b1.sigma"]
  }))
  expect_lt(max(abs(apply(E, 2, stats::sd) / ri$sd - 1)), 0.08)
  expect_lt(max(abs(colMeans(E) - ri$estimate) / ri$sd), 0.1)
  # Without the per-cell precision the mixture mean is reported, its spread
  # is not.
  f0 <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(arm = ap),
    prior = list(list(type = "iid", n_units = G, obs_idx = list(grp))),
    control = list(diagnose_k = FALSE)))
  r0 <- ranef(f0)
  expect_equal(r0$estimate, ri$estimate, tolerance = 1e-8)
  expect_true(all(is.na(r0$sd)) && all(r0$source == "mode"))

  # A block the prediction has no new-row index for is refused by name.
  expect_error(predict(fi, newdata = list(X = cbind(1, x[1:5]))),
               "block 1 \\(iid\\)")
})
