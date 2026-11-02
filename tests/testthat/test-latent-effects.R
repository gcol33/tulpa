# latent_effects() reads a latent(tgmrf()) block's per-coordinate posterior as
# the grid mixture of each cell's loading, the construction field_effects()
# uses. The mixture means are linear in the cell modes, so the fixed part, the
# field and the block add up to the in-sample linear predictor's mean exactly.

test_that("latent_effects() returns the block's mixture and adds up to eta", {
  skip_if_fast()
  set.seed(3)
  K <- 12L
  W <- matrix(0, K, K)
  for (i in seq_len(K)) { j <- if (i < K) i + 1L else 1L; W[i, j] <- W[j, i] <- 1 }
  d <- data.frame(region = rep(seq_len(K), each = 10L))
  d$x <- stats::rnorm(nrow(d))
  d$y <- stats::rpois(nrow(d), exp(0.5 + 0.4 * d$x + sin(2 * pi * d$region / K) +
                                     stats::rnorm(nrow(d), 0, 0.3)))
  blk <- tgmrf(Q = function(theta) Matrix::Diagonal(nrow(d), exp(theta[1])),
               prior = function(theta) stats::dnorm(theta[1], 2, 1.5, log = TRUE),
               init = c(log_tau = 2), bounds = list(lower = -1, upper = 7),
               obs_idx = seq_len(nrow(d)), name = "obs_iid")
  fit <- suppressWarnings(tulpa(y ~ x + spatial(region) + latent(blk), data = d,
    family = "poisson", spatial = list(type = "icar", adjacency = W),
    mode = "auto", control = list(progress = FALSE)))
  le <- latent_effects(fit, "obs_iid")
  expect_identical(nrow(le), nrow(d))
  expect_true(all(le$source == "mixture"))
  expect_true(all(is.finite(le$sd) & le$sd > 0))
  expect_identical(latent_effects(fit, 1L), le)
  expect_error(latent_effects(fit, "nope"), "block names")

  fe <- field_effects(fit)
  w <- fit$weights; w <- w / sum(w)
  b <- as.numeric(crossprod(w, fit$modes[, 1:2]))
  pl <- posterior_linpred(fit, summary = TRUE)
  expect_equal(pl$estimate,
               b[1] + b[2] * d$x + fe$estimate[d$region] + le$estimate,
               tolerance = 1e-6)
})
