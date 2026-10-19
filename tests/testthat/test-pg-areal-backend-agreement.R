# The Polya-Gamma Gibbs areal kernels sample the model the ModelData sampler
# samples. On a disconnected graph the ICAR prior is tau Q_aug, so an island
# carries the field's own precision (gcol33/tulpa#963), and a BYM2 field is
# reported on the scale it enters eta (gcol33/tulpa#964).

# A chain of 8 regions and an island, region 9, with 4 / 4 successes: a
# separated island, which a near-flat island prior lets run off.
skip_on_cran()

.island_chain_data <- function() {
  J <- 9
  A <- matrix(0, J, J)
  for (i in 1:7) A[i, i + 1] <- A[i + 1, i] <- 1
  set.seed(2)
  d <- data.frame(region = rep(1:J, c(rep(40, 8), 4)))
  d$yb <- rbinom(nrow(d), 1, plogis(c(sin(1:8) * 0.8, 0)[d$region]))
  d$yb[d$region == 9] <- 1L
  list(A = A, d = d)
}

test_that("PG Gibbs and HMC agree on an ICAR field with an island (#963)", {
  skip_on_cran()
  dd <- .island_chain_data()
  sp <- suppressMessages(spatial_car(dd$A, level = "group",
                                     group_var = "region"))
  fit <- function(m) suppressWarnings(tulpa(
    yb ~ 1 + spatial(region), spatial = sp, data = dd$d, family = "binomial",
    mode = m, control = list(seed = 1, n_iter = 4000, warmup = 2000)))
  fg <- fit("gibbs")
  fh <- fit("hmc")
  eg <- field_effects(fg)
  eh <- field_effects(fh)
  # The island's posterior SD is about 0.45; under the near-flat prior the
  # Gibbs island read 14.5 and dragged the mainland and intercept with it.
  expect_lt(abs(eg$estimate[9] - eh$estimate[9]), 0.25)
  expect_lt(max(abs(eg$estimate[1:8] - eh$estimate[1:8])), 0.15)
  expect_lt(abs(coef(fg)[1] - coef(fh)[1]), 0.15)
  expect_lt(eg$sd[9], 1)
})

test_that("PG Gibbs reports a BYM2 field on the scale it enters eta (#964)", {
  skip_on_cran()
  J <- 8
  A <- matrix(0, J, J)
  for (i in 1:7) A[i, i + 1] <- A[i + 1, i] <- 1
  set.seed(2)
  d <- data.frame(region = rep(1:J, 40))
  d$yb <- rbinom(nrow(d), 1, plogis(sin(1:8)[d$region] * 0.8))
  sb <- spatial_bym2(A, level = "group", group_var = "region")
  fit <- function(m) suppressWarnings(tulpa(
    yb ~ 1 + spatial(region), spatial = sb, data = d, family = "binomial",
    mode = m, control = list(seed = 1, n_iter = 4000, warmup = 2000)))
  fg <- fit("gibbs")
  fh <- fit("hmc")
  eg <- field_effects(fg)
  eh <- field_effects(fh)
  # The unit-scale phi + theta read about twice the combined field here.
  expect_lt(max(abs(eg$estimate - eh$estimate)), 0.15)

  # The field is the per-draw sigma (sqrt(rho) s phi + sqrt(1 - rho) theta).
  D <- fg$draws
  sig <- exp(D[, "log_sigma_spatial"])
  rho <- plogis(D[, "logit_rho_bym2"])
  u1 <- sig * (sqrt(rho) * sb$scale_factor * D[, "phi_spatial[1]"] +
               sqrt(1 - rho) * D[, "theta_spatial[1]"])
  expect_equal(eg$estimate[1], mean(u1), tolerance = 1e-10)
})
