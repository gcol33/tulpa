# Recovery tests for the Polya-Gamma sampler (src/pg_rng.cpp).
# A PG sampler is a statistical object: shape checks are insufficient. These
# tests compare sampled moments against the exact closed forms for PG(1, z) and
# PG(b, z) (Polson, Scott & Windle 2013; moment formulas as in BayesLogit /
# pgR::pgdraw.moments).

# Exact PG(1, z) mean: tanh(z/2) / (2 z), with the z -> 0 limit 1/4.
pg1_mean_exact <- function(z) {
  if (abs(z) < 1e-8) return(0.25)
  tanh(z / 2) / (2 * z)
}

# Exact PG(1, z) variance: (sinh(z) - z) / (4 z^3 cosh(z/2)^2),
# with the z -> 0 limit 1/24.
pg1_var_exact <- function(z) {
  if (abs(z) < 1e-6) return(1 / 24)
  (sinh(z) - z) / (4 * z^3 * cosh(z / 2)^2)
}

test_that("cpp_rpg1 recovers the closed-form PG(1, z) mean and variance", {
  skip_on_cran()
  set.seed(20260530)

  N <- 200000
  zs <- c(0, 0.3, 1, 3, 8)

  for (z in zs) {
    x <- tulpa:::cpp_rpg1(rep(z, N))

    expect_true(all(x > 0))

    m_hat <- mean(x)
    v_hat <- var(x)
    m_exact <- pg1_mean_exact(z)
    v_exact <- pg1_var_exact(z)

    # Monte-Carlo SE of the mean; 4 SE tolerance.
    se_mean <- sqrt(v_exact / N)
    expect_lt(abs(m_hat - m_exact), 4 * se_mean,
              label = sprintf("z=%.2f mean (hat=%.6f exact=%.6f)", z, m_hat, m_exact))

    # SE of the sample variance ~ sqrt((mu4 - v^2) / N); use a 5% relative
    # tolerance, comfortably above MC noise at this N for these z.
    expect_lt(abs(v_hat - v_exact) / v_exact, 0.05,
              label = sprintf("z=%.2f var (hat=%.6f exact=%.6f)", z, v_hat, v_exact))
  }
})

test_that("cpp_rpg recovers PG(b, z) mean for exact and approx paths", {
  skip_on_cran()
  set.seed(7)

  N <- 100000

  # Small b: exact summation path.
  for (cfg in list(c(b = 1, z = 0.5), c(b = 5, z = 2))) {
    b <- as.integer(cfg["b"]); z <- cfg["z"]
    x <- tulpa:::cpp_rpg(rep(b, N), rep(z, N))
    m_exact <- b * pg1_mean_exact(z)
    v_exact <- b * pg1_var_exact(z)
    se_mean <- sqrt(v_exact / N)
    expect_lt(abs(mean(x) - m_exact), 4 * se_mean,
              label = sprintf("b=%d z=%.2f mean", b, z))
  }

  # Large b: Gaussian moment-match path (b >= 200 threshold).
  for (cfg in list(c(b = 500, z = 1), c(b = 1000, z = 3))) {
    b <- as.integer(cfg["b"]); z <- cfg["z"]
    x <- tulpa:::cpp_rpg(rep(b, N), rep(z, N))
    m_exact <- b * pg1_mean_exact(z)
    v_exact <- b * pg1_var_exact(z)
    se_mean <- sqrt(v_exact / N)
    expect_lt(abs(mean(x) - m_exact), 4 * se_mean,
              label = sprintf("b=%d z=%.2f mean (approx path)", b, z))
    # The CLT match should also recover the variance to a few percent.
    expect_lt(abs(var(x) - v_exact) / v_exact, 0.05,
              label = sprintf("b=%d z=%.2f var (approx path)", b, z))
  }
})
