# tulpa_simulate: generic simulator dispatching through family$simulate_fn

make_gaussian_family <- function() {
  tulpa_family(
    name = "gaussian",
    simulate_fn = function(eta, params, n_obs, ...) {
      rnorm(n_obs, eta[[1]], params$sigma_y)
    },
    extra_params = list(sigma_y = prior_half_normal(1))
  )
}

set.seed(7)
df <- data.frame(
  y = rep(0, 25),
  x = rnorm(25),
  g = factor(rep(1:5, each = 5))
)


test_that("tulpa_simulate with NULL theta is a single prior draw", {
  fam <- make_gaussian_family()
  s <- tulpa_simulate(y ~ x, fam, df, theta = NULL, n_sims = 1, seed = 1)

  expect_s3_class(s, "tulpa_simulate")
  expect_length(s$y, 1)
  expect_length(s$y[[1]], 25)
  expect_length(s$theta, 1)
  expect_length(s$theta[[1]]$beta[[1]], 2L)
})


test_that("tulpa_simulate accepts a fixed theta list", {
  fam <- make_gaussian_family()
  theta <- list(
    beta = list(y = c(0.0, 1.0)),
    u = list(y = list()),
    extras = list(sigma_y = 0.01)
  )
  s <- tulpa_simulate(y ~ x, fam, df, theta = theta, n_sims = 4, seed = 11)

  expect_length(s$y, 4)
  # With sigma_y near zero, simulated y should be ~ X %*% beta
  X <- model.matrix(~ x, df)
  expected <- as.numeric(X %*% theta$beta$y)
  for (k in seq_along(s$y)) {
    expect_lt(max(abs(s$y[[k]] - expected)), 0.1)
  }
})


test_that("tulpa_simulate validates theta against parsed model", {
  fam <- make_gaussian_family()
  bad <- list(beta = list(y = c(1.0)),  # too short
              u = list(y = list()),
              extras = list(sigma_y = 1))
  expect_error(tulpa_simulate(y ~ x, fam, df, theta = bad), "expected 2")

  miss_extras <- list(beta = list(y = c(0, 1)), u = list(y = list()),
                      extras = list())
  expect_error(tulpa_simulate(y ~ x, fam, df, theta = miss_extras),
               "sigma_y")
})


test_that("tulpa_simulate is reproducible with seed", {
  fam <- make_gaussian_family()
  s1 <- tulpa_simulate(y ~ x, fam, df, n_sims = 3, seed = 99)
  s2 <- tulpa_simulate(y ~ x, fam, df, n_sims = 3, seed = 99)
  expect_identical(s1$y, s2$y)
})


test_that("tulpa_simulate works with random effects", {
  fam <- make_gaussian_family()
  s <- tulpa_simulate(y ~ x + (1 | g), fam, df, n_sims = 2, seed = 3)
  expect_length(s$y[[1]], 25)
  expect_equal(nrow(s$theta[[1]]$u[[1]][[1]]), 5)
})


test_that("tulpa_simulate dispatches via family$simulate_fn (no model branch)", {
  # Custom family: response is a constant tag
  custom <- tulpa_family(
    name = "tag",
    simulate_fn = function(eta, params, n_obs, ...) rep("tag", n_obs)
  )
  s <- tulpa_simulate(y ~ x, custom, df, theta = list(
    beta = list(y = c(0, 0)), u = list(y = list()), extras = list()
  ), n_sims = 2)
  expect_true(all(s$y[[1]] == "tag"))
})


test_that("print method runs", {
  fam <- make_gaussian_family()
  s <- tulpa_simulate(y ~ x, fam, df, n_sims = 1, seed = 0)
  expect_output(print(s), "tulpa simulated datasets")
})


# gcol33/tulpa#891: a vector `u`, or one with too few groups, reached build_eta
# as "incorrect number of dimensions" / "subscript out of bounds".
test_that("tulpa_simulate validates the shape of theta$u", {
  fam <- make_gaussian_family()
  th <- list(beta = c(0, 1), u = c(-1, 0, 1), extras = list(sigma_y = 1))
  expect_error(tulpa_simulate(y ~ x + (1 | g), fam, df, theta = th),
               "theta\\$u\\$y\\[\\[1\\]\\] must be a 5 x 1 numeric matrix")
  th$u <- list(y = list(matrix(0, 5, 2)))
  expect_error(tulpa_simulate(y ~ x + (1 | g), fam, df, theta = th),
               "got a 5 x 2 matrix")
  th$u <- list(y = list())
  expect_error(tulpa_simulate(y ~ x + (1 | g), fam, df, theta = th),
               "got nothing")
  # A length-n_groups vector is a single-coefficient block.
  th$u <- c(-2, -1, 0, 1, 2)
  th$extras$sigma_y <- 1e-6
  s <- tulpa_simulate(y ~ x + (1 | g), fam, df, theta = th, seed = 1)
  expect_equal(s$y[[1]], df$x + th$u[as.integer(df$g)], tolerance = 1e-4)
})


# A hand-built fit standing in for the two shapes #891 read as zero: an
# unnamed draws tail (mala) and a separate `$re` matrix (re_cov_gibbs), with a
# correlated slope term so the group-major layout is exercised.
.sim_fake_fit <- function(shape) {
  G <- 5L
  b <- cbind(seq(-2, 2, length.out = G), c(0.5, -0.5, 0.25, -0.25, 0))
  re_row <- as.numeric(t(b))              # group-major: g1c1, g1c2, g2c1, ...
  S <- 3L
  beta <- matrix(c(0.1, 0.2), S, 2L, byrow = TRUE)
  fit <- list(n_fixed = 2L, fixed_names = c("(Intercept)", "x"),
              re_layout = list(list(group_var = "g", levels = as.character(1:G),
                                    coef_labels = c("(Intercept)", "x"),
                                    n_groups = G, n_coefs = 2L)))
  re <- matrix(re_row, S, length(re_row), byrow = TRUE)
  if (shape == "tail") {
    fit$draws <- cbind(beta, re)
  } else {
    fit$draws <- beta
    colnames(fit$draws) <- fit$fixed_names
    fit$re <- re
  }
  structure(fit, class = "tulpa_fit")
}

test_that("tulpa_simulate(theta = fit) uses the fit's random effects (#891)", {
  fam <- make_gaussian_family()
  for (shape in c("tail", "re")) {
    fit <- .sim_fake_fit(shape)
    s <- tulpa_simulate(y ~ x + (1 + x | g), fam, df, theta = fit,
                        n_sims = 2, seed = 1)
    u <- s$theta[[1]]$u$y[[1]]
    expect_equal(u[, 1], seq(-2, 2, length.out = 5))
    expect_equal(u[, 2], c(0.5, -0.5, 0.25, -0.25, 0))
    gi <- as.integer(df$g)
    expect_equal(unname(s$linpred[[1]]$y),
                 0.1 + 0.2 * df$x + u[gi, 1] + df$x * u[gi, 2])
  }

  # Levels are matched by name, not position: a data subset holding groups
  # 2 and 4 only reads those groups' effects.
  sub <- droplevels(df[df$g %in% c("2", "4"), ])
  s <- tulpa_simulate(y ~ x + (1 + x | g), fam, sub, theta = .sim_fake_fit("re"))
  expect_equal(s$theta[[1]]$u$y[[1]][, 1], c(-1, 1))

  # A level the fit never saw, a term shape it does not have, and a fit with
  # no RE posterior are errors, never zeros.
  new <- df
  new$g <- factor(c(as.character(df$g[-1]), "9"))
  expect_error(tulpa_simulate(y ~ x + (1 + x | g), fam, new,
                              theta = .sim_fake_fit("re")),
               "level\\(s\\) of `g` the fit never saw \\(9\\)")
  expect_error(tulpa_simulate(y ~ x + (1 | g), fam, df,
                              theta = .sim_fake_fit("re")),
               "1 coefficient\\(s\\) in the formula but 2 in the fit")
  bare <- .sim_fake_fit("re")
  bare$re <- NULL
  expect_error(tulpa_simulate(y ~ x + (1 + x | g), fam, df, theta = bare),
               "carries no random-effect draws or point values")
})

test_that("tulpa_simulate(theta = fit) reproduces the fitted group means (#891)", {
  skip_on_cran()
  set.seed(1)
  d <- data.frame(x = rnorm(200), g = factor(rep(1:4, 50)))
  d$y <- rpois(200, exp(0.3 + 0.3 * d$x + c(-1.5, 0, 1, 2)[d$g]))
  fam <- tulpa_family("pois", function(eta, params, n_obs, ...) {
    rpois(n_obs, exp(eta[[1]]))
  })
  obs <- tapply(d$y, d$g, mean)
  for (m in c("mala", "laplace")) {
    f <- suppressWarnings(tulpa(y ~ x + (1 | g), d, family = "poisson",
                                mode = m))
    s <- tulpa_simulate(y ~ x + (1 | g), fam, d, theta = f, n_sims = 20,
                        seed = 1)
    sim <- rowMeans(sapply(s$y, function(y) tapply(y, d$g, mean)))
    # Before the fix every effect was zero and the four means were ~2.4 each.
    expect_lt(max(abs(log(sim) - log(obs))), 0.3)
  }
})
