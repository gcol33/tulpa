# Inline areal varying-coefficient field: spatial(graph, ~ 1 + time || cell).
#
# The bar's LHS expands (via model.matrix) to one independent CAR/Besag field
# per design column -- the intercept column is the spatial intercept field, a
# covariate column is a spatially varying slope on it (a per-region trend).
# Coverage:
#   (1) Construction + model.matrix expansion to per-column blocks.
#   (2) Bar-grammar validation (|| only; no |, no nesting, no expression RHS,
#       must have a bar, by= reserved, proper reserved).
#   (3) Formula integration: the field's own bar is not leaked into the model
#       random effects, the fixed part is stripped, and the bare spatial(col)
#       path is untouched.
#   (4) Parameter recovery end to end through tulpa().

.chain_adj_field <- function(n_s) {
  adj <- matrix(0, n_s, n_s)
  for (i in seq_len(n_s - 1L)) adj[i, i + 1L] <- adj[i + 1L, i] <- 1
  adj
}

# --------------------------------------------------------------------------- #
# (1) Construction + expansion                                                #
# --------------------------------------------------------------------------- #

test_that("spatial() builds a field spec and expands one CAR block per column", {
  adj <- .chain_adj_field(10L)
  f <- spatial(graph = adj, formula = ~ 1 + time || cell)
  expect_s3_class(f, "tulpa_spatial_field")
  expect_identical(f$group_var, "cell")
  expect_false(f$correlated)
  expect_false(f$proper)
  expect_identical(as.integer(f$n_spatial), 10L)

  d <- data.frame(cell = rep(1:10, each = 4), time = rnorm(40))
  cols <- tulpa:::.bar_field_columns(f, d)
  expect_length(cols, 2L)
  expect_identical(cols[[1]]$name, "Intercept")
  expect_true(cols[[1]]$is_intercept)
  expect_true(all(cols[[1]]$weight == 1))
  expect_identical(cols[[2]]$name, "time")
  expect_false(cols[[2]]$is_intercept)
  expect_equal(cols[[2]]$weight, d$time)

  blks <- tulpa:::.spatial_field_blocks(f, d)
  expect_length(blks, 2L)
  expect_identical(vapply(blks, function(b) b$name, ""),
                   c("cell.Intercept", "cell.time"))
  # The intercept field carries no svc_weight (it is the all-ones column); the
  # slope field carries the covariate as its per-row weight.
  expect_null(blks[[1]]$svc_weight)
  expect_equal(blks[[2]]$svc_weight[[1]], d$time)
  expect_identical(blks[[1]]$type, "icar")
})

test_that("spatial() intercept-only and slope-only expand correctly", {
  adj <- .chain_adj_field(6L)
  d <- data.frame(cell = rep(1:6, each = 3), time = rnorm(18))

  f_int <- spatial(graph = adj, formula = ~ 1 || cell)
  ci <- tulpa:::.bar_field_columns(f_int, d)
  expect_length(ci, 1L)
  expect_true(ci[[1]]$is_intercept)

  f_slope <- spatial(graph = adj, formula = ~ 0 + time || cell)
  cs <- tulpa:::.bar_field_columns(f_slope, d)
  expect_length(cs, 1L)
  expect_identical(cs[[1]]$name, "time")
  expect_false(cs[[1]]$is_intercept)
})

# --------------------------------------------------------------------------- #
# (2) Bar-grammar validation                                                  #
# --------------------------------------------------------------------------- #

test_that("spatial() enforces the bar grammar", {
  adj <- .chain_adj_field(8L)
  # Single bar | now builds correlated fields (separable MCAR); see
  # test-spatial-mcar.R. It constructs without error.
  expect_true(spatial(graph = adj, formula = ~ 1 + time | cell)$correlated)
  # Nested grouping is rejected.
  expect_error(spatial(graph = adj, formula = ~ 1 || cell / site),
               "Nested grouping")
  # Interaction / expression RHS is rejected.
  expect_error(spatial(graph = adj, formula = ~ 1 || cell:site),
               "single bare graph-node")
  # A bar is required.
  expect_error(spatial(graph = adj, formula = ~ 1 + time),
               "grouping bar")
  # by= builds a replicated-CAR field (see test-spatial-field-by.R); it
  # constructs and records the replication factor as a column name.
  f_by <- spatial(graph = adj, formula = ~ 1 || cell, by = hab)
  expect_identical(f_by$by_var, "hab")
  expect_identical(spatial(graph = adj, formula = ~ 1 || cell,
                           by = "hab")$by_var, "hab")
  # proper CAR is now wired (independent fields with estimated rho_car); see
  # test-spatial-proper-car.R. It constructs without error.
  expect_s3_class(spatial(graph = adj, formula = ~ 1 || cell, proper = TRUE),
                  "tulpa_spatial_field")
  # Non-square / non-symmetric graphs are rejected.
  expect_error(spatial(graph = matrix(0, 3, 4), formula = ~ 1 || cell),
               "square")
})

# --------------------------------------------------------------------------- #
# (3) Formula integration                                                     #
# --------------------------------------------------------------------------- #

test_that("inline spatial() does not leak its bar into random effects", {
  adj <- .chain_adj_field(10L)
  p <- tulpa_parse_formula(
    y ~ time + spatial(graph = adj, formula = ~ 1 + time || cell) + (1 | site))
  expect_identical(p$n_spatial_field_blocks, 1L)
  # The (1 | site) bar is the ONLY random effect; the field's own || cell bar
  # must not be counted as a model random effect.
  expect_identical(p$n_re_terms, 1L)
  expect_identical(p$random_effects[[1]]$group_var, "site")
  # The field's covariate is stripped from the fixed part (only the marginal
  # `time` fixed effect remains).
  expect_identical(deparse(p$fixed_formula), "y ~ time")
  # The bare spatial(col) areal path is untouched by the field machinery.
  expect_null(p$spatial_var)
})

test_that("the bare spatial(col) areal-naming path still parses", {
  p <- tulpa_parse_formula(y ~ x + spatial(region))
  expect_identical(p$spatial_var, "region")
  expect_identical(p$n_spatial_field_blocks, 0L)
})

# --------------------------------------------------------------------------- #
# (4) Parameter recovery end to end                                           #
# --------------------------------------------------------------------------- #

test_that("spatial() recovers the intercept and slope CAR fields", {
  skip_on_cran()
  skip_if_fast()
  set.seed(1)

  n_s <- 30L
  adj <- .chain_adj_field(n_s)
  u <- cumsum(rnorm(n_s)); u <- (u - mean(u)) / sd(u)
  s <- cumsum(rnorm(n_s)); s <- (s - mean(s)) / sd(s)

  n_per <- 40L
  cell <- rep(seq_len(n_s), each = n_per)
  N <- length(cell)
  time <- rnorm(N)
  b0 <- 0.3; b1 <- -0.5; sig_u <- 1.1; sig_s <- 0.9
  eta <- b0 + b1 * time + sig_u * u[cell] + time * (sig_s * s[cell])
  y <- rnorm(N, eta, 0.4)
  d <- data.frame(y = y, time = time, cell = cell)

  fit <- suppressWarnings(tulpa(
    y ~ time + spatial(graph = adj, formula = ~ 1 + time || cell),
    data = d, family = "gaussian", mode = "laplace",
    control = list(n_draws = 400L)))

  expect_s3_class(fit, "tulpa_fit")
  expect_setequal(fit$spatial_field_names, c("cell.Intercept", "cell.time"))

  u_hat <- fit$spatial_fields[["cell.Intercept"]]$mean
  s_hat <- fit$spatial_fields[["cell.time"]]$mean
  expect_gt(abs(cor(u_hat, u)), 0.7)
  expect_gt(abs(cor(s_hat, s)), 0.7)

  # Fixed effects recovered.
  cf <- coef(fit)
  expect_lt(abs(cf[["(Intercept)"]] - b0), 0.15)
  expect_lt(abs(cf[["time"]] - b1), 0.15)

  # Generic methods work on the wrapped fit.
  sm <- summary(fit)
  expect_true(is.data.frame(sm) || is.matrix(sm))
})
