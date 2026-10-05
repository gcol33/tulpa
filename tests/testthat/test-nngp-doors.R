# An NNGP field reached through each nested-Laplace door: the single-block
# kernel, the single-arm multi-block driver and the joint multi-block driver
# (gcol33/tulpa#943). All three build the block from make_nngp_block, so on
# one declared grid they are one model, gathered or read through a projector.

.nngp_door_data <- function(seed = 3L, n_loc = 40L, reps = 2L) {
  set.seed(seed)
  loc <- data.frame(lon = stats::runif(n_loc), lat = stats::runif(n_loc))
  d <- loc[rep(seq_len(n_loc), each = reps), ]
  rownames(d) <- NULL
  d$x <- stats::rnorm(nrow(d))
  d$count <- stats::rpois(nrow(d), exp(0.3 + 0.5 * d$x + 0.5 * sin(4 * d$lon)))
  blk <- tulpa:::.spatial_spec_to_nl_prior(
    tulpa:::validate_gp(spatial_gp(~ lon + lat, nn = 6), d))
  g <- expand.grid(s = c(0.2, 0.6), p = c(0.1, 0.3, 0.8))
  blk$sigma2_grid <- g$s
  blk$phi_gp_grid <- g$p
  N <- nrow(d)
  S <- matrix(0, N, blk$n_spatial)
  S[cbind(seq_len(N), blk$spatial_idx)] <- 1
  list(d = d, N = N, X = stats::model.matrix(~ x, d), blk = blk, S = S)
}

.nngp_door_ctl <- list(axis_refine = "none", prune = FALSE, diagnose_k = FALSE)

.nngp_registry <- function(dd, prior) {
  tulpa_nested_laplace(y = dd$d$count, n_trials = rep(1L, dd$N), X = dd$X,
                       family = "poisson", prior = prior,
                       control = .nngp_door_ctl)
}

.nngp_joint <- function(dd, blk) {
  tulpa_nested_laplace_joint(
    responses = list(a = list(y = dd$d$count, n_trials = rep(1L, dd$N),
                              X = dd$X, family = "poisson")),
    prior = list(blk), control = .nngp_door_ctl)
}


test_that("the joint door fits an nngp block as the registry door does", {
  skip_on_cran()
  dd <- .nngp_door_data()
  reg <- .nngp_registry(dd, dd$blk)
  jnt <- .nngp_joint(dd, dd$blk)
  expect_equal(unname(as.matrix(jnt$theta_grid)),
               unname(as.matrix(reg$theta_grid)))
  expect_equal(jnt$log_marginal, reg$log_marginal, tolerance = 1e-6)
  expect_equal(unname(coef(jnt)), unname(coef(reg)), tolerance = 1e-6)
})


test_that("a joint nngp block read through its incidence fits the gathered model", {
  skip_on_cran()
  dd <- .nngp_door_data()
  via_S <- dd$blk
  via_S$spatial_idx <- NULL
  via_S$projector <- list(dd$S)
  gathered <- .nngp_joint(dd, dd$blk)
  incident <- .nngp_joint(dd, via_S)
  expect_equal(incident$log_marginal, gathered$log_marginal, tolerance = 1e-6)
  expect_equal(unname(coef(incident)), unname(coef(gathered)),
               tolerance = 1e-6)
})


test_that("a restricted nngp block gives the same fit on the joint and registry doors", {
  skip_on_cran()
  dd <- .nngp_door_data()
  P <- tulpa:::.rsr_unit_projection(dd$X, dd$blk$spatial_idx,
                                    dd$blk$n_spatial)
  A <- P[dd$blk$spatial_idx, , drop = FALSE]
  restricted <- dd$blk
  restricted$spatial_idx <- NULL
  restricted$projector <- A
  reg <- .nngp_registry(dd, restricted)
  restricted$projector <- list(A)
  jnt <- .nngp_joint(dd, restricted)
  expect_equal(jnt$log_marginal, reg$log_marginal, tolerance = 1e-6)
  expect_equal(unname(coef(jnt)), unname(coef(reg)), tolerance = 1e-6)
})
