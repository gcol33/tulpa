# A free-covariance block's stated hyperprior (`logchol_prior`): one joint
# density over the block's log-Cholesky columns, replacing the PC + LKJ default
# on the joint door and kept under hyperprior = "flat".

skip_on_cran()

.jlp_fit <- function(blk, hyperprior = "proper") {
  set.seed(3)
  G <- 20L; per <- 6L; n <- G * per
  grp <- rep(seq_len(G), each = per); x <- rnorm(n)
  y <- rpois(n, exp(0.2 + 0.4 * x + rnorm(G, 0, 0.5)[grp]))
  arm <- list(y = y, X = cbind(1, x), family = "poisson")
  b <- c(list(type = "miid", obs_idx = grp, n_groups = G, n_fields = 2L,
              field_weight = list(list(rep(1, n)), list(x)),
              logchol_grid = as.matrix(expand.grid(
                L11 = c(-1, -0.5, 0), L21 = c(-0.2, 0.2), L22 = c(-1.5, -1)))),
         blk)
  tulpa_nested_laplace_joint(list(y = arm), prior = list(b),
    hyperprior = hyperprior,
    control = list(diagnose_k = FALSE, axis_refine = "none",
                   var_of_means_consistency = FALSE, progress = FALSE))
}

test_that("a stated logchol_prior is the density folded on the block's cells", {
  skip_if_fast()
  dflt <- .jlp_fit(list())
  same <- .jlp_fit(list(logchol_prior = re_cov_pc_lkj_prior(2L)))
  expect_identical(same$log_marginal, dflt$log_marginal)

  f <- re_cov_pc_lkj_prior(2L, prior_sigma = c(1, 0.05), eta = 4)
  own <- .jlp_fit(list(logchol_prior = f))
  tg <- own$theta_grid[, c("b1.L11", "b1.L21", "b1.L22")]
  expect_equal(as.numeric(own$log_hyperprior), apply(tg, 1L, f),
               tolerance = 1e-12)
  expect_true(all(c("b1.L11", "b1.L21", "b1.L22") %in%
                  own$log_hyperprior_axes))
})

test_that("a stated logchol_prior is kept under hyperprior = 'flat'", {
  skip_if_fast()
  f <- function(th) -sum(th^2)
  flat <- .jlp_fit(list(logchol_prior = f), hyperprior = "flat")
  tg <- flat$theta_grid[, c("b1.L11", "b1.L21", "b1.L22")]
  expect_equal(as.numeric(flat$log_hyperprior), apply(tg, 1L, f),
               tolerance = 1e-12)
  expect_error(.jlp_fit(list(logchol_prior = 3)), "must be a function")
})
