test_that("all fitting doors reject an invalid device before fitting", {
  calls <- list(
    function() tulpa(y ~ 1, data.frame(y = 1:3), control = list(device = "invalid")),
    function() tulpa_nested_laplace(1:3, rep(1L, 3), matrix(1, 3, 1),
                                   control = list(device = "invalid")),
    function() tulpa_nested_laplace_joint(NULL, NULL, control = list(device = "invalid")),
    function() tulpa_nuts_beta(NULL, NULL, control = list(device = "invalid")),
    function() tulpa_nuts_spde(NULL, NULL, NULL, control = list(device = "invalid")),
    function() tulpa:::tulpa_sample_glmm(NULL, NULL, NULL, "gaussian", "nuts",
                                        control = list(device = "invalid")))
  for (call in calls) expect_error(call(), "control\\$device must be")
})

skip_on_cran()

test_that("registry and joint NNGP doors record their actual device stages", {
  skip_on_cran()
  skip_if_no_cuda("cusolver")
  args <- device_nngp_fixture(64L, 10L)
  n <- length(args$w)
  prior <- list(type = "nngp", coords = args$coords, nn_idx = args$nn_idx,
                nn_dist = args$nn_dist, nn_order = args$nn_order + 1L,
                n_spatial = n, nn = 10L, cov_type = 0L,
                sigma2_grid = c(0.9, 1.1), phi_gp_grid = c(0.4, 0.6))
  X <- matrix(1, n, 1, dimnames = list(NULL, "intercept"))
  y <- args$w + 0.1 * cos(seq_len(n))
  runners <- list(
    registry = function(control) tulpa_nested_laplace(y, rep(1L, n), X,
      prior = prior, family = "gaussian", control = control),
    joint = function(control) tulpa_nested_laplace_joint(
      responses = list(a = list(y = y, n_trials = rep(1L, n), X = X,
                                family = "gaussian", phi = 1)),
      prior = list(field = c(prior, list(spatial_idx = list(seq_len(n))))),
      control = control))
  for (run in runners) {
    fits <- lapply(c("cpu", "auto", "cuda"), function(choice) {
      fit <- run(list(device = choice, diagnose_k = FALSE, auto_recenter = FALSE,
                      adaptive_grid = FALSE, prune = FALSE, max_iter = 100L,
                      tol = 1e-8, n_threads = 1L))
      expect_identical(fit$device$requested, choice)
      stage <- fit$device$stages[fit$device$stages$stage == "nngp_cholesky", , drop = FALSE]
      expect_equal(nrow(stage), 1L)
      want <- tulpa:::cpp_device_probe("nngp_cholesky", 64, 10L, choice)$device
      expect_identical(stage$device, want)
      expect_true(length(coef(fit)) > 0)
      expect_true(all(is.finite(coef(fit))))
      fit
    })
    for (i in 2:3) {
      expect_device_equal(coef(fits[[i]]), coef(fits[[1L]]), 1e-8, "fixed effects")
      expect_device_equal(fits[[i]]$log_marginal, fits[[1L]]$log_marginal,
                          1e-8, "cell log marginal")
    }
  }
})
