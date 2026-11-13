test_that("true-density data augmentation agrees with an independent sparse ZIP sampler", {
  skip_if_not_slow()
  source(test_path("fixtures", "em_exact_zip.R"), local = TRUE)
  fixture <- .em_zip_fixture()
  set.seed(117501L)
  fit <- tulpa_em_mc(fixture$e_sample, fixture$encode, n_mc = 4L,
                      n_mc_max = 20L, max_iter = 6L, verbose = FALSE,
                      beta_prior = fixture$beta_prior,
                      correction = "gibbs", n_gibbs = 12000L,
                      correction_control = list(warmup = 2000L, mh_steps = 2L))
  set.seed(117502L)
  reference <- .em_zip_reference(fixture, n_iter = 60000L, warmup = 10000L)
  colnames(reference$draws) <- colnames(fit$draws)
  comparison <- .em_summary_compare(.em_chain_summary(fit$draws),
                                    .em_chain_summary(reference$draws))
  expect_true(all(abs(comparison$z_mcse) < 5),
                info = paste(capture.output(print(comparison)), collapse = "\n"))
  z <- do.call(rbind, fit$latent_draws)
  joint <- cbind(fit$draws, z0 = z[, 1L], alpha_z0 = fit$draws[, 1L] * z[, 1L])
  reference_joint <- cbind(reference$draws, z0 = reference$latent[, 1L],
                             alpha_z0 = reference$draws[, 1L] * reference$latent[, 1L])
  sj <- subset(.em_chain_summary(joint), statistic == "mean")
  rj <- subset(.em_chain_summary(reference_joint), statistic == "mean")
  pair_comparison <- .em_summary_compare(sj, rj)
  expect_true(all(abs(pair_comparison$z_mcse) < 5),
                info = paste(capture.output(print(pair_comparison)), collapse = "\n"))
  expect_identical(fit$inference_tier, 1L)
  expect_identical(fit$draws_kind, "chain")
  expect_gt(fit$mean_accept, .2)
  expect_equal(nrow(fit$draws), length(fit$latent_draws))
  expect_true(all(z[, fixture$y > 0, drop = FALSE] == 1))
  expect_identical(fit$pooled_target, "Rubin-pooled complete-data Laplace estimates")
})
