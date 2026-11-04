test_that("continuous spatial scale anchors reach nested and sampler routes", {
  skip_on_cran_budget()
  d <- data.frame(lon = c(0, 1, .2, .7), lat = c(0, .1, 1, .8))
  for (approximation in c("nngp", "hsgp")) {
    for (stated in c(FALSE, TRUE)) {
      args <- list(coords = ~ lon + lat, approx = approximation)
      if (approximation == "hsgp") args <- c(args, list(m = 3L, c = 1.5))
      if (stated) args <- c(args, list(sigma_prior_U = .7, sigma_prior_alpha = .2))
      sp <- do.call(spatial_gp, args)
      sp <- if (approximation == "hsgp") validate_hsgp(sp, d) else validate_gp(sp, d)
      block <- tulpa:::.spatial_spec_to_nl_prior(sp)
      sampler <- if (approximation == "hsgp") tulpa:::.hsgp_sampler_spec(sp) else
        tulpa:::.gp_sampler_spec(sp)
      anchor <- c(sampler$sigma2_prior_U, sampler$sigma2_prior_alpha)
      x <- exp(seq(-4, 2, length.out = 30L))
      rate <- -log(anchor[2L]) / anchor[1L]
      expected <- log(rate / 2) + log(x) / 2 - rate * sqrt(x)
      proper <- tulpa:::.hp_axis_prior("sigma2", block)$fn
      expect_equal(proper(x), expected, tolerance = 1e-12)
      flat <- tulpa:::.hp_axis_prior("sigma2", block, hyperprior = "flat")$fn
      if (stated) {
        expect_true(is.function(flat))
        if (is.function(flat)) expect_equal(flat(x), expected, tolerance = 1e-12)
      } else expect_null(flat)
    }
  }
})

test_that("SPDE scale and range anchors survive conversion to a nested block", {
  skip_on_cran_budget()
  n <- 6L
  C <- Matrix::Diagonal(n)
  G <- Matrix::bandSparse(n, k = c(-1, 0, 1),
    diagonals = list(rep(-1, n - 1), c(1, rep(2, n - 2), 1), rep(-1, n - 1)))
  coords <- cbind(seq_len(n), c(1, 3, 2, 5, 6, 4))
  for (stated in c(FALSE, TRUE)) {
    args <- list(C = C, G = G, A = C, coords = coords)
    if (stated) args <- c(args, list(prior_range = c(.7, .2), prior_sigma = c(.8, .1)))
    sp <- do.call(spatial_spde_custom, args)
    block <- tulpa:::.spatial_spec_to_nl_prior(sp)
    x <- exp(seq(-4, 2, length.out = 30L))
    for (axis in c("sigma", "range")) {
      anchor <- sp[[paste0("prior_", axis)]]
      rate <- -log(anchor[2L]) * if (axis == "sigma") 1 / anchor[1L] else anchor[1L]
      expected <- if (axis == "sigma") log(rate) + log(x) - rate * x else
        log(rate) - log(x) - rate / x
      proper <- tulpa:::.hp_axis_prior(axis, block)$fn
      expect_true(is.function(proper))
      if (is.function(proper)) expect_equal(proper(x), expected, tolerance = 1e-12)
      flat <- tulpa:::.hp_axis_prior(axis, block, hyperprior = "flat")$fn
      if (stated) {
        expect_true(is.function(flat))
        if (is.function(flat)) expect_equal(flat(x), expected, tolerance = 1e-12)
      } else expect_null(flat)
    }
  }
})
