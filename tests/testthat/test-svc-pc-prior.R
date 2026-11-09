svc_pc_fixture <- function(approx = "nngp", parameterization = "noncentered",
                           anchor = c(1, 0.01)) {
  d <- data.frame(lon = c(0, 0.8, 0.2, 1), lat = c(0.1, 0, 1, 0.7),
                  x = c(-1, 0.5, 1, -0.5), y = 0)
  X <- model.matrix(~ x, d)
  sp <- validate_svc(spatial_svc(~ lon + lat, terms = 1:2, nn = 3L,
    approx = approx, m = 3L, c_boundary = 1.5,
    parameterization = parameterization, sigma_prior_U = anchor[1],
    sigma_prior_alpha = anchor[2]), d, X)
  list(d = d, X = X, spec = tulpa:::.svc_sampler_spec(sp, X))
}

svc_pc_density <- function(f, theta = matrix(0, 0, 0)) {
  load_cpp_fixture("smc_normalizer")
  cpp_test_log_prior_normalized(theta, f$d$y, integer(0), f$X,
    "gaussian", 1, 2.5, 2.5, svc_spec = f$spec)
}

test_that("SVC scale anchors share the HSGP and TVC defaults", {
  for (approx in c("nngp", "hsgp")) {
    s <- spatial_svc(~lon + lat, approx = approx)
    h <- spatial_gp(~lon + lat, approx = "hsgp")
    t <- temporal_tvc("time")
    expect_equal(c(s$sigma_prior_U, s$sigma_prior_alpha),
                 c(h$sigma2_prior_U, h$sigma2_prior_alpha))
    expect_equal(c(s$sigma_prior_U, s$sigma_prior_alpha),
                 c(t$sigma_prior_U, t$sigma_prior_alpha))
    for (bad in list(c(0, .01), c(-1, .01), c(Inf, .01), c(NA, .01),
                     c(1, 0), c(1, 1), c(1, 2), c(1, NA))) {
      expect_error(spatial_svc(~lon+lat, approx = approx,
        sigma_prior_U = bad[1], sigma_prior_alpha = bad[2]), "sigma_prior")
    }
  }
})

test_that("each SVC basis carries the normalized PC density and log-variance Jacobian", {
  load_cpp_fixture("smc_normalizer")
  for (approx in c("nngp", "hsgp")) {
    for (anchor in list(c(1, 0.01), c(2.5, 0.2))) {
      f <- svc_pc_fixture(approx, anchor = anchor)
      expect_equal(c(f$spec$sigma_prior_U, f$spec$sigma_prior_alpha), anchor)
      probe <- svc_pc_density(f)
      theta <- matrix(0, 3L, length(probe$names),
                       dimnames = list(NULL, probe$names))
      log_s2 <- 2 * log(c(0.08, 0.4, 1.2))
      theta[, "log_sigma2_svc[1]"] <- log_s2
      theta[, "log_sigma2_svc[2]"] <- 2 * log(0.6)
      theta[, grep("^log_phi_svc", probe$names)] <- log(0.3)
      got <- svc_pc_density(f, theta)
      range_rate <- -f$spec$phi_prior_U * log(f$spec$phi_prior_alpha)
      rest <- 2 * dnorm(0, sd = 2.5, log = TRUE) +
        sum(grepl("^svc_w", probe$names)) * dnorm(0, log = TRUE) +
        2 * (log(range_rate) - log(0.3) - range_rate / 0.3)
      rate <- -log(anchor[2]) / anchor[1]
      want <- rest + dexp(exp(log_s2 / 2), rate, log = TRUE) +
        log_s2 / 2 - log(2) + dexp(0.6, rate, log = TRUE) + log(0.6 / 2)
      expect_length(got$declined, 0L)
      expect_equal(got$log_prior, want, tolerance = 1e-11,
                   info = paste(approx, paste(anchor, collapse = ",")))
    }
  }
})

test_that("centered SVC scale anchors change the shipped density by the PC prior ratio", {
  load_cpp_fixture("smc_normalizer")
  f <- svc_pc_fixture(parameterization = "centered")
  probe <- svc_pc_density(f)
  theta <- matrix(0, 3L, length(probe$names),
                   dimnames = list(NULL, probe$names))
  sigma <- c(.08, .4, 1.2)
  theta[, "log_sigma2_svc[1]"] <- 2 * log(sigma)
  first <- svc_pc_density(f, theta)$log_prior
  f$spec$sigma_prior_U <- 2.5
  f$spec$sigma_prior_alpha <- .2
  second <- svc_pc_density(f, theta)$log_prior
  ratio <- function(s) dexp(s, -log(.2) / 2.5, log = TRUE) -
    dexp(s, -log(.01), log = TRUE)
  expect_equal(second - first, ratio(sigma) + ratio(1), tolerance = 1e-11)
})

test_that("the SVC compiled entry validates anchors and initializes every noncentered scale", {
  load_cpp_fixture("smc_normalizer")
  for (approx in c("nngp", "hsgp")) {
    f <- svc_pc_fixture(approx, anchor = c(2.5, .2))
    init <- function() cpp_test_sampler_init(f$d$y, integer(0), f$X,
      "gaussian", 1, 2.5, 2.5, svc_spec = f$spec)
    s <- init()
    expect_equal(unname(s[grep("^log_sigma2_svc", names(s))]),
                 rep(2 * log(log(2) * 2.5 / -log(.2)), 2), tolerance = 1e-12)
    f$spec$sigma_prior_U <- f$spec$sigma_prior_alpha <- NULL
    s <- init()
    expect_equal(unname(s[grep("^log_sigma2_svc", names(s))]),
                 rep(2 * log(log(2) / -log(.01)), 2), tolerance = 1e-12)
    for (bad in list(c(0, .01), c(Inf, .01), c(1, 0), c(1, 1), c(1, NA))) {
      f$spec$sigma_prior_U <- bad[1]
      f$spec$sigma_prior_alpha <- bad[2]
      expect_error(init(), "spatial_svc.*SD.*sigma_prior")
    }
  }
})

test_that("weighted SVC nested blocks retain their stated PC scale density", {
  skip_on_cran_budget()
  sigma <- c(.08, .4, 1.2)
  for (type in c("nngp", "hsgp")) {
    for (anchor in list(c(1, .01), c(2.5, .2))) {
      block <- list(type = type, prior_sigma = anchor)
      if (type == "hsgp") block$svc_column <- 2L else block$svc_weight <- list(c(-1, 1))
      want <- dexp(sigma, -log(anchor[2]) / anchor[1], log = TRUE) + log(sigma / 2)
      for (choice in c("proper", "flat")) {
        density <- tulpa:::.hp_axis_prior("sigma2", block, hyperprior = choice)$fn
        expect_true(is.function(density))
        expect_equal(density(sigma^2), want, tolerance = 1e-12)
      }
    }
  }
})
