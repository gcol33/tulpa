# The accessors, driven on fits the front door produced (gcol33/tulpa#607,
# #608, #609).
#
# test-svc-nuts-frontdoor.R and test-tvc-nuts-frontdoor.R assert on
# colnames(fit$draws) and on the parameter counts, which is why all three
# accessors could be unreachable for every fit tulpa() makes while the suite
# stayed green: `svc()` read `fit$svc` and `fit$.internal$svc_draws`, and the
# front door sets neither. So the assertions here call the accessor, and check
# what it returns against the raw draw columns rather than against itself --
# a reshape that runs is not a reshape that pairs the right coefficient with
# the right unit.
#
# The flat layouts the checks encode are the ones the eta assembly indexes:
# `w_flat[j * n_obs + i]` for SVC (src/hmc_svc.h) and
# `w_flat[(g * n_tvc + j) * n_times + t]` for TVC (src/hmc_tvc.h).

svc_fixture <- function(n = 40L, seed = 303L) {
  set.seed(seed)
  df <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  bsurf <- 0.9 * sin(2.8 * df$lon) + 0.7 * cos(2.2 * df$lat)
  df$count <- rpois(n, exp(0.2 + (0.8 + bsurf) * df$x))
  tulpa(count ~ x, data = df, family = "poisson",
        spatial = spatial_svc(~ lon + lat, terms = ~ x - 1, nn = 5L),
        mode = "exact",
        control = list(n_iter = 60L, n_warmup = 30L, seed = 1L))
}

tvc_fixture <- function(n_t = 8L, reps = 4L, seed = 160L) {
  set.seed(seed)
  walk <- cumsum(rnorm(n_t, 0, 0.35)); walk <- walk - mean(walk)
  year <- rep(seq_len(n_t), each = reps)
  d <- data.frame(year = year, x = rnorm(length(year)))
  d$count <- rpois(nrow(d), exp(0.3 + (0.5 + walk[year]) * d$x))
  tulpa(count ~ x, data = d, family = "poisson",
        temporal = temporal_tvc("year", terms = ~ x - 1, structure = "rw1"),
        mode = "exact",
        control = list(n_iter = 60L, n_warmup = 30L, seed = 1L))
}

plain_fixture <- function(n = 40L, seed = 2L) {
  set.seed(seed)
  d <- data.frame(x = rnorm(n))
  d$y <- rpois(n, exp(0.4 + 0.2 * d$x))
  tulpa(y ~ x, data = d, family = "poisson", mode = "hmc",
        control = list(n_iter = 60L, n_warmup = 30L, seed = 1L))
}


test_that("tulpa() attaches the validated temporal spec to the fit (#608)", {
  skip_if_fast()
  fit <- tvc_fixture()
  expect_s3_class(fit$temporal, "tulpa_tvc")
  expect_identical(fit$temporal$time_var, "year")
  expect_identical(fit$temporal$n_times, 8L)
  expect_identical(fit$temporal$tvc_names, "x")
})


test_that("svc() reads a front-door SVC fit, unit for unit (#607)", {
  skip_if_fast()
  fit <- svc_fixture()
  post <- svc(fit)

  expect_s3_class(post, "tulpa_svc_posterior")
  expect_identical(dim(post$draws), c(nrow(fit$draws), 40L, 1L))
  expect_identical(post$term_names, fit$spatial$svc_names)
  expect_identical(post$n_obs, 40L)

  # Every element against the column the sampler wrote it to.
  n_obs <- 40L
  for (j in seq_len(dim(post$draws)[3L])) {
    for (i in seq_len(n_obs)) {
      expect_equal(unname(post$draws[, i, j]),
                   unname(fit$draws[, paste0("svc_w[", (j - 1L) * n_obs + i, "]")]))
    }
  }

  s <- summary(post)
  expect_identical(nrow(s), n_obs * dim(post$draws)[3L])
  expect_equal(unname(s$mean[seq_len(n_obs)]),
               unname(colMeans(post$draws[, , 1L])))
})


test_that("tvc() reads a front-door TVC fit, time point for time point (#608)", {
  skip_if_fast()
  fit <- tvc_fixture()
  post <- tvc(fit)

  expect_s3_class(post, "tulpa_tvc_posterior")
  expect_identical(dim(post$draws), c(nrow(fit$draws), 8L, 1L))
  expect_identical(post$time_levels, fit$temporal$time_levels)
  expect_identical(post$n_times, 8L)

  n_times <- 8L
  for (j in seq_len(dim(post$draws)[3L])) {
    for (tt in seq_len(n_times)) {
      expect_equal(unname(post$draws[, tt, j]),
                   unname(fit$draws[, paste0("tvc_w[", (j - 1L) * n_times + tt, "]")]))
    }
  }

  s <- summary(post)
  expect_identical(nrow(s), n_times * dim(post$draws)[3L])
})


test_that("temporal() reads a multi-scale fit and selects a component (#609)", {
  skip_if_fast()
  set.seed(131)
  d <- data.frame(year = 1:40, x = rnorm(40))
  d$count <- rpois(40, exp(1 + 0.2 * d$x))
  fit <- tulpa(count ~ x, data = d, family = "poisson",
               temporal = temporal_multiscale("year", trend = "rw2", seasonal = 12),
               mode = "exact",
               control = list(n_iter = 60L, n_warmup = 30L, seed = 1L))

  expect_s3_class(fit$temporal, "tulpa_temporal_multiscale")

  post <- temporal(fit)
  expect_s3_class(post, "tulpa_temporal_posterior")
  expect_true(is.list(post$draws))
  expect_identical(names(post$draws), fit$temporal$components)
  expect_true("trend" %in% names(post$draws))

  # The component arms carry the columns the sampler named for them.
  for (k in c(1L, 5L, 40L)) {
    expect_equal(unname(post$draws$trend[, k]),
                 unname(fit$draws[, paste0("trend[", k, "]")]))
  }

  one <- temporal(fit, component = "trend")
  expect_true(is.matrix(one$draws))
  expect_equal(one$draws, post$draws$trend)

  expect_error(temporal(fit, component = "nope"), "not in model")
})


test_that("temporal() reads a single-component field (#609)", {
  skip_if_fast()
  set.seed(7)
  d <- data.frame(t = 1:30, x = rnorm(30))
  d$y <- rpois(30, exp(0.5 + 0.3 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "poisson",
               temporal = temporal_rw1("t"), mode = "exact",
               control = list(n_iter = 60L, n_warmup = 30L, seed = 1L))

  post <- temporal(fit)
  expect_s3_class(post, "tulpa_temporal_posterior")
  expect_true(is.matrix(post$draws))
  expect_identical(ncol(post$draws), fit$temporal$n_times)
  expect_equal(unname(post$draws[, 3L]),
               unname(fit$draws[, "phi_temporal[3]"]))
})


test_that("the accessors refuse a fit that carries no such field", {
  skip_if_fast()
  plain <- plain_fixture()
  expect_error(svc(plain), "spatially-varying")
  expect_error(tvc(plain), "temporally-varying")
  expect_error(temporal(plain), "not fitted with temporal effects")

  # A field of the other kind is not read as this one.
  expect_error(tvc(svc_fixture()), "temporally-varying")
  expect_error(svc(tvc_fixture()), "spatially-varying")
})


test_that("a grouped TVC field is refused rather than collapsed to group 1", {
  # `tulpa_tvc_posterior` has no group axis, and the flat layout is
  # group-major, so reshaping a grouped field into it would report the first
  # group as the whole field.
  info <- structure(
    list(type = "tvc", group_var = "site", n_groups = 3L, n_times = 4L,
         n_tvc = 1L, tvc_names = "x", time_levels = as.character(1:4),
         structure = "rw1"),
    class = c("tulpa_tvc", "tulpa_temporal", "list")
  )
  draws <- matrix(0, nrow = 5L, ncol = 12L,
                  dimnames = list(NULL, paste0("tvc_w[", 1:12, "]")))
  fake <- structure(list(temporal = info, draws = draws), class = "tulpa_fit")
  expect_error(tvc(fake), "no group dimension")
})


test_that(".draws_by_prefix orders by index, not by column position", {
  m <- matrix(seq_len(3L * 4L), nrow = 3L,
              dimnames = list(NULL, c("w[10]", "w[2]", "w[1]", "other")))
  got <- tulpa:::.draws_by_prefix(m, "w")
  expect_identical(colnames(got), c("w[1]", "w[2]", "w[10]"))
  expect_null(tulpa:::.draws_by_prefix(m, "nope"))
  expect_null(tulpa:::.draws_by_prefix(NULL, "w"))
})
