# Rscript identity_fits.R <lib> <tulpa repo> <out.rds> <arm: default|proper|flat>
# Fits four fixtures through the two front doors and saves the fit objects.
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]; repo <- args[2]; out <- args[3]; arm <- args[4]
.libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(tulpa, lib.loc = lib))
cat("tulpa from", find.package("tulpa"), " arm", arm, "\n")
env <- new.env(parent = asNamespace("tulpa"))
sys.source(file.path(repo, "tests/testthat/helper-outer-grid-dump.R"), envir = env)
hp <- if (arm == "default") list() else list(hyperprior = arm)

fits <- list()
# 1. multi-block joint, three crossed iid blocks, gaussian arm (the ogd fixture)
sim <- env$ogd_fixture_sim(c(0.8, 0.5, 0.3), 4242L)
prior3 <- lapply(seq_along(sim$grp), function(k) {
  s <- sim$sd_true[k]
  list(type = "iid", obs_idx = list(sim$grp[[k]]), n_units = sim$G,
       sigma_grid = exp(seq(log(s / 3), log(s * 3), length.out = 4L)))
})
fits$joint_multi <- do.call(tulpa_nested_laplace_joint, c(list(
  responses = list(a = list(y = sim$y, n_trials = rep(1L, sim$N), X = sim$X,
                            family = "gaussian", phi = 0.0625)),
  prior = prior3,
  control = list(n_threads = 1L, max_iter = 100L, tol = 1e-8,
                 integration = "grid")), hp))
# 3. single-block nested door, ICAR chain on a poisson response
S <- 30L
nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
site <- rep(seq_len(S), each = 4L)
Xs <- cbind(1, rnorm(length(site)))
phi_s <- cumsum(rnorm(S, 0, 0.3)); phi_s <- phi_s - mean(phi_s)
nn <- lengths(nb)
icar <- list(type = "icar", n_spatial_units = S, spatial_idx = site,
             adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(nb) - 1L,
             n_neighbors = nn)
yp <- rpois(length(site), exp(drop(Xs %*% c(0.5, 0.3)) + phi_s[site]))
fits$nl_icar <- do.call(tulpa_nested_laplace, c(list(
  y = yp, n_trials = rep(1L, length(yp)), X = Xs,
  prior = icar,
  family = "poisson", control = list(n_threads = 1L)), hp))
# 2. single-block joint, ICAR chain on a binomial arm (default grid, k-hat on)
yb <- rbinom(length(site), 1L, plogis(drop(Xs %*% c(-0.2, 0.6)) + phi_s[site]))
fits$joint_single <- do.call(tulpa_nested_laplace_joint, c(list(
  responses = list(a = list(y = yb, n_trials = rep(1L, length(yb)), X = Xs,
                            spatial_idx = as.integer(site), family = "binomial")),
  prior = icar[c("type", "n_spatial_units", "adj_row_ptr", "adj_col_idx",
                 "n_neighbors")],
  control = list(n_threads = 1L)), hp))
# 4. multi-block nested door: the same ICAR plus an iid block
grp2 <- sample.int(12L, length(site), TRUE)
fits$nl_multi <- do.call(tulpa_nested_laplace, c(list(
  y = yp, n_trials = rep(1L, length(yp)), X = Xs,
  prior = list(icar, list(type = "iid", obs_idx = grp2, n_units = 12L)),
  family = "poisson", control = list(n_threads = 1L)), hp))
saveRDS(fits, out)
cat("saved", out, "\n")
