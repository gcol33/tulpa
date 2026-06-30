# Joint nested-Laplace: prior_phi re-weights the dispersion (phi_<arm>) axis.
#
# phi_grid lifts a per-arm dispersion off the parse-time scalar onto the outer
# grid (test-nested-laplace-joint-phi-grid.R). Without a hyperprior that axis
# sits on an implicit flat prior over its bounds. prior_phi mirrors prior_sigma:
# a single PC / half-normal spec re-weights every phi_<arm> cell by its density
# (gcol33/tulpa#139). The mechanism is family-agnostic (phi is just a grid
# axis), so the well-exercised gaussian copy-arm fixture isolates it.

.chain_adj_pp <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    n_neighbors <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(n_neighbors))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(n_neighbors), n_spatial_units = n_s)
}

.simulate_joint_pp <- function(N = 400, n_s = 30, sigma = 0.6, rho = 0.7,
                               beta_occ = c(-0.3, 0.5), beta_pos = c(0.2, -0.4),
                               sd_pos = 0.3, seed = 7) {
    set.seed(seed)
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    w_s <- sigma * (sqrt(rho) * rnorm(n_s) + sqrt(1 - rho) * rnorm(n_s))
    x <- rnorm(N); Xocc <- cbind(1, x)
    occur <- rbinom(N, 1, plogis(as.numeric(Xocc %*% beta_occ) + w_s[spatial_idx]))
    is_pos <- occur == 1L
    Xpos <- Xocc[is_pos, , drop = FALSE]; spi_pos <- spatial_idx[is_pos]
    eta_pos <- as.numeric(Xpos %*% beta_pos) + w_s[spi_pos]
    y_pos <- eta_pos + rnorm(sum(is_pos), 0, sd_pos)
    list(N = N, n_s = n_s, spatial_idx = as.integer(spatial_idx),
         Xocc = Xocc, occur = occur, Xpos = Xpos, y_pos = y_pos,
         spi_pos = as.integer(spi_pos), sd_pos = sd_pos)
}

.fit_pp <- function(sim, adj, phi_axis, prior_phi = NULL) {
    arm_occ <- list(y = as.numeric(sim$occur), n_trials = rep(1L, sim$N),
                    X = sim$Xocc, spatial_idx = sim$spatial_idx,
                    re_idx = rep(0, sim$N), n_re_groups = 0L, sigma_re = 1.0,
                    family = "binomial", phi = 1.0)
    arm_pos <- list(y = sim$y_pos, n_trials = rep(1L, length(sim$y_pos)),
                    X = sim$Xpos, spatial_idx = sim$spi_pos,
                    re_idx = rep(0, length(sim$y_pos)), n_re_groups = 0L,
                    sigma_re = 1.0, family = "gaussian", phi = 1.0,
                    field_coef = list(name = "alpha", grid = 1))
    prior <- list(type = "bym2", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, scale_factor = 1.0,
                  sigma_grid = 0.6, rho_grid = 0.7)
    tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior = prior, phi_grid = list(pos = phi_axis),
        prior_phi = prior_phi,
        control = list(diagnose_k = FALSE))
}

test_that("prior_phi adds exactly the PC log-density to every phi cell", {
    skip_on_cran()
    sim <- .simulate_joint_pp(N = 400, n_s = 30, sd_pos = 0.3, seed = 919)
    adj <- .chain_adj_pp(sim$n_s)
    phi_axis <- exp(seq(log(0.08), log(1.0), length.out = 9))

    U <- 1.0; a <- 0.01; lambda <- -log(a) / U
    flat <- .fit_pp(sim, adj, phi_axis, prior_phi = NULL)
    pc   <- .fit_pp(sim, adj, phi_axis, prior_phi = list("pc.prec", c(U, a)))

    expect_true("phi_pos" %in% colnames(flat$theta_grid))
    # Same grid in both fits, so the only difference in log_marginal is the
    # added prior density at each cell's phi.
    expect_identical(flat$theta_grid[, "phi_pos"], pc$theta_grid[, "phi_pos"])
    expected_add <- log(lambda) - lambda * pc$theta_grid[, "phi_pos"]
    delta <- pc$log_marginal - flat$log_marginal
    expect_equal(unname(delta), unname(expected_add), tolerance = 1e-9)
})

test_that("a sharp half-normal prior_phi shrinks the phi posterior toward zero", {
    skip_on_cran()
    sim <- .simulate_joint_pp(N = 400, n_s = 30, sd_pos = 0.3, seed = 919)
    adj <- .chain_adj_pp(sim$n_s)
    phi_axis <- exp(seq(log(0.08), log(1.0), length.out = 9))

    flat <- .fit_pp(sim, adj, phi_axis, prior_phi = NULL)
    hn   <- .fit_pp(sim, adj, phi_axis, prior_phi = list("half_normal", 0.15))
    expect_lt(hn$theta_mean[["phi_pos"]], flat$theta_mean[["phi_pos"]])
})

test_that("prior_phi recovers true dispersion when the data identifies it", {
    skip_on_cran()
    sd_true <- 0.3
    sim <- .simulate_joint_pp(N = 600, n_s = 30, sd_pos = sd_true, seed = 451)
    adj <- .chain_adj_pp(sim$n_s)
    phi_axis <- exp(seq(log(0.08), log(1.0), length.out = 9))
    # A weak PC prior with U above truth is essentially harmless when n_pos
    # identifies phi.
    fit <- .fit_pp(sim, adj, phi_axis, prior_phi = list("pc.prec", c(1.0, 0.01)))
    expect_lt(abs(fit$theta_mean[["phi_pos"]] - sd_true) / sd_true, 0.3)
})

test_that("prior_phi is a no-op when no phi_grid is declared", {
    skip_on_cran()
    sim <- .simulate_joint_pp(N = 300, n_s = 25, sd_pos = 0.3, seed = 77)
    adj <- .chain_adj_pp(sim$n_s)
    mk <- function(prior_phi) {
        arm_occ <- list(y = as.numeric(sim$occur), n_trials = rep(1L, sim$N),
                        X = sim$Xocc, spatial_idx = sim$spatial_idx,
                        re_idx = rep(0, sim$N), n_re_groups = 0L, sigma_re = 1.0,
                        family = "binomial", phi = 1.0)
        arm_pos <- list(y = sim$y_pos, n_trials = rep(1L, length(sim$y_pos)),
                        X = sim$Xpos, spatial_idx = sim$spi_pos,
                        re_idx = rep(0, length(sim$y_pos)), n_re_groups = 0L,
                        sigma_re = 1.0, family = "gaussian", phi = 0.3,
                        field_coef = list(name = "alpha", grid = 1))
        prior <- list(type = "bym2", n_spatial_units = adj$n_spatial_units,
                      adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                      n_neighbors = adj$n_neighbors, scale_factor = 1.0,
                      sigma_grid = 0.6, rho_grid = 0.7)
        tulpa_nested_laplace_joint(responses = list(occ = arm_occ, pos = arm_pos),
                                   prior = prior, prior_phi = prior_phi,
                                   control = list(diagnose_k = FALSE))
    }
    expect_equal(mk(list("pc.prec", c(1.0, 0.01)))$log_marginal,
                 mk(NULL)$log_marginal, tolerance = 1e-12)
})

test_that("prior_phi validates malformed specs", {
    expect_error(
        .joint_parse_sigma_prior(list("pc.prec", c(-1, 0.5)), "prior_phi"),
        "prior_phi")
    expect_error(
        .joint_parse_sigma_prior(list("nope", 1), "prior_phi"),
        "Unknown hyperprior")
})
