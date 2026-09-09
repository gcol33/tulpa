# Shared fixture for the outer-grid dispersion-axis placement passes
# (gcol33/tulpa#663): a joint occurrence + gaussian-cover pair whose dispersion
# posterior sits between the nodes of a deliberately coarse declared axis, so a
# marked (`auto_grid()`) axis has something to be placed onto. Used by
# test-phi-grid-placement.R and by the checkpoint file, which needs a fit that
# actually triggers a placement rescue.

.pgp_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

.pgp_fixture <- function(N = 3000, n_s = 40, sigma = 0.6, rho = 0.7,
                         sd_pos = 0.3, seed = 60101) {
    set.seed(seed)
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    w_s <- sigma * (sqrt(rho) * rnorm(n_s) + sqrt(1 - rho) * rnorm(n_s))
    Xocc <- cbind(1, rnorm(N))
    occur <- rbinom(N, 1, plogis(as.numeric(Xocc %*% c(-0.3, 0.5)) +
                                 w_s[spatial_idx]))
    is_pos <- occur == 1L
    Xpos <- Xocc[is_pos, , drop = FALSE]
    spi <- spatial_idx[is_pos]
    y_pos <- as.numeric(Xpos %*% c(0.2, -0.4)) + w_s[spi] +
        rnorm(sum(is_pos), 0, sd_pos)
    adj <- .pgp_chain_adj(n_s)
    list(
        truth_phi = sd_pos^2,
        responses = list(
            occ = list(y = as.numeric(occur), n_trials = rep(1L, N), X = Xocc,
                       spatial_idx = spatial_idx, re_idx = rep(0, N),
                       n_re_groups = 0L, sigma_re = 1.0,
                       family = "binomial", phi = 1.0),
            pos = list(y = y_pos, n_trials = rep(1L, length(y_pos)), X = Xpos,
                       spatial_idx = spi, re_idx = rep(0, length(y_pos)),
                       n_re_groups = 0L, sigma_re = 1.0,
                       family = "gaussian", phi = 1.0,
                       field_coef = list(name = "alpha", grid = 1))),
        prior = list(type = "bym2", n_spatial_units = adj$n_spatial_units,
                     adj_row_ptr = adj$adj_row_ptr,
                     adj_col_idx = adj$adj_col_idx,
                     n_neighbors = adj$n_neighbors, scale_factor = 1.0,
                     sigma_grid = sigma, rho_grid = rho))
}

# Four nodes over residual SD 0.02 to 2 -- the shape the reported case had: a
# span wide enough that no node is within a factor of two of the posterior.
.PGP_COARSE <- exp(seq(log(0.02), log(2), length.out = 4))^2
