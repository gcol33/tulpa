# The two-arm ICAR joint fixture behind the joint posterior-draw tests
# (`test-posterior-draws-joint.R`) and the hyperparameter continuization
# (`test-hyper-draws.R`). Shared rather than copied: the second file needs the
# SAME occu_cover-shaped fit -- a binomial occupancy arm and a gaussian cover
# arm carrying a copy coefficient -- and two fixtures drifting apart is how one
# of them stops testing the shape it was written for.

# Build a small two-arm ICAR joint fit (binomial occupancy + gaussian cover,
# copy coefficient on the cover arm). store_Q is on so the sampler has the
# per-grid precision. `sigma_grid` length controls the outer-grid size.
build_icar_joint_fit <- function(nr = 4L, nc = 4L, sigma_grid = c(0.5, 1.0),
                                 alpha_grid = c(0.4, 0.8), seed = 11L) {
    set.seed(seed)
    adj_list <- lapply(grid_neighbours(nr, nc), sort)
    n_s <- length(adj_list)
    n_neighbors <- vapply(adj_list, length, integer(1))
    adj_row_ptr <- c(0L, cumsum(n_neighbors))
    adj_col_idx <- unlist(adj_list) - 1L

    rr <- ((seq_len(n_s) - 1L) %/% nc) + 1L
    cc <- ((seq_len(n_s) - 1L) %% nc) + 1L
    f_true <- scale(sin(rr / 2) + cos(cc / 3))[, 1] * 0.8

    X1 <- cbind(1, rnorm(n_s))
    X2 <- cbind(1, rnorm(n_s))
    eta1 <- as.numeric(X1 %*% c(0.2, 0.5)) + f_true
    y1 <- rbinom(n_s, 1, plogis(eta1))
    eta2 <- as.numeric(X2 %*% c(-0.3, 0.8)) + 0.6 * f_true
    y2 <- rnorm(n_s, eta2, 0.5)

    responses <- list(
        occ = list(y = y1, n_trials = rep(1L, n_s), X = X1,
                   spatial_idx = seq_len(n_s), family = "binomial"),
        cover = list(y = y2, n_trials = rep(1L, n_s), X = X2,
                     spatial_idx = seq_len(n_s), family = "gaussian",
                     field_coef = list(name = "alpha", grid = alpha_grid))
    )
    prior <- list(type = "icar", n_spatial_units = n_s,
                  adj_row_ptr = adj_row_ptr, adj_col_idx = adj_col_idx,
                  n_neighbors = n_neighbors, sigma_grid = sigma_grid)

    fit <- tulpa_nested_laplace_joint(
        responses = responses, prior = prior,
        control = list(store_Q = TRUE, diagnose_k = FALSE)
    )
    fit$.n_s <- n_s
    fit
}

