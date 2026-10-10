# gcol33/tulpa#951: a non-copy block's scale axis (an iid RE block's
# b<k>.sigma, an arm-specific ICAR block's b<k>.tau) is never placed, and an
# axis that holds one node in the fit's kept cells (adaptive lattice, pilot
# screen) is frozen out of the placement mode-find. On data where those axes
# are sharply identified the grid then integrates them at one node: posterior
# SD 0, the value a grid node.
#
# Fixture: two arms (binomial occurrence, gaussian positive part) sharing a
# copied ICAR field (block 1), plus
#   block 2  an arm-specific ICAR field on the positive arm alone (tau axis),
#   block 3  an iid random intercept on the positive arm (sigma axis),
# both with marked (auto_grid) default axes, integrated with
# integration = "grid_adaptive" as the occu_cover server scripts do.
#
# Run:  Rscript dev_notes/issue951/repro.R [lib]
args <- commandArgs(TRUE)
if (length(args) >= 1L && nzchar(args[1])) .libPaths(c(args[1], .libPaths()))
suppressPackageStartupMessages(library(tulpa))
cat("tulpa", as.character(packageVersion("tulpa")), "from", find.package("tulpa"), "\n")

chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

sim <- function(seed, N = 6000L, n_s = 60L, n_g = 12L, sd_re = 0.8,
                sd_pf = 0.7) {
    set.seed(seed)
    sp <- sample.int(n_s, N, replace = TRUE)
    w_s  <- 0.9 * as.numeric(scale(cumsum(rnorm(n_s, sd = 0.6))))
    w_pf <- sd_pf * as.numeric(scale(cumsum(rnorm(n_s, sd = 0.6))))
    g <- sample.int(n_g, N, replace = TRUE)
    u <- rnorm(n_g, sd = sd_re)
    x <- rnorm(N); X <- cbind(1, x)
    occ <- rbinom(N, 1, plogis(as.numeric(X %*% c(-0.2, 0.5)) + w_s[sp]))
    k <- occ == 1L
    y <- rnorm(sum(k), as.numeric(X[k, ] %*% c(0.2, -0.4)) + 0.6 * w_s[sp[k]] +
                   w_pf[sp[k]] + u[g[k]], 0.3)
    list(adj = chain_adj(n_s), sp = sp, k = k, g = g, n_g = n_g,
         responses = list(
             occ = list(y = as.numeric(occ), n_trials = rep(1L, N), X = X,
                        spatial_idx = sp, re_idx = rep(0, N), n_re_groups = 0L,
                        sigma_re = 1, family = "binomial", phi = 1),
             pos = list(y = y, n_trials = rep(1L, sum(k)), X = X[k, ],
                        spatial_idx = sp[k], re_idx = rep(0, sum(k)),
                        n_re_groups = 0L, sigma_re = 1, family = "gaussian",
                        phi = 0.09)))
}

fit_one <- function(s, control = list()) {
    a <- s$adj
    icar <- function(extra) c(list(type = "icar", n_spatial_units = a$n_spatial_units,
                                   adj_row_ptr = a$adj_row_ptr,
                                   adj_col_idx = a$adj_col_idx,
                                   n_neighbors = a$n_neighbors), extra)
    n_occ <- length(s$responses$occ$y); n_pos <- length(s$responses$pos$y)
    prior <- list(
        icar(list(sigma_grid = auto_grid(c(0.25, 0.5, 1, 2, 4)),
                  spatial_idx = list(s$sp, s$sp[s$k]))),
        icar(list(tau_grid = auto_grid(1 / exp(seq(log(0.05), log(5),
                                                   length.out = 9))^2),
                  spatial_idx = list(rep(0L, n_occ), s$sp[s$k]))),
        list(type = "iid", n_units = s$n_g,
             sigma_grid = auto_grid(c(0.2, 0.6, 1.5)),
             obs_idx = list(rep(0L, n_occ), s$g[s$k])))
    tulpa_nested_laplace_joint(
        s$responses, prior,
        copy = list(arm = "pos", block = 1L,
                    alpha_grid = auto_grid(c(0, 0.3, 0.6, 1.2))),
        control = utils::modifyList(list(diagnose_k = FALSE,
                                         integration = "grid_adaptive"),
                                    control))
}

report <- function(fit, label) {
    tg <- fit$theta_grid; w <- fit$weights
    cat(sprintf("\n== %s: %d cells, integration %s, placement %s\n", label,
                nrow(tg), fit$integration, fit$outer_grid_placement))
    for (a in colnames(tg)) {
        agg <- tapply(w, tg[, a], sum)
        cat(sprintf("  %-9s nodes %2d  mean %.4g  sd %.4g  max node weight %.3f\n",
                    a, length(agg), fit$theta_mean[[a]], fit$theta_sd[[a]],
                    max(agg)))
    }
    cat("  recentred axes:", paste(fit$outer_grid_recenter_axes, collapse = ","), "\n")
    d <- fit$outer_grid_axis_declined
    if (length(d)) cat("  declined:", paste(names(d), d, sep = "=", collapse = " "), "\n")
    cat("  mode-find axes varied:",
        paste(fit$outer_mode_axis_names[diag(fit$outer_mode_cov_u %||%
                                              matrix(0, 0, 0)) > 0],
              collapse = ","), "\n")
}
`%||%` <- function(x, y) if (is.null(x)) y else x

s <- sim(951L)
t0 <- Sys.time()
f1 <- fit_one(s)
report(f1, sprintf("grid_adaptive (%.0fs)", as.numeric(Sys.time() - t0, units = "secs")))
t0 <- Sys.time()
f2 <- fit_one(s, list(recenter_pilot = TRUE, prune = TRUE))
report(f2, sprintf("grid_adaptive + pilot + prune (%.0fs)",
                   as.numeric(Sys.time() - t0, units = "secs")))
cat("\ntruth: pos-field sd 0.7 (tau ~", round(1 / 0.7^2, 2), "), RE sd 0.8\n")
