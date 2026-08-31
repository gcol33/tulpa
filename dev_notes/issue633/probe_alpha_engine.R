## gcol33/tulpa#633, engine-side. Does the copy_alpha axis densify with the
## request, and is the saturation in the PLACEMENT or in the prune?
##
## The reported fixture is a tulpaObs occu_cover hurdle; the axis it binds on is
## the engine's, so this reproduces it through tulpa_nested_laplace_joint()
## directly -- no consumer package, and the engine can then test its own axis.
suppressMessages(pkgload::load_all(".", quiet = TRUE))

.chain_adj <- function(n) {
    rp <- integer(n + 1L); ci <- integer(0); nb <- integer(n)
    for (i in seq_len(n)) {
        nbr <- c(if (i > 1L) i - 1L, if (i < n) i + 1L)
        ci <- c(ci, nbr - 1L); nb[i] <- length(nbr); rp[i + 1L] <- length(ci)
    }
    list(n_spatial_units = n, adj_row_ptr = rp, adj_col_idx = ci, n_neighbors = nb)
}

## `reps` is the analogue of visits-per-site: more replication => sharper
## hyperparameter posterior, which is the regime the report says binds.
sim_fit <- function(n_nodes, reps, seed = 1L, prune = FALSE, alpha_grid = NULL,
                    alpha_n = NULL) {
    set.seed(seed)
    n_s <- 40L
    sigma_true <- 0.8; alpha_true <- 1.0
    f <- as.numeric(scale(cumsum(rnorm(n_s)))) * sigma_true

    N1 <- n_s * reps
    s1 <- rep(seq_len(n_s), each = reps)
    X1 <- cbind(1, rnorm(N1))
    e1 <- X1 %*% c(0.2, 0.5) + f[s1]
    y1 <- rbinom(N1, 1L, 1 / (1 + exp(-e1)))

    N2 <- n_s * reps
    s2 <- rep(seq_len(n_s), each = reps)
    X2 <- cbind(1, rnorm(N2))
    y2 <- rnorm(N2, X2 %*% c(0.1, 0.2) + alpha_true * f[s2], 0.3)

    arm1 <- list(y = as.numeric(y1), n_trials = rep(1L, N1), X = X1,
                 re_idx = rep(0, N1), n_re_groups = 0L, sigma_re = 1.0,
                 family = "binomial", phi = 1.0)
    arm2 <- list(y = y2, n_trials = rep(1L, N2), X = X2,
                 re_idx = rep(0, N2), n_re_groups = 0L, sigma_re = 1.0,
                 family = "gaussian", phi = 1.0)
    adj <- .chain_adj(n_s)
    block <- c(adj, list(type = "icar", spatial_idx = list(s1, s2),
               sigma_grid = exp(seq(log(0.1), log(3), length.out = n_nodes))))

    suppressWarnings(tulpa_nested_laplace_joint(
        responses = list(occ = arm1, pos = arm2), prior = list(block),
        copy = list(list(arm = "pos", block = 1L, alpha_grid = alpha_grid,
                         alpha_n = alpha_n)),
        control = list(diagnose_k = FALSE, prune = prune)))
}

report <- function(tag, fit, req) {
    tg <- fit$theta_grid
    w  <- fit$weights
    d  <- vapply(colnames(tg), function(a) length(unique(tg[, a])), integer(1))
    cat(sprintf("%-10s req=%2d cells=%5d ess=%6.1f  %s   place=%s\n",
                tag, req, nrow(tg), 1 / sum(w^2),
                paste(sprintf("%s=%d", names(d), d), collapse = " "),
                fit$outer_grid_placement %||% "-"))
    invisible(d)
}

cat("### declared axis\n"); print(.nl_grid_axis("copy_alpha"))

for (reps in c(3L, 30L)) {
    cat(sprintf("\n### reps = %d  (prune = FALSE, the engine default)\n", reps))
    for (k in c(13L, 21L, 29L)) report(sprintf("reps%d", reps), sim_fit(k, reps), k)
}

cat("\n### reps = 30, prune = TRUE  (is the saturation the prune?)\n")
for (k in c(13L, 21L, 29L)) report("prune", sim_fit(k, 30L, prune = TRUE), k)

cat("\n### reps = 30, alpha_grid supplied explicitly at the request\n")
for (k in c(13L, 21L, 29L))
    report("explicit", sim_fit(k, 30L, alpha_grid = seq(0, 3, length.out = k)), k)

cat("\n### reps = 30, alpha_n = the request (the fix: resolution, not nodes)\n")
for (k in c(13L, 21L, 29L)) report("alpha_n", sim_fit(k, 30L, alpha_n = k), k)

cat("\n### the declared shape is preserved at any resolution\n")
for (k in c(5L, 12L, 28L)) {
    ax <- .nl_grid_axis("copy_alpha", n = k)
    cat(sprintf("n=%2d -> %2d nodes, atom=%g, slab [%g, %g]\n",
                k, length(ax), ax[1], min(ax[-1]), max(ax)))
}

cat("\n### both at once is refused\n")
print(tryCatch(sim_fit(13L, 30L, alpha_grid = c(0, 1), alpha_n = 9L),
               error = function(e) conditionMessage(e)))
