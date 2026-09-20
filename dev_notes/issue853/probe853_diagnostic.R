# gcol33/tulpa#853 -- does the engine ALREADY name the configurations the
# continuization reads miscalibrated on?
#
# `probe853_resolution.R` measured where the grid route's hyperparameter PIT
# stops tracking the exact posterior. The engine carries a resolution
# diagnostic of its own (`outer_grid_h_over_sd` against `.nl_diag("grid_resolved")`,
# the railed-axis test, and `outer_grid_edge_mass_axes`), so the question is not
# what to build -- it is whether what is built fires here. This runs the SAME
# configurations and reports the rate at which the shipped diagnostic speaks.

suppressMessages(library(tulpa))
local({
  ns <- asNamespace("tulpa")
  tmp <- new.env(parent = ns)
  for (f in list.files("R", pattern = "[.][Rr]$", full.names = TRUE)) {
    try(sys.source(f, envir = tmp, keep.source = FALSE), silent = TRUE)
  }
  for (nm in ls(tmp, all.names = TRUE)) {
    v <- get(nm, envir = tmp)
    if (is.function(v)) environment(v) <- ns
    if (exists(nm, envir = ns, inherits = FALSE) &&
        bindingIsLocked(nm, ns)) unlockBinding(nm, ns)
    assign(nm, v, envir = ns)
  }
})

lpost2 <- function(l1, l2, Sy, Sz, n1, n2, p) {
  v1 <- exp(2 * l1); vz <- v1 + exp(2 * l2)
  -n1 * l1 - Sy / (2 * v1) - (n2 / 2) * log(vz) - Sz / (2 * vz) -
    0.5 * (l1 / p)^2 - 0.5 * (l2 / p)^2
}

make_fit <- function(Sy, Sz, n1, n2, p, K, span) {
  ax <- rep(list(exp(p * span * seq(-2.5, 2.5, length.out = K))), 2)
  tg <- as.matrix(expand.grid(sigma = ax[[1]], sigma_pos = ax[[2]]))
  lm <- lpost2(log(tg[, 1]), log(tg[, 2]), Sy, Sz, n1, n2, p)
  specs <- tulpa:::.joint_axis_specs_from_grid(tg)
  lq <- tulpa:::.hyper_log_quad_weights(tg, specs)
  res <- list(theta_grid = tg, log_marginal = lm, log_quad = lq,
              integration = "grid", prior = list(type = "bym2"),
              blocks = NULL, axis_offsets = NULL)
  res$weights <- tulpa:::.nl_normalise_weights_safe(lm, "outer grid",
                                                    log_quad = lq)
  res <- tulpa:::.nl_posterior_moments(res, "bym2", within = "box_uniform")
  res$within_cell_requested <- "box_uniform"
  structure(res, class = c("tulpa_nested_laplace", "list", "tulpa_fit"))
}

# Per replicate: does the SHIPPED diagnostic speak, and what does it say.
scan <- function(n1, n2, p, K, span, n_sim, seed = 853L) {
  set.seed(seed)
  t1 <- exp(stats::rnorm(n_sim, 0, p)); t2 <- exp(stats::rnorm(n_sim, 0, p))
  spoke <- railed <- edge <- coarse <- unscored <- logical(n_sim)
  hos <- numeric(n_sim)
  for (s in seq_len(n_sim)) {
    y <- stats::rnorm(n1, 0, t1[s])
    z <- stats::rnorm(n2, 0, sqrt(t1[s]^2 + t2[s]^2))
    fit <- make_fit(sum(y^2), sum(z^2), n1, n2, p, K, span)
    rs <- tulpa:::.tulpa_grid_resolution(fit)
    note <- tulpa:::.tulpa_grid_resolution_note(rs)
    spoke[s] <- !is.null(note)
    railed[s] <- length(rs$railed) > 0L
    unscored[s] <- length(rs$unscored) > 0L
    coarse[s] <- !is.na(rs$max) && rs$max > tulpa:::.nl_diag("grid_resolved")
    edge[s] <- length(fit$outer_grid_edge_mass_axes %||% character(0)) > 0L
    hos[s] <- if (is.na(rs$max)) NA_real_ else rs$max
  }
  c(spoke = mean(spoke), coarse = mean(coarse), railed = mean(railed),
    unscored = mean(unscored), edge = mean(edge),
    h_over_sd = stats::median(hos, na.rm = TRUE))
}

n_sim <- as.integer(Sys.getenv("PROBE_NSIM", "300"))
cfg <- list(
  list(tag = "res  K=3  span=1.0", K = 3L,  sp = 1.0, verdict = "MISCALIBRATED"),
  list(tag = "res  K=5  span=1.0", K = 5L,  sp = 1.0, verdict = "MISCALIBRATED"),
  list(tag = "res  K=9  span=1.0", K = 9L,  sp = 1.0, verdict = "MISCALIBRATED"),
  list(tag = "res  K=19 span=1.0", K = 19L, sp = 1.0, verdict = "borderline"),
  list(tag = "ext  K=6  span=0.3", K = 6L,  sp = 0.3, verdict = "MISCALIBRATED"),
  list(tag = "ext  K=10 span=0.5", K = 10L, sp = 0.5, verdict = "MISCALIBRATED"),
  list(tag = "ext  K=15 span=0.8", K = 15L, sp = 0.8, verdict = "tracks exact"),
  list(tag = "ext  K=38 span=2.0", K = 38L, sp = 2.0, verdict = "tracks exact")
)

cat(sprintf("%-20s %-14s %7s %7s %7s %8s %6s %9s\n", "config", "PIT verdict",
            "spoke", "coarse", "railed", "unscored", "edge", "h/sd"))
for (c0 in cfg) {
  r <- scan(400L, 400L, 0.5, c0$K, c0$sp, n_sim)
  cat(sprintf("%-20s %-14s %7.3f %7.3f %7.3f %8.3f %6.3f %9.2f\n",
              c0$tag, c0$verdict, r[["spoke"]], r[["coarse"]], r[["railed"]],
              r[["unscored"]], r[["edge"]], r[["h_over_sd"]]))
}
