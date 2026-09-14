# gcol33/tulpa#328 item 1, at the fit level: the same four-axis joint model
# refined under the two local-CCD selection rankings, both read against a
# finer-grid reference through the #322 outer-grid harness.
#
# The joint multi-block driver caps an outer tensor at 2048 cells, so in four
# axes the finest reference available is six levels (1296 cells). The base grid
# is four levels (256), and the five-level read (625) is carried beside it as
# the scale one step of plain grid refinement moves the answer on.
#
# The two rankings are the engine's own `control$local_ccd$rank`: `"weight"`
# (the cell's integration weight, the default) and `"mass_moved"` (the weight
# times `|exp(log_box_ratio) - 1|`). Each is a separate fit of the installed
# package; nothing in its namespace is replaced.
#
#   Rscript measure_fit_ranking.R <lib> <repo> <hyperprior> <within_cell> <out.csv> [seeds=1:8]

args <- commandArgs(trailingOnly = TRUE)
lib <- args[1L]; repo <- args[2L]; hyperprior <- args[3L]
within_cell <- args[4L]; out <- args[5L]
seeds <- if (length(args) >= 6L) eval(parse(text = args[6L])) else 1:8
stopifnot(hyperprior %in% c("flat", "proper"))

source(file.path(repo, "dev_notes", "harness", "load_harness.R"))
env <- harness_load(lib, repo)
id <- harness_identity(lib, hyperprior, within_cell)
harness_print_identity(id)

sim <- function(seed, sd_true = c(0.8, 0.5, 0.35, 0.25), N = 900L, G = 40L,
                phi = 0.25) {
  set.seed(seed)
  grp <- lapply(seq_along(sd_true), function(k) sample.int(G, N, replace = TRUE))
  X <- cbind(1, stats::rnorm(N))
  eta <- as.numeric(X %*% c(0.2, 0.6))
  for (k in seq_along(sd_true)) eta <- eta + stats::rnorm(G, 0, sd_true[k])[grp[[k]]]
  list(y = eta + stats::rnorm(N, 0, tulpa:::.phi_to_kernel("gaussian", phi)), X = X,
       grp = grp, N = N, G = G, phi = phi,
       sd_true = sd_true)
}

# The residual variance is the one the data were simulated at (gcol33/tulpa#744).
fit <- function(s, levels, local_ccd = NULL, spread = 3) {
  prior <- lapply(seq_along(s$grp), function(k) {
    sd <- s$sd_true[k]
    list(type = "iid", obs_idx = list(s$grp[[k]]), n_units = s$G,
         sigma_grid = exp(seq(log(sd / spread), log(sd * spread),
                              length.out = levels)))
  })
  suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = s$y, n_trials = rep(1L, s$N), X = s$X,
                              family = "gaussian", phi = s$phi)),
    prior = prior, hyperprior = hyperprior,
    control = list(n_threads = 1L, diagnose_k = FALSE, max_iter = 100L,
                   tol = 1e-8, integration = "grid", progress = FALSE,
                   var_of_means_consistency = FALSE, local_ccd = local_ccd,
                   within_cell = within_cell)))
}

acc <- list()
with(env, for (seed in seeds) {
  s <- sim(seed)
  t0 <- Sys.time()
  ref <- outer_grid_rebuild(outer_grid_dump(fit(s, 6L)))
  d5 <- outer_grid_dump(fit(s, 5L))
  d0 <- outer_grid_dump(fit(s, 4L))
  fl <- outer_grid_noise_floor(d0)
  e_base <- outer_grid_read_diff(ref, outer_grid_rebuild(d0))
  e_l5 <- outer_grid_read_diff(ref, outer_grid_rebuild(d5))
  for (mc in c(2L, 4L, 8L)) {
    lc <- function(rank) list(max_cells = mc, skew_max = Inf, rank = rank)
    dw <- outer_grid_dump(fit(s, 4L, lc("weight")))
    dm <- outer_grid_dump(fit(s, 4L, lc("mass_moved")))
    e_w <- outer_grid_read_diff(ref, outer_grid_rebuild(dw))
    e_m <- outer_grid_read_diff(ref, outer_grid_rebuild(dm))
    acc[[length(acc) + 1L]] <<- data.frame(
      seed = seed, max_cells = mc,
      same_grid = isTRUE(all.equal(dw$joint_grid, dm$joint_grid)),
      ep_base = e_base$endpoints, ep_l5 = e_l5$endpoints,
      ep_weight = e_w$endpoints, ep_move = e_m$endpoints,
      wd_base = e_base$widths, wd_l5 = e_l5$widths,
      wd_weight = e_w$widths, wd_move = e_m$widths,
      md_base = e_base$median, md_l5 = e_l5$median,
      md_weight = e_w$median, md_move = e_m$median,
      fl_ep = fl$endpoints, fl_wd = fl$widths, fl_md = fl$median,
      hyperprior = hyperprior, within_cell = id$within_cell,
      build = harness_build_tag(id))
  }
  cat(sprintf("seed %d done, %.0fs\n", seed,
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
})

s <- do.call(rbind, acc)
utils::write.csv(s, out, row.names = FALSE)
cat("\n=== totals over", nrow(s), "configurations (", length(unique(s$seed)),
    "seeds x 3 budgets ), hyperprior", hyperprior, "===\n")
for (p in c("ep", "wd", "md")) {
  e_w <- s[[paste0(p, "_weight")]]; e_m <- s[[paste0(p, "_move")]]
  fl <- s[[paste0("fl_", p)]]
  cat(sprintf("%-3s base %7.4f  5-level %7.4f  weight %7.4f  mass_moved %7.4f  floor %7.4f | arm-to-arm difference above floor in %d of %d, mass_moved nearer in %d\n",
              p, sum(s[[paste0(p, "_base")]]), sum(s[[paste0(p, "_l5")]]),
              sum(e_w), sum(e_m), sum(fl), sum(abs(e_w - e_m) > fl), nrow(s),
              sum(e_m < e_w)))
}
for (mc in sort(unique(s$max_cells))) {
  z <- s[s$max_cells == mc, ]
  cat(sprintf(" max_cells %d: identical grids %d/%d\n", mc, sum(z$same_grid), nrow(z)))
}
writeLines(format(Sys.time()), paste0(out, ".done"))
