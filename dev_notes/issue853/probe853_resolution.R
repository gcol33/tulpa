# gcol33/tulpa#853 -- WHERE the outer grid stops resolving its own
# hyperparameter, measured, so the diagnostic's thresholds are read off a curve
# rather than chosen.
#
# Two independent ways a grid fails its axis, swept one at a time on the same
# exact-posterior fixture as `probe853.R`:
#
#   RESOLUTION -- cell width against the posterior's own sd, at constant grid
#     extent (sweep the node count). A posterior narrower than a cell is
#     reported by the read as the whole cell, whatever within-cell density the
#     cell carries.
#   EXTENT -- posterior mass sitting in the outermost cell, at constant
#     resolution (sweep the grid span). Mass past the outermost node is mass no
#     draw can place, and it pins the PIT at 0 or 1.

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

exact_ref <- function(Sy, Sz, n1, n2, p, m = 301L, span = 8) {
  g <- seq(-span, span, length.out = m)
  G <- expand.grid(l1 = g, l2 = g)
  lp <- lpost2(G$l1, G$l2, Sy, Sz, n1, n2, p)
  w <- exp(lp - max(lp))
  list(s1 = exp(G$l1), s2 = exp(G$l2), w = w / sum(w))
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

# The two statistics the diagnostic will report. The grid's own weighted SD is
# NOT one of them: it is the quantity that collapses toward zero exactly when
# the grid stops resolving the axis, so sd-over-cell-width saturates and cannot
# tell "one cell holds everything" from "the posterior is narrow". What the
# grid can say about itself is how many of its cells the posterior actually
# uses -- the marginal weights' effective count -- and how much of the mass
# sits in the outermost cell, which is mass no draw can place beyond.
axis_stats <- function(fit, j) {
  tg <- fit$theta_grid
  w <- fit$weights; w <- w / sum(w)
  uv <- sort(unique(as.numeric(tg[, j])))
  if (length(uv) < 2L) return(c(ess = NA_real_, edge = NA_real_))
  mw <- vapply(uv, function(v) sum(w[tg[, j] == v]), numeric(1))
  s <- sum(mw)
  if (!is.finite(s) || s <= 0) return(c(ess = NA_real_, edge = NA_real_))
  mw <- mw / s
  c(ess = 1 / sum(mw^2), edge = mw[1L] + mw[length(mw)])
}

run <- function(n1, n2, p, K, span, n_sim, n_draw = 3000L, seed = 853L) {
  set.seed(seed)
  t1 <- exp(stats::rnorm(n_sim, 0, p)); t2 <- exp(stats::rnorm(n_sim, 0, p))
  box <- ex <- matrix(NA_real_, n_sim, 3)
  st <- matrix(NA_real_, n_sim, 2)
  for (s in seq_len(n_sim)) {
    y <- stats::rnorm(n1, 0, t1[s])
    z <- stats::rnorm(n2, 0, sqrt(t1[s]^2 + t2[s]^2))
    Sy <- sum(y^2); Sz <- sum(z^2)
    fit <- make_fit(Sy, Sz, n1, n2, p, K, span)
    cells <- tulpa:::.nl_mixture_cells(fit$weights, seq_along(fit$weights),
                                       n_draw)$row_cells
    B <- tulpa_hyper_draws(fit, cells = cells)
    ref <- exact_ref(Sy, Sz, n1, n2, p)
    tv <- c(t1[s], t2[s], t2[s] / t1[s])
    box[s, ] <- c(mean(B[, 1] <= tv[1]), mean(B[, 2] <= tv[2]),
                  mean(B[, 2] / B[, 1] <= tv[3]))
    ex[s, ] <- c(sum(ref$w[ref$s1 <= tv[1]]), sum(ref$w[ref$s2 <= tv[2]]),
                 sum(ref$w[ref$s2 / ref$s1 <= tv[3]]))
    st[s, ] <- axis_stats(fit, 1L)
  }
  ks <- function(v) suppressWarnings(stats::ks.test(v, "punif"))
  list(res = mean(st[, 1], na.rm = TRUE),
       edge = mean(st[, 2], na.rm = TRUE),
       box_ks = ks(box[, 1])$statistic, box_p = ks(box[, 1])$p.value,
       ex_ks = ks(ex[, 1])$statistic, ex_p = ks(ex[, 1])$p.value,
       al_ks = ks(box[, 3])$statistic, al_p = ks(box[, 3])$p.value,
       pinned = mean(box[, 1] <= 1e-3 | box[, 1] >= 1 - 1e-3))
}

n_sim <- as.integer(Sys.getenv("PROBE_NSIM", "300"))
hdr <- function(t) {
  cat("\n== ", t, "\n", sep = "")
  cat(sprintf("%5s %6s %8s %7s %8s %8s %8s %8s %8s\n", "K", "span",
              "ess", "edge", "boxKS", "boxP", "exKS", "alphaKS", "pinned"))
}
row <- function(K, span, r) {
  cat(sprintf("%5d %6.2f %8.2f %7.3f %8.4f %8.4f %8.4f %8.4f %8.3f\n",
              K, span, r$res, r$edge, r$box_ks, r$box_p, r$ex_ks,
              r$al_ks, r$pinned))
}

out <- list()
hdr("RESOLUTION sweep: node count at constant extent (span 1.0), n=400")
for (K in c(3L, 5L, 7L, 9L, 13L, 19L)) {
  r <- run(400L, 400L, 0.5, K, 1.0, n_sim); row(K, 1.0, r)
  out[[paste0("res_K", K)]] <- c(K = K, span = 1, unlist(r))
}

hdr("EXTENT sweep: grid span at constant cell width (K scaled with span), n=400")
for (sp in c(0.3, 0.5, 0.8, 1.2, 2.0)) {
  K <- max(3L, as.integer(round(19 * sp)))
  r <- run(400L, 400L, 0.5, K, sp, n_sim); row(K, sp, r)
  out[[paste0("ext_", sp)]] <- c(K = K, span = sp, unlist(r))
}

# Where the edge mass stops mattering: the crossing the flag is set from, at
# the same cell width, so the only thing moving is how much mass the outermost
# cell holds.
hdr("EXTENT fine sweep: locating the edge-mass crossing, n=400")
for (sp in c(0.55, 0.62, 0.68, 0.74, 0.90)) {
  K <- max(3L, as.integer(round(19 * sp)))
  r <- run(400L, 400L, 0.5, K, sp, n_sim); row(K, sp, r)
  out[[paste0("fine_", sp)]] <- c(K = K, span = sp, unlist(r))
}

saveRDS(out, "dev_notes/issue853/resolution_curve.rds")
