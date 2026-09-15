# gcol33/tulpa#760: the registry placement stencil and the one-row probe against
# the grid cell they sit on, through the closure tulpa_nested_laplace() hands
# .nl_registry_grid_rescue(). Rscript repro_registry_stencil_probe.R [lib]
# Before the fix (tulpa 6779fda9), sigma held at 0.8: stencil centre -1.022622;
# rho held at 0.5: one-row +1.014204. After: 0 on all four lines.
lib <- commandArgs(trailingOnly = TRUE)[1]
if (!is.na(lib) && nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressMessages(library(tulpa))
cat("tulpa", as.character(packageVersion("tulpa")), "from", find.package("tulpa"), "\n")
ns <- asNamespace("tulpa")

chain_adj <- function(n_s) {
  nbr <- lapply(seq_len(n_s), function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  nn <- vapply(nbr, length, integer(1))
  list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L, n_neighbors = as.integer(nn))
}
S <- 40L
set.seed(11)
eff <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4)), scale = FALSE)) + rnorm(S, 0, 0.2)
idx <- rep(seq_len(S), each = 6L)
X <- cbind(1, rnorm(length(idx)))
y <- as.numeric(X %*% c(-0.2, 0.7)) + eff[idx] + rnorm(length(idx), 0, sqrt(0.5))

sg <- ns$.nl_grid_axis("field_sd")
blk <- c(list(type = "bym2", n_spatial_units = S, spatial_idx = idx, scale_factor = 1,
              sigma_grid = sg, rho_grid = rep(0.5, length(sg))), chain_adj(S))

grab <- new.env()
for (fn in c(".nl_registry_grid_rescue")) {
  trace(fn, where = ns, print = FALSE,
        tracer = bquote(assign(.(if (TRUE) "rlm"), refit_log_marginal, envir = .(grab))))
}
ctrl <- list(max_iter = 200L, tol = 1e-9, n_threads = 1L, diagnose_k = FALSE,
             diagnose_skew = FALSE, auto_recenter = FALSE)
run <- function(prior) {
  rm(list = ls(grab), envir = grab)
  fit <- suppressWarnings(tulpa_nested_laplace(
    y = y, n_trials = rep(1L, length(y)), X = X, prior = prior,
    family = "gaussian", phi = 0.5, control = ctrl))
  list(fit = fit, rlm = grab$rlm)
}

report <- function(label, fit, rlm, prior_i) {
  tg <- fit$theta_grid
  cn <- colnames(tg)
  m  <- which.max(fit$weights)
  whole <- rlm(prior_i, tg)
  cat(sprintf("\n[%s] grid %d x %d, integrated axes: %s, folded: %s\n", label,
              nrow(tg), ncol(tg), paste(ns$.hp_integrated_axes(tg), collapse = ","),
              paste(fit$log_hyperprior_axes, collapse = ",")))
  cat(sprintf("  whole-grid re-evaluation vs stored, max |diff| = %.3g\n",
              max(abs(whole - fit$log_marginal))))
  tags <- ns$.nl_registry_axis_tags(fit, if (is.null(prior_i$type)) prior_i else list(prior_i),
                                    is.null(prior_i$type))
  u0 <- vapply(seq_along(cn), function(j) ns$.joint_pareto_fwd(tags[j], tg[m, j]), 0)
  # The placement stencil (axial + corner rows about the modal cell), one batch.
  U <- do.call(rbind, c(ns$.ccd_axial_rows(u0, rep(0.1, length(u0)))$rows,
                        ns$.ccd_corner_rows(u0, rep(0.1, length(u0)))$rows))
  TM <- sapply(seq_along(cn), function(j) ns$.joint_pareto_inv(tags[j], U[, j])$theta)
  TM <- matrix(TM, ncol = length(cn), dimnames = list(NULL, cn))
  st <- rlm(prior_i, TM)
  centre <- which(apply(abs(sweep(TM, 2, tg[m, ])), 1, max) < 1e-12)[1]
  cat(sprintf("  stencil centre row (= modal grid cell) minus stored cell: %+.6f\n",
              st[centre] - fit$log_marginal[m]))
  one <- rlm(prior_i, tg[m, , drop = FALSE])
  cat(sprintf("  one-row evaluation at the modal cell minus stored cell: %+.6f\n",
              one - fit$log_marginal[m]))
  mc <- ns$.nl_registry_axis_mode_cov(fit, tags, function(tm) rlm(prior_i, tm))
  cat("  stencil covariance (as evaluated):\n"); print(signif(mc$cov, 5))
  invisible(list(st = st, TM = TM, centre = centre, mc = mc))
}

a <- run(blk)
ra <- report("single-block bym2, rho fixed at 0.5", a$fit, a$rlm, a$fit$prior %||% blk)
cat("  log density of rho at 0.5 under the default prior:",
    format(ns$.hp_axis_prior("rho", list(type = "bym2"), hyperprior = "proper")$fn(0.5)), "\n")

b <- run(list(blk))
rb <- report("multi-block [bym2], rho fixed at 0.5", b$fit, b$rlm, b$fit$prior %||% list(blk))

rg <- c(0.2, 0.4, 0.6, 0.8, 0.95)
blk_s <- c(list(type = "bym2", n_spatial_units = S, spatial_idx = idx, scale_factor = 1,
                sigma_grid = rep(0.8, length(rg)), rho_grid = rg), chain_adj(S))
cc <- run(blk_s)
rc <- report("single-block bym2, sigma fixed at 0.8", cc$fit, cc$rlm, cc$fit$prior %||% blk_s)
cat("  log density of sigma at 0.8 under the default prior:",
    format(ns$.hp_axis_prior("sigma", list(type = "bym2"), hyperprior = "proper")$fn(0.8)), "\n")
dd <- run(list(blk_s))
rd <- report("multi-block [bym2], sigma fixed at 0.8", dd$fit, dd$rlm, dd$fit$prior %||% list(blk_s))
