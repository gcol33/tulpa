# gcol33/tulpa#760 follow-up: a free-covariance block's design is read off the
# rows handed to the fold. Rscript repro_logchol_batch.R [lib]
# Engine e66c1338: 7 rows of the default MCAR grid read 2.39 to 4.12 nats above
# the whole grid; a 2-block tensor folds no L column and measures NA.
lib <- commandArgs(trailingOnly = TRUE)[1]
if (!is.na(lib) && nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressMessages(library(tulpa)); ns <- asNamespace("tulpa")
g <- ns$.mcar_default_logchol_grid(2L); colnames(g) <- paste0("b1.", colnames(g))
bl <- list(list(type = "mcar"))
axes <- ns$.hp_integrated_axes(g)
whole <- ns$.joint_hyperprior(g, bl, axes = axes)
cat("single-block grid", nrow(g), "cells; folded:", whole$axes, "\n")
set.seed(1); draws <- sample(nrow(g), 7)
part <- ns$.joint_hyperprior(g[draws, , drop = FALSE], bl, axes = axes)
cat("7 rows of it | folded:", part$axes, "| declined:", unique(part$declined), "\n")
cat("  batch minus whole lp:", format(part$lp - whole$lp[draws], digits = 5), "\n")
tg <- cbind(g[rep(seq_len(nrow(g)), 2), ], b2.sigma = rep(c(0.5, 1), each = nrow(g)))
w2 <- ns$.joint_hyperprior(tg, list(list(type = "mcar"), list(type = "icar")), axes = ns$.hp_integrated_axes(tg))
cat("two-block tensor (mcar x icar sigma)", nrow(tg), "cells | folded:", w2$axes, "| declined:", unique(w2$declined), "\n")
sp <- ns$.joint_axis_specs_from_grid(tg, folded_axes = w2$axes)
cat("  measure all NA:", all(is.na(ns$.hyper_log_quad_weights(tg, sp, absolute = TRUE))), "\n")
