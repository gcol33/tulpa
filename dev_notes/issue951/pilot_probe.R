# The pilot + prune variant of the issue951 fixture: what the placement pass
# read and laid on each axis.
args <- commandArgs(TRUE)
if (length(args) >= 1L && nzchar(args[1])) .libPaths(c(args[1], .libPaths()))
suppressPackageStartupMessages(library(tulpa))
env <- new.env(); env$test_that <- function(...) invisible(NULL)
sys.source(file.path("tests", "testthat", "test-joint-placement-block-scale.R"),
           envir = env, toplevel.env = env,
           keep.source = FALSE)
fx <- env$.bs_fixture()
fit <- tulpa_nested_laplace_joint(
    fx$responses, fx$prior, copy = fx$copy,
    control = list(diagnose_k = FALSE, integration = "grid_adaptive",
                   recenter_pilot = TRUE, prune = TRUE))
tg <- fit$theta_grid
for (a in colnames(tg)) {
    agg <- tapply(fit$weights, tg[, a], sum)
    cat(sprintf("%-9s %s\n", a, paste(sprintf("%.4g:%.3f", as.numeric(names(agg)), agg),
                                      collapse = " ")))
}
str(fit[c("outer_grid_recenter_axes", "outer_grid_axis_declined",
          "outer_mode_status", "outer_mode_axis_names", "outer_mode_u",
          "outer_grid_recenter_sd_used", "outer_grid_recenter_sd_raw",
          "outer_grid_recenter_sd_clamp", "outer_grid_recenter_attempts",
          "outer_grid_pilot", "integration", "outer_grid_single_node_axes",
          "var_of_means_consistency_info", "adaptive_grid_info")])
print(sqrt(diag(fit$outer_mode_cov_u)))
