# gcol33/tulpa#622's own repro, run against the labels as they now stand.
suppressMessages(devtools::load_all(".", quiet = TRUE))

nodes <- exp(seq(log(0.15), log(2.0), length.out = 9))

probe <- function(w, label) {
  w  <- w / sum(w)
  tg <- matrix(nodes, ncol = 1L, dimnames = list(NULL, "sigma"))
  res <- list(theta_grid = tg, log_marginal = log(w), weights = w)
  rail <- .nl_axis_rail(res, "sigma")
  rg   <- .joint_pareto_grid_regime(res)
  cat(sprintf("%-28s ceiling %5.1f%%  mode %d/9  ess %4.1f  rail %-5s  railed %-12s  edge_mass %-12s regime %s\n",
              label, 100 * w[9], which.max(w), rg$ess_grid,
              if (is.null(rail)) "NULL" else rail$side,
              if (length(.nl_railed_axes(res))) paste(.nl_railed_axes(res), collapse = ",") else "<empty>",
              if (length(.nl_edge_mass_axes(res))) paste(.nl_edge_mass_axes(res), collapse = ",") else "<empty>",
              rg$regime))
}

probe(c(0.0181,0.0444,0.1534,0.0655,0.1242,0.1186,0.1160,0.2907,0.0691), "1. observed fit")
probe(c(0.010,0.020,0.050,0.060,0.080,0.100,0.120,0.360,0.200),          "2. 20% on the ceiling")
probe(c(0.005,0.010,0.020,0.030,0.050,0.070,0.130,0.345,0.340),          "3. 34% on the ceiling")
probe(c(0.005,0.010,0.020,0.030,0.050,0.070,0.130,0.245,0.440),          "4. ceiling is mode")
probe(c(0.001,0.002,0.004,0.008,0.015,0.020,0.030,0.720,0.200),          "5. ess < 2, interior MAP")
probe(exp(-0.5 * ((seq_len(9) - 5) / 1.2)^2),                            "6. resolved, no truncation")
