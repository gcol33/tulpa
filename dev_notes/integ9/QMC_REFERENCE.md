# Multiscale Gaussian reference

The seed-1 fixture in `test-outer-qmc-recovery.R` has 60 locations and residual
variance 0.1. Integrating its intercept and both Gaussian fields analytically
gives covariance

`V = 6.25 * 11' + 0.1 I + sigma_local^2 exp(-D/ell_local) + sigma_regional^2 exp(-D/ell_regional)`.

Each field SD has an exponential PC prior with `P(sigma > 2) = 0.05`.
The physical range is `rho = 2 ell`. Its dimension-two PC prior has
`P(rho < U) = 0.05`, with `U = 0.2` locally and `U = 1.5` regionally.
Consequently the exponential rate on inverse lengthscale is
`-log(0.05) * U / 2`.

The integration coordinates are log variance, log lengthscale, log variance,
log lengthscale. Their respective windows are `[-18,3]`, `[log(.02),log(1500)]`,
`[-18,3]`, and `[log(.1),log(30000)]`. Uniform midpoint quadrature at spacing
0.25 uses 16,581,375 cells; spacing 0.2 uses 40,988,928 cells. Fixed-effect and
field covariance calculations use a direct dense Cholesky, independently of
the package's nested solver.

The fine-grid log evidence is -83.3996422521. The coarse-grid value is
-83.3996500561. The fine-grid largest endpoint-level mass is 1.67120200439e-5.
The reported quantiles use the uniform cell edges in these log coordinates.
The unchanged test tolerances are 0.1 posterior log-coordinate SD for each
quantile and 0.015 for log evidence.

The reference with log evidence -83.63646 uses `U` on lengthscale. Commit
`0c0cf2db21a436ad3ea433e073043b73847205cb` intentionally changes the prior
to the physical-range convention. The reference values follow that model
change. The integration report retains the density-history localization,
both reference grids' marginal masses, and their refinement comparison.

From the package root, reproduce both reference grids with
`Rscript dev_notes/integ9/reproduce_reference.R <output-directory> 0.25 0.2`.
The independent dense oracle is `dev_notes/integ9/exact_multiscale.cpp`.
The driver evaluates the physical coordinates `log(rho) = log(ell) + log(2)`
and distance matrix `2 D`, preserving the covariance and log-coordinate
measure while applying the physical-range prior.
