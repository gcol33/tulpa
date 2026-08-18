# Marginal fixed-effect Hessian for spatial-field Laplace fits

The raw fixed-effect block of the joint Hessian gives the *conditional*
precision on \\\beta \mid u^\*\\, which under-states uncertainty; this
returns the marginal precision instead.

## Details

The correct marginal precision comes from a Schur complement on the
joint Hessian at the mode:

\$\$H\_\beta^{\mathrm{marg}} = X'WX - X'WZ (Z'WZ + Q_u)^{-1} Z'WX\$\$

where \\Z\\ is the obs-\>latent map (SPDE: projection matrix A; NNGP:
indicator from obs to unique-location field) and \\Q_u\\ is the spatial
precision at the fitted hyperparameters.

These helpers rebuild the spatial precision in R from the spec and
fitted hyperparameters, then solve via sparse Cholesky. The shape
matches what `cpp_laplace_fit_spde` / `cpp_laplace_fit_gp` use
internally – Q construction here is a 1:1 port of `spde_qbuilder.h`
(orphan ridge included).
