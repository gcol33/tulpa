# Mean marginal variance of the rSPDE field u = Pr x, x ~ N(0, Q^-1)

Estimates `mean_i [Pr Q^{-1} Pr']_ii = tr(Pr Q^{-1} Pr') / n` by
Hutchinson probing: for `z ~ N(0, I)`, `a = Pr' z`, `Q v = a`, then
`E[a' v] = tr(...)`. The solve is against `Q` through the SAME sparse
Cholesky the precomputed C++ fit uses, so the normalization is
consistent with the fit even when the wide rational spectrum makes `Q`
ill-conditioned: a shared solver makes the implied field covariance
identical between the normalization and the likelihood, which is what
the nested `(range, sigma)` integration needs (an inconsistent solver
breaks the cross-grid marginal). The probe matrix is fixed across calls
for a deterministic, grid-smooth normalization.

## Usage

``` r
.spde_mean_marginal_var(Q, Pr, C0, n_probe = .SPDE_VARNORM_NPROBE)
```
