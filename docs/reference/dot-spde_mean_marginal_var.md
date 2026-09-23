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

## Details

Returns `list(value, singular)`. `singular` is TRUE where the
factorization could not be completed – `Q` is not merely ill-conditioned
but numerically SINGULAR, so `Q^-1` has no value and neither does the
trace this estimates. CHOLMOD reports that as a "not positive definite"
warning and then recovers on its own terms, which is a decision about
the model taken by a library (gcol33/tulpa#845): the warning is caught
here and turned into a flag the caller acts on, instead of reaching the
user as a raw message from a probe they did not ask for.

It happens far outside the range a fit integrates. Measured on the
issue's n = 120, nu = 1.5 fixture: `rcond(Q)` is already 2.4e-17 at
range 40 on a domain of extent 13.8, and every warning came from the
MODE SEARCH at its own box corner, `range = 100 * range_init ~ 275.6`.
The CCD grid the fit then integrates topped out at range 1.53 and
carried no weight above 100.
