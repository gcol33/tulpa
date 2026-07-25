# Exact Hessian of the joint-field Laplace marginal w.r.t. the RE covariance

Target: `d^2 m(theta) / d theta d theta'` for the EB / nested outer objective, in
closed form (no finite differences), building directly on the exact gradient in
`laplace_exact_gradient.md`. That note ends at

    dm/dtheta_j = <dSigma_j, S_{m(j)}>,   S_m = 0.5 ( Omega_m M0_m Omega_m - G_m Omega_m )

where `<X, Y> = tr(X Y) = sum(X * Y)` for symmetric X, Y, `dSigma_j` is the
parameterization derivative of block `m(j)`'s covariance, and

    M0_m = R_m + V_m - C_m           (the Laplace second moment, curvature-corrected)
    R_m  = sum_g b_g b_g'            (mode outer products, nc x nc)
    V_m  = sum_g [H^{-1}]_{gg}        (posterior covariance of the same coords)
    C_m  = sym( sum_g b_g u_g' ),    u = H^{-1} A' (dw/deta * s),  s_i = (A H^{-1} A')_ii

Same model as the gradient note: `x = (beta, b)`, `eta = A x + offset`,
`H = A' W A + P`, `w_i = -l''(eta_i)`, `P_re = blockdiag_m ( I_{G_m} (x) Omega_m )`,
`Omega_m = Sigma_m^{-1}`.

## Two channels

`g_j = <dSigma_j, S_{m(j)}>` is a product of two theta-dependent factors: the
parameterization derivative `dSigma_j`, which depends only on block `m(j)`'s
coordinates, and the Sigma-gradient `S_m`, which depends on ALL coordinates
through the mode. The product rule gives

    H_jk = <d2Sigma_{jk}, S_{m(j)}>          (A: parameterization curvature)
         + <dSigma_j, dS_{m(j)}/dtheta_k>    (B: transport of the Sigma-gradient)

A is block-local: `dSigma_j` and `d2Sigma_{jk}` vanish outside block `m(j)`, so A
contributes only when `j` and `k` share a block. B couples every block, because
`S_m` carries `V_m` and `C_m`, and both read the whole inverse Hessian, which
moves when ANY block's theta moves. So `H_theta` is dense across blocks even
though the gradient touches one block at a time.

## Term A: parameterization curvature

`Sigma = L L'`, `L = L(theta)` the log-Cholesky map (log `L_ii` on the diagonal,
raw `L_ij` below). With `L_p = dL/dtheta_p` and `L_pq = d^2 L / dtheta_p dtheta_q`,

    d2Sigma/dtheta_p dtheta_q = L_pq L' + L_p L_q' + L_q L_p' + L L_pq'

`L_pq = 0` except when `p = q` is a diagonal (log) coordinate `(i,i)`, where
`L_pp = E_ii L_ii` (because `d^2 exp(theta)/dtheta^2 = exp(theta)`). So

    off-diagonal p, or p != q:  d2Sigma = L_p L_q' + L_q L_p'
    diagonal p = q = (i,i):     d2Sigma = E_ii L_ii L' + L L_ii E_ii' + 2 L_p L_p'

with `L_p = E_ij` for an off-diagonal coordinate `(i,j)` and `L_p = E_ii L_ii` for
a diagonal one. Term A is then `<d2Sigma_{jk}, S_m>`, a cheap contraction against
the Sigma-gradient already formed for the gradient. It is symmetric in `j, k` and
FD-checkable on its own against a difference of `dSigma_dtheta`.

## Term B: transport of the Sigma-gradient

    dS_m/dtheta_k = 0.5 ( dOmega_m M0_m Omega_m + Omega_m dM0_m Omega_m
                          + Omega_m M0_m dOmega_m - G_m dOmega_m )

    dOmega_m = -Omega_m dSigma_k Omega_m     (nonzero only if k is in block m)
    dM0_m    = dR_m + dV_m - dC_m

Per block, `<dSigma_j, dS_m/dtheta_k>` over all `j` in block `m` is exactly
`recov_block_grad(dS_m/dtheta_k, L_m)`: the same verified log-Cholesky VJP the
gradient uses, now applied to a new matrix. The Hessian re-uses that chain rule
rather than re-deriving it, so the `log L_ii` / `log sigma_i` diagonal handling
and the LKJ penalty come along for free.

For `k` in a different block from `m`, only `Omega_m dM0_m Omega_m` survives
(`dOmega_m = 0`), which is the cross-block coupling.

### The moving pieces of M0

All three use the exact mode Jacobian `J[,k] = dx_hat/dtheta_k = -H^{-1}
(dP_k) x_hat`, already formed in closed form for the gradient's `want_jacobian`
path. Write `Jb_m` for `J[,k]` restricted to block `m`'s coordinates, reshaped
`nc x G` group-major, alongside `b_m` (the mode) and `u_m`.

**dR_m.** `R_m = sum_g b_g b_g'`, so

    dR_m/dtheta_k = Jb_m b_m' + b_m Jb_m'

**dV_m.** `dH^{-1}/dtheta_k = -H^{-1} dH_k H^{-1}` with

    dH_k = A' diag(dW_k) A + dP_k
    dW_k = (dw/deta) * (A J[,k])          (* elementwise; dw/deta = curvature_deta)
    dP_k = I_G (x) dOmega_k                 (block m(k) only)

    dV_m/dtheta_k = -sum_g [H^{-1} dH_k H^{-1}]_{gg}    (block m groups)

**dC_m.** `C_m = sym(sum_g b_g u_g') = sym(b_m u_m')`, `u = H^{-1} A' r`,
`r = (dw/deta) * s`. Then

    dC_m/dtheta_k = sym( Jb_m u_m' + b_m du_m' )
    du/dtheta_k   = -H^{-1} dH_k u + H^{-1} A' dr_k
    dr_k          = (d2w/deta2) * (A J[,k]) * s  +  (dw/deta) * ds_k
    ds_k          = (A dH^{-1}_k A')_ii = -(A H^{-1} dH_k H^{-1} A')_ii

The ONE quantity here not already computed for the gradient is `d2w/deta2`, the
second eta-derivative of the weight `H` is built from -- the sibling of
`dw/deta = curvature_deta_for_family`. For the families whose `w` is the true
observed curvature (poisson, binomial, neg_binomial_2, the truncated pair) it is
`-l''''(eta)`; for the working-weight families (neg_binomial_1, beta_binomial,
t, tweedie, the generic mu-space route) it is the second eta-derivative of the
working weight, under the same naming discipline as `laplace_family_curvature.h`:
it is the derivative of what the Newton system uses, not a labelled `d4`.

## Assembled algorithm (one column k)

    mk       = block(k)
    dOmega_k = -Omega_mk dSigma_k Omega_mk            # block mk
    J_k      = -H^{-1} (dP_k x_hat)                    # mode Jacobian column
    eta_dot  = A J_k
    dH_k     = A' diag((dw/deta) * eta_dot) A + dP_k
    dHinv_k  = -H^{-1} dH_k H^{-1}
    ds_k     = rowSums((A dHinv_k) * A)
    dr_k     = (d2w/deta2 * eta_dot) * s + (dw/deta) * ds_k
    du_k     = dHinv_k (A' r) + H^{-1} (A' dr_k)
    for each block m:
        dR_m  = Jb_m b_m' + b_m Jb_m'
        dV_m  = sum_g dHinv_k[gg]
        dC_m  = sym( Jb_m u_m' + b_m du_m' )
        dM0_m = dR_m + dV_m - dC_m
        dS_m  = 0.5( [m==mk](dOmega_k M0_m Om_m + Om_m M0_m dOmega_k - G_m dOmega_k)
                     + Om_m dM0_m Om_m )
        H[block m, k] += recov_block_grad(dS_m, L_m)          # term B
    H[block mk, k] += ( <d2Sigma_{j,k}, S_mk> )_{j in block mk}  # term A

Symmetrize at the end; `H` is symmetric up to the inner solver's noise.

## Gaussian control

For a Gaussian response `dw/deta = 0` and `d2w/deta2 = 0`, so `C = 0`, `dC = 0`
and `dW_k = 0`, leaving `dH_k = dP_k`. The Hessian does NOT collapse to term A:
`dR_m` (through `J` with `dH_k = dP_k`), `dV_m` and the parameterization curvature
all remain. So the Gaussian row exercises `J`, `dR`, `dV` and term A while pinning
the curvature channel (`dw`, `d2w`) at exactly zero -- the analogue of the
gradient note's Gaussian control, where `C` vanishes.

## Numerical confirmation

`dev_notes/proto_exact_hessian.R` builds the model, mode, `H`, the Laplace
marginal and the analytic gradient from scratch in base R, then compares the
assembled analytic Hessian against a central difference of the analytic gradient
(itself checked against the marginal in `proto_exact_gradient.R`), and against a
second difference of the marginal directly.

    case                          k   H max|rel| vs FD   raw asymmetry
    poisson scalar RE             1   2.6e-11            0
    poisson correlated (nc=2)     3   7.8e-10            2.7e-15
    binomial scalar RE            1   2.4e-11            0
    binomial correlated (nc=2)    3   1.6e-09            1.5e-15
    gaussian scalar RE (control)  1   4.3e-11            0
    poisson two crossed blocks    2   8.0e-10            2.7e-16
    binomial intercept + slope    4   5.1e-09            1.1e-15

The middle column is the analytic `H_theta` against a central difference of the
analytic gradient (`h = 1e-5`, so its own `O(h^2)` truncation is ~1e-10 on top of
the gradient's ~1e-9 noise); every case sits at that floor. The right column is
the natural asymmetry of the analytic form BEFORE symmetrizing: the two
independent code paths to `H_jk` and `H_kj` -- column `j` and column `k`, built
from different `dP`, `dH`, `du`, `dHinv` -- agree to machine precision, a check
the finite difference cannot supply. The two multi-block rows (crossed intercepts;
an intercept block beside a correlated-slope block) exercise the cross-block
coupling `Omega_m dM0_m Omega_m` that the gradient has no analogue for. The
parameterization second derivative `d2Sigma` is separately checked against a
difference of `dSigma` at 9.2e-11.

## What this buys

Closed-form `H_theta`: one assembly reusing the gradient's factorization of `H`,
against `2k` exact-gradient evaluations (the current `.eb_exact_stencil`) or
`1 + 2k^2` objective re-solves (`.eb_fd_stencil`). It removes the last `O(h^2)`
truncation term from the marginal correction's `H_theta`; `J` is already
closed-form there, so the whole correction becomes finite-difference-free on the
family set where `has_curvature_2nd_derivative()` holds.

## Ingredients needed from the inner solve

Everything the gradient already needs (`H^{-1}`, `s`, `dw/deta`, `J`, per-block
`R, V, C, Omega, L`), plus:

  * `d2w/deta2` per family (new; `curvature_deta2_for_family`, gated by
    `has_curvature_2nd_derivative`).
  * `d2Sigma/dtheta_p dtheta_q` per block (new; `.re_block_d2Sigma`, a pure
    parameterization derivative, no inner-solve state).
  * `dH_k`, `dH^{-1}_k` for `k` right-hand sides, reusing the factorization.

## Dispersion coordinate

When `estimate_phi = TRUE`, `log phi` joins the stacked theta. The `phi x phi`
and `phi x theta` Hessian entries need second phi-derivatives of loglik / score /
weight -- an extension of `.family_dphi` (`family_dispersion.R:156`) with
`dloglik2 / dscore2 / dweight2` -- and are deferred to a second pass. Until then
the closed-form route fills the `theta x theta` block and the `phi` row/column
falls back to differencing the analytic gradient, which already carries the exact
`dm/dphi`.
