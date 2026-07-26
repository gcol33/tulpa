# The dispersion (log phi) block of the exact outer Hessian

Extends `laplace_exact_hessian.md` from the `theta x theta` RE-covariance block to
the dispersion coordinate. When `estimate_phi = TRUE` the stacked outer parameter
is `chi = (theta_1, ..., theta_k, psi)` with `psi = log phi`, and the marginal
correction needs the two further Hessian blocks

    H_{theta_j, psi}   (the phi column's RE rows)
    H_{psi, psi}       (the phi diagonal)

on top of the RE block already assembled by `.laplace_exact_re_hess`. The phi
GRADIENT `dm/dpsi = grad_phi` is already exact (`.laplace_exact_re_grad`,
`want_phi`); this note differentiates it once more.

Symbols carry over from the RE note. Per observation, with `wt_i` the prior
weight, `mu = mu(eta)`, and the family evaluated at `(eta, phi)`:

    score_i = dloglik_i/deta          W_i     = -l''_i           (Newton weight)
    dwde_i  = dW_i/deta               d2wde_i = d2W_i/deta2       (curvature ladder)
    L1_i    = dloglik_i/dphi          DW_i    = dW_i/dphi         Sc1_i = dscore_i/dphi

`L1, DW, Sc1` are the three already in `.FAMILY_DPHI` (`dloglik / dweight /
dscore`). The gradient reads

    grad_phi = phi * g_pre,
    g_pre    = sum_i wt_i L1_i - 0.5 sum_i s_i ( wt_i DW_i + dw_i deta_dphi_i )

with `dw_i = wt_i dwde_i`, `s_i = (A H^-1 A')_ii`, and the mode's phi-motion

    dx/dphi   = H_true^-1 A' (wt * Sc1),     deta_dphi = A dx/dphi

carrying the OBSERVED-curvature Hessian `H_true` from the mode-Jacobian note (for
the registered dispersion families neg_binomial_2 / gaussian / gamma the working
and observed weights coincide, so `H_true = H_joint`; the general case reuses the
Road-A correction `H_true = H_joint + A' diag(W_obs - w) A`).

## The phi column, H_{theta_j, psi}, reuses term B

`g_theta_j = <dSigma_j, S_m>`, `S_m = 0.5(Omega_m M0_m Omega_m - G_m Omega_m)`.
Only the mode moves with `psi` (the RE prior `P` is phi-free), so exactly as for a
cross-block RE coordinate,

    H_{theta_j, psi} = <dSigma_j, dS_m/dpsi>,   dS_m/dpsi = 0.5 Omega_m dM0_m^psi Omega_m

(no `dOmega` term: `psi` does not touch `Sigma`). This is the term-B assembly with
a phi "column k = psi" whose ingredients are:

    dP_psi   = 0                                        (RE prior is phi-free)
    J_psi    = dx/dpsi = phi * H_true^-1 A'(wt Sc1)      (= phi dx/dphi)
    eta_dot  = A J_psi
    dW_psi_i = dw_i eta_dot_i + phi wt_i DW_i            (through-mode + explicit)
    dH_psi   = A' diag(dW_psi) A                         (= dH_joint/dpsi, no dP)
    dHinv_psi= -H_joint^-1 dH_psi H_joint^-1
    ds_psi   = rowSums( (A dHinv_psi) * A )
    dr_psi_i = ( d2w_i eta_dot_i + phi wt_i DWde_i ) s_i + dw_i ds_psi_i
    du_psi   = dHinv_psi v_r + H_joint^-1 A' dr_psi
    per block m:
      dR_m^psi = Jb_m^psi b_m' + b_m Jb_m^psi'
      dV_m^psi = sum_g dHinv_psi[gg]
      dC_m^psi = sym( Jb_m^psi u_m' + b_m du_m^psi' )
      dM0_m^psi= dR_m^psi + dV_m^psi - dC_m^psi
      dS_m^psi = 0.5 Omega_m dM0_m^psi Omega_m
      H[block m, psi] += recov_block_grad(dS_m^psi, L_m)

The ONE new family quantity here beyond `.FAMILY_DPHI` is

    DWde_i = d(dW/dphi)/deta = d2W/(deta dphi)          (mixed eta-phi curvature)

everything else is `L1, DW, Sc1, dwde, d2wde` already available. Note `d(dwde)/dphi
= d2W/(deta dphi) = DWde` as well, which is why the single mixed derivative covers
both `dr_psi`'s explicit piece and the weight's cross term.

## The phi diagonal, H_{psi, psi}

`H_{psi,psi} = d(grad_phi)/dpsi = grad_phi + phi^2 dg_pre/dphi`. Differentiating
`g_pre` totally in phi (explicit + through the mode) gives, per observation,

    d/dpsi [ sum wt L1 ]
      = sum wt ( Sc1 eta_dot + phi L2 )                 L2 = d2loglik/dphi2  (NEW)
    d/dpsi [ sum s (wt DW + dw deta_dphi) ]
      = sum ds_psi (wt DW + dw deta_dphi)
      + sum s ( wt (DWde eta_dot + phi DW2)             DW2 = d2W/dphi2      (NEW)
                + d(dw)/dpsi deta_dphi + dw d(deta_dphi)/dpsi )
    d(dw)/dpsi   = d2w eta_dot + phi wt DWde
    d(deta_dphi)/dpsi = A d(dx/dphi)/dpsi

so `H_{psi,psi} = grad_phi + phi * [ d/dpsi(sum wt L1) - 0.5 d/dpsi(sum s (...)) ]`.
The one genuinely new solve is the SECOND mode derivative `d(dx/dphi)/dpsi`. From
differentiating `H_true dx/dphi = A'(wt Sc1)` in psi,

    H_true d(dx/dphi)/dpsi = A' d(wt Sc1)/dpsi - dH_true/dpsi dx/dphi
    d(wt Sc1)/dpsi = wt ( Sc1_de eta_dot + phi Sc2 )    Sc2 = d2score/dphi2  (NEW)
    Sc1_de = d(dscore/dphi)/deta = d3loglik/(deta2 dphi) = -DW  (free identity)

so `d(dx/dphi)/dpsi` costs one more `H_true` solve against a right-hand side built
from `Sc2` (new), `DW` (have), and `dH_true/dpsi` (the observed-curvature analogue
of `dH_psi`; equals `dH_psi` for the registered coincident families).

## New family derivatives (`.FAMILY_DPHI` extension)

Four scalars per observation, siblings of the existing `dloglik / dscore /
dweight`, differentiating the SAME registered `loglik / score / weight`:

    dloglik2 = d2loglik/dphi2          (L2)
    dscore2  = d2score/dphi2           (Sc2)
    dweight2 = d2W/dphi2               (DW2)
    dweight_deta = d2W/(deta dphi)     (DWde, the mixed term the RE column needs)

`Sc1_de = -DW` is an identity, not a new function. Gate as `.family_dphi2()`
alongside `.family_dphi()`; a family missing the second-order block keeps the
current behaviour (the phi row/column differences the analytic gradient).

## Validation (proto-first, per the codebase discipline) -- DONE

1. `proto_phi_hessian.R`: build the `(theta, psi)` model, mode, objective, analytic
   gradient (RE + phi) and analytic Hessian (RE block + phi column + phi diagonal)
   from scratch, check against a central difference of the analytic gradient and of
   the objective. gaussian FIRST (constant weight: exercises L1/L2/DW/DW2/Sc2,
   dV/dpsi, dR/dpsi, the mode motion, but NOT the through-mode weight terms), then
   neg_binomial_2 (working == observed, dwde != 0: adds the C / dr / d2x-dphi2
   terms and the mixed DWde). PASSED: gaussian 3.7e-10, neg_binomial_2 2.3e-9. The
   second-order family derivatives are numeric in the proto (so it pins the
   ASSEMBLY); the closed forms of L2/Sc2/DW2/DWde are FD-checked separately.
2. Wired: `.family_dphi2` (closed forms, FD-checked in `test-family-dispersion.R`
   against the first-order forms, which chains to the registry); the phi column
   and diagonal into `.laplace_exact_re_hess`, formed once with the phi gradient
   in `.laplace_exact_re_grad`; the log-phi column of `J`; and `log phi` threaded
   through all three marginal-correction routes (`.eb_marginal_correction` takes
   `chi = (theta_hat, log phi_hat)`, the prior Hessian is zero-padded for the
   unpenalized phi, and the FD evaluator splits chi). `proto_phi_hessian_inpkg.R`
   re-checks the whole assembly end-to-end: gaussian 3.6e-10, neg_binomial_2
   2.0e-9 against a difference of the analytic gradient.
3. In-package: route equivalence (closed == exact-stencil == fd, `H_theta` and the
   phi column of `J`) in `test-eb-marginal.R`, and a recovery/coverage check
   (`phi_coverage.R`): the log-phi Wald interval covers log(phi_true) at ~0.95
   (0.95 at 14 groups, 0.92 at 8 where the log-phi marginal is skewed, 0.98 at 40),
   and carrying phi uncertainty into the fixed-effect block is near-inert on
   balanced designs (SD ratio within 0.002 of 1) -- the phi block's payoff is
   phi's own interval and the BLUP block, mirroring the RE hyperparameter
   correction's near-inertness on the fixed effects.

## Scope note

The registered dispersion families are neg_binomial_2, gaussian, gamma. beta stays
unregistered (its assembled `dm/dphi` is already inexact, see
`family_dispersion.R`), so its second-order block is moot until the first is fixed.
The phi diagonal's `d2x/dphi2` solve and the RE column's `J_psi` both take
`H_true`, so the Road-A observed-curvature machinery is a prerequisite already in
place.
