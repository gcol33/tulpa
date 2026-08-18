# Intrinsic ICAR field precision, augmented to full rank.

`L = D - W` (degree minus adjacency) plus the sum-to-zero augmentation
`sum_c 1_c 1_c' / J_c` that identifies each component's constant null
direction (`inst/include/tulpa/sum_to_zero.h`). The components are the
actual connected components of the graph (`.graph_components`), each
pinned over its own node set of size `J_c` – matching
`for_each_icar_component` (`src/icar_kernel.h`), so an unequal /
non-contiguous disconnected map reconstructs the same precision the
kernel penalizes with.

## Usage

``` r
.icar_precision_Q(spatial)
```

## Details

The conditional Laplace kernel passes `tau_spatial = 1` on both the
ICAR/CAR path (`R/fit_laplace.R`) and the BYM2 structured block (which
carries sigma and rho in its `d_fac` instead), and the field enters the
linear predictor with `d_fac = 1`, so this is exactly the field block of
that fit's joint Hessian. `test-marginal-se-areal.R` pins it against the
kernel's own `log_prior_icar`.
