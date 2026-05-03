# Gradient Kernel Coverage Matrix

Single source of truth for which feature combinations each specialized
H-mode gradient kernel supports. Each entry must agree with the named
predicate in `hmc_gradient_dispatch_predicates.h` — the predicate is the
runtime guard, this table is the human-readable summary.

When adding a new latent-structure feature (a new `layout.has_*` flag):
every kernel below must be reviewed. Either the kernel handles the new
feature (column gains a `Y`) or its predicate must add the new flag to
its exclusion list (column stays `N`).

## Legend

- `Y` — kernel supports this feature.
- `N` — kernel does not support this feature; predicate must exclude it.
- `OPT` — kernel handles only one variant of this feature; see footnote.
- `—` — flag is mutually exclusive with the kernel's primary feature.

## Kernel × feature matrix

| Kernel                           | RE intercept | RE slopes | Spatial (ICAR/BYM2/CAR-prop) | GP / mGP / HSGP | Temporal (RW/AR1) | Temporal GP | Multi-scale temporal | SVC | TVC | Spatiotemporal | Latent | ZI / OI |
|----------------------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `analytical` (catch-all)         | Y | Y | Y | N | Y | N | N | N | N | N | N | Y |
| `composite` (exotic catch-all)   | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `hsgp` (HSGP spatial only)       | Y | N | — | OPT¹ | N | N | N | N | N | N | N | N |
| `gp_collapsed`                   | Y | N | — | OPT² | N | N | N | N | N | N | N | N |
| `icar_collapsed`                 | Y | N | OPT³ | — | N | N | N | N | N | N | N | N |
| `gp_handcoded`                   | Y | N | — | OPT⁴ | N | N | N | N | N | N | N | N |
| `gp_plus_temporal_handcoded`     | Y | N | — | OPT⁴ | Y (RW/AR1) | N | N | N | N | N | N | N |
| `msgp_hsgp`                      | Y | N | — | OPT⁵ | N | N | N | N | N | N | N | N |
| `msgp_handcoded`                 | Y | N | — | OPT⁶ | Y (RW/AR1) | N | N | N | N | N | N | N |
| `msgp_plus_temporal_handcoded`   | Y | N | — | OPT⁶ | Y (RW/AR1) | N | N | N | N | N | N | N |
| `svc_handcoded`                  | Y | N | — | N | N | N | N | OPT⁷ | N | N | N | N |
| `svc_hsgp_handcoded`             | Y | N | — | N | N | N | N | OPT⁸ | N | N | N | N |
| `tvc_handcoded`                  | Y | N | N | N | N | N | N | N | Y | N | N | N |
| `temporal_gp_handcoded`          | Y | N | N | N | Y | OPT⁹ | N | N | N | N | N | N |
| `ms_temporal_handcoded`          | Y | N | N | N | N | N | Y | N | N | N | N | N |
| `spatiotemporal_handcoded`       | Y | N | Y | N | Y (RW only) | N | N | N | N | OPT¹⁰ | N | N |
| `latent_handcoded`               | Y | N | N | N | N | N | N | N | N | N | Y | N |

### Footnotes

1. `hsgp`: `layout.is_hsgp && data.has_hsgp`.
2. `gp_collapsed`: `layout.is_gp_collapsed && data.has_gp && data.gp_collapsed`.
3. `icar_collapsed`: `layout.is_icar_collapsed || layout.is_bym2_collapsed`.
4. `gp_handcoded` / `gp_plus_temporal`: `layout.is_gp && data.has_gp` (full GP basis).
5. `msgp_hsgp`: `layout.is_multiscale_gp && data.has_multiscale_gp && data.msgp_is_hsgp`.
6. `msgp_handcoded` / `msgp_plus_temporal`: `layout.is_multiscale_gp && data.has_multiscale_gp` (any basis).
7. `svc_handcoded`: full GP-basis SVC (`!data.svc_is_hsgp`).
8. `svc_hsgp_handcoded`: HSGP-basis SVC (`data.svc_is_hsgp`).
9. `temporal_gp_handcoded`: exponential covariance only (`cov_type == EXPONENTIAL`).
10. `spatiotemporal_handcoded`: structured ST with non-AR1 temporal type
    (AR1+ST routes to arena autodiff; see dispatch.h pre-filter).

## Cross-cutting exclusions

The dispatch resolver in `hmc_gradient_dispatch.h` applies these
*before* per-kernel guards:

- **Generic multi-process models** (`data.n_processes > 0` with a
  `LikelihoodSpec`): always route to `compute_gradient_generic_arena`
  (or `_numerical` if no arena likelihood).
- **Collapsed ICAR/BYM2 + AUTODIFF mode**: redirect to numerical (the
  inner Newton solve is not differentiable through templated AD).
- **ZOIB**: always arena (pre-existing H-mode bug).
- **ZI/OI**: route to `composite` after the analytical fast path.
- **Crossed RE (n > 1) + any exotic feature**: route to arena.
- **TVC + latent**: route to arena (no specialized kernel).
- **Spatiotemporal + AR1 temporal**: route to arena.

## Updating this file

When you add a new specialized kernel:

1. Add a row above with `Y`/`N`/`OPT` per column.
2. Add a corresponding `can_use_<kernel>(...)` predicate in
   `hmc_gradient_dispatch_predicates.h` whose negative tests match the
   `N` columns exactly.
3. Add a dispatch line in `hmc_gradient_dispatch.h` calling the
   predicate, ordered by specificity (most specific first).
4. Add at least one test in `tests/testthat/` that hits the new branch.
