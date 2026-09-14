# A flat outer prior at the nested-Laplace doors, and the research figures re-measured under it

## Build

Every figure below was measured on two libraries installed from clean exports
(no object files before the build) of commit `85a3161`, R 4.6.1,
x86_64-w64-mingw32:

- `lib_head`: `git archive 85a3161` as is.
- `lib_hp`: the same export with these files overlaid (git blob hashes):
  `R/hyperprior_default.R 9f31fa1`, `R/nested_laplace_joint_ccd_local.R 2d5d0dc`,
  `R/nested_laplace.R 6efe21c`, `R/nested_laplace_joint.R 914b9b5`,
  `R/nested_laplace_joint_multi.R d07bc86`, `R/nested_laplace_re_cov.R f6a281f`,
  `R/tulpa.R a6c2962`, `man-roxygen/hyperprior.R 30361fd`.
  `src/` and `R/RcppExports.R` are the commit's. `out/BUILD_ID_lib_hp.txt` is the
  identity file every sweep printed and stamped into its output.

## The default is unchanged

`identity_fits.R` fits four fixtures (a three-block joint gaussian grid, a
single-block joint ICAR binomial fit with its default grid and k-hat, and the
single-block and multi-block `tulpa_nested_laplace()` doors) on both libraries;
`identity_compare.R` compares the fit objects. With no `hyperprior` argument and
with `hyperprior = "proper"`, every element is `identical()` to `lib_head`'s on
all four except `diagnose_cost_ratio`, which is diagnostic seconds over fit
seconds. Under `"flat"` every fit carries `log_hyperprior` 0, names every axis
`flat_hyperprior`, and declines its evidence with `improper_hyperprior`.

## Figures, by the script that produces them

Old values are the ones the tests quoted before 33bda43 / 878534c. "box" is the
`box_uniform` within-cell read the ogd fixture states (gcol33/tulpa#599),
"chord" the read these figures were first measured under.

### #331, `issue331/coverage331.R` (200 seeds, chord, as `lccd_arm_sweep()`)

| figure | old | flat | proper |
|---|---|---|---|
| location covers sigma_1, 4 levels | 128 / 200 | 128 | 128 |
| per-seed distance to the 6-level width, location / shipped | 0.3576 / 0.4194 | 0.3576 / 0.4194 | 0.3546 / 0.4240 |
| seeds location loses | 72 | 72 (71 high, 1 low) | 72 (70 high, 2 low) |
| the whole recovery-file table (coverage, widths, discordance, width ratios) | as quoted | identical to 4 decimals | see `out/cov331_proper.txt` |

### #328, `issue_328/measure_fit_ranking.R` (8 seeds x 3 budgets)

The ranking is selected through `control$local_ccd$rank`; `"mass_moved"`
reproduces the base, 5-level, weight and moved totals of the earlier
namespace-patching run to four decimals under flat + chord.

| figure | old | flat chord | proper chord | flat box | proper box |
|---|---|---|---|---|---|
| median ranking difference above the floor | 14 of 24 | 14 | 15 | 10 | 10 |
| endpoints / widths above the floor | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 4 | 0 / 2 |
| moved ranking nearer on the median | 0 of 24 | 0 | 1 | 1 | 1 |

The noise floors themselves moved (median floor sum 0.2859 then, 0.2381 flat
chord now), which the prior does not explain.

### #327, `issue327/bary327.R`

- Numeric route against Simpson: 1.55e-15 on x86_64-w64-mingw32 (this build).
  The 2e-12 is aarch64-apple-darwin23 from GitHub Actions run 32475678231
  (image macos-26-arm64, R 4.6.1); `out/ci_run_32475678231_macos_arm64_excerpt.txt`
  is the log lines. Not re-run on arm64 here.
- Five-level placement report, box: endpoints / widths / median 1.09x / 3.93x /
  1.34x their floors flat, 0.97x / 1.94x / 1.70x proper. The flat prior restores
  "all three parts above the floor" and "widths more than 2.5x".
- Four-arm tables: under flat + chord the seed-4242 four- and five-level tables
  match the old ones to four decimals; under box they are the same for both
  priors to within 0.035 and carry the claims 878534c wrote.

### #333, `issue333/plane333.R` + `analyse333.R`

Test sweep = spread 3, seeds 1-8, gaussian. Three-spread sweep = spreads 2, 3, 4,
seeds 1-8, gaussian and binomial, 72 fits each. The old "1680 cells from 144
fits" is not reproduced by that layout (840 cells from 72); the old sweep's
seeds and spreads beyond "8 seeds x 3 spreads" were not recorded, and its
binomial figures (0.9072, -0.3057, -136x / -668x) are not reproduced.

| figure | old | flat box (test) | proper box (test) | flat box (3 spreads) |
|---|---|---|---|---|
| median abs rho, loc vs mass (bound) | < 0.2 | 0.0885 | 0.2051 | 0.1816 |
| A-metric Spearman pooled / 4 levels | 0.9495 / 0.9830 | 0.9209 / 0.9761 | 0.9204 / 0.9752 | 0.9483 / 0.9776 |
| median contingency p, 4 / 5 levels | < 1e-4 / < 1e-4 | 0.261 / 0.595 | 0.267 / 0.391 | 2.1e-10 / 0.0016 |
| strongest loc-vs-mass rho | -0.3178 (5-level median) | +0.3810 (4-level median) | +0.4651 | +0.3807 |
| same, chord | | +0.2253 | -0.2045 (5-level median) | -0.2795 (5-level median; proper -0.2437 vs +0.2598) |

Full outputs: `issue333/out/an_<prior>_<family>[_chord]_all.txt` and
`an_<prior>_g_s3[_chord].txt`.

### RE-covariance CCD interval, `issue730/re_cov_ccd_interval.R`

log(sigma_1) nodes / interval: flat [-0.32935, 0.26114] / [-0.33464, 0.26504];
pc_lkj [-0.33816, 0.24291] / [-0.34293, 0.24240]. Under both the interval is
m +- z s of the design-weighted moments to 1.1e-16.

## Reading

On these fixtures the prior moves little. What restores most of the removed
figures is running the experiment as it was run: #331 needs only the flat
prior; the #327 tables, #328's 14 of 24 and the #333 sign claims need the
`chord` read as well, the engine default those figures were taken under before
0.0.188; the ogd fixture has stated `box_uniform` since gcol33/tulpa#599. The
tests keep that read and quote the chord figures beside it.
