# Which per-axis SD estimator to report, and at what resolution (gcol33/tulpa#621)

Two estimators of one quantity, the posterior SD of one outer axis: the
grid-weighted spread of the axis marginal, and the 3-point parabola at the modal
node. `probe_axis_sd_estimators.R` and `probe_ess_threshold.R` sweep node
spacing `h / sd` from 0.25 to 4 and the mode's offset inside its cell from 0 to
0.9, on a Gaussian axis marginal (where the parabola is exact by construction,
so the arm says what the weighted read costs) and on a skewed one (where the
parabola targets a different number).

## The ladder the threshold came off

Worst RELATIVE error over every arrangement clearing each ESS threshold:

| threshold | weighted (gaussian) | weighted (skew) | parabola (skew) |
|---|---|---|---|
| 1.5 | 1.00 | 0.526 | 0.92 |
| 2.0 | 1.00 | 0.526 | 0.92 |
| 2.5 | 8.5e-04 | 0.303 | 0.92 |
| 3.0 | 9.8e-06 | 0.074 | 0.92 |
| 3.5 | 1.1e-07 | 0.045 | 0.92 |
| 4.0 | 7.5e-11 | 0.025 | 0.92 |

Below 2.5 the weighted read is up to 100 % wrong on a Gaussian and the parabola
is exact, which is the regime the parabola serves. At 3 the weighted read is
exact on a Gaussian to 1e-05 and within 7.4 % on a skewed marginal, against the
parabola's 92 %. `.NL_DIAG$axis_sd_ess = 3`.

## Grid dependence on ONE density

At `ess >= 3`, over every arrangement of the same skewed density, the weighted
read spans 0.974 to 1.074 of the truth and the parabola spans 0.080 to 0.642.

`probe_gamma_fixture.R` is the fixture the regression test uses, a Gamma(1.5, 2)
marginal (own SD 2.449) read on three grids:

```
coarse   ess  7.70  source weighted sd 2.3018  stencil 1.1583
fine     ess 22.97  source weighted sd 2.2598  stencil 1.3948
shifted  ess  7.16  source weighted sd 2.2815  stencil 1.8789
```

The parabola moves by a factor of 1.62 across grids of one density and sits at
half the spread; the weighted read holds to 2 %. That is #621's reported factor
of two on one data set, reproduced without fitting anything.
