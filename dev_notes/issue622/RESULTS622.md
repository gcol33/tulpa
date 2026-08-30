# What a boundary node's weight says about the tail a grid leaves out (gcol33/tulpa#622)

`probe_edge_mass_lift.R`: 1400 arrangements -- four marginal shapes (gaussian,
Laplace, Student-3, lognormal), five node counts, 35 spans -- each scored on the
outer node's `lift = m * w_edge` and on the true mass beyond the outer CELL EDGE,
which is the support a reported interval extends to.

## Lift against the mass the span does not reach

| tail band | n | lift min | lift median | lift max |
|---|---|---|---|---|
| <= 1e-4 | 128 | 5.1e-08 | 0.0008 | 0.051 |
| 1e-3 to 1e-2 | 347 | 0.011 | 0.094 | 1.21 |
| 1e-2 to 0.05 | 265 | 0.059 | 0.206 | 2.16 |
| 0.05 to 0.1 | 142 | 0.274 | 0.473 | 3.07 |
| 0.1 to 0.25 | 196 | 0.410 | 0.972 | 4.38 |
| > 0.25 | 210 | 0.906 | 2.530 | 7.09 |

## A lift cutoff as a detector of "more than 5 % left outside"

| cutoff | caught | false alarm | worst falsely flagged tail |
|---|---|---|---|
| 0.50 | 0.858 | 0.079 | 0.049 |
| 0.75 | 0.723 | 0.016 | 0.046 |
| 1.00 | 0.600 | 0.011 | 0.046 |
| 1.25 | 0.520 | 0.002 | 0.030 |

And what a flagged axis is leaving out:

```
lift >= 0.75 : n  410  tail min 0.0074  q10 0.0846  median 0.258
lift >= 1.00 : n  338  tail min 0.0074  q10 0.0947  median 0.540
lift >= 1.25 : n  287  tail min 0.0304  q10 0.1055  median 0.726
```

`.NL_DIAG$edge_mass_lift = 1` is the point at which the boundary node carries
what a flat marginal would put there. It is a ONE-SIDED read: a high lift means
the span truncates; a low one does not certify that it does not. That is why it
is reported rather than acted on.

## The issue's own cases, at m = 9

```
observed fit         ceiling   6.9%  lift 0.62   not named
20% on the ceiling   ceiling  20.0%  lift 1.80   named
34% on the ceiling   ceiling  34.0%  lift 3.06   named
ceiling is mode      ceiling  44.0%  lift 3.96   named, and railed
```

`repro_labels.R` runs the issue's five cases plus a resolved control through the
shipped labels.
