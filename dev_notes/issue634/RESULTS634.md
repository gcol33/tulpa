# gcol33/tulpa#634 -- is the skew-normal rescue reachable? Yes, and my premise was wrong

Filed while fixing gcol33/tulpa#631, on the observation that the rescue stopped
being adopted on `test-outer-skew-rescue.R`'s fixtures once the outer-k read was
made budget-stable. The issue generalized that to "not adopted on any fixture in
the repo". **That is wrong, and the data contradicting it was already committed
when I filed** -- `dev_notes/issue629/sweep629.csv`, 3300 rows, has 7 adoptions.
The correct statement is narrower: not adopted on the handful of TEST fixtures.

## The sweep, re-run on current main

`probe629.R` re-run to `sweep634.csv` (the #629 baseline left untouched). It
scores at `n_samples = 500`, which is the default budget, so #631's held tail
fraction is a no-op there by construction.

**Bit-identical to `sweep629.csv`: 3300 rows x 24 columns, every column, `src`
included.** That is the #630 verification standard, and it is the strongest
statement available that #631 did not move a default fit.

## Adoption

| adopted source | rows | share |
|---|---|---|
| grid_moment | 1823 | 55.2% |
| grid_mixture | 1367 | 41.4% |
| moment_matched | 103 | 3.1% |
| **skew_normal** | **7** | **0.2%** |

The rescue is SCORED on 34 of 3300 (1.03%) and adopted on 7 of those 34.

## What it buys where adopted

| target | nodes | half-width | seed | k_gm | k_gauss | k_mix | k_skew | shipped |
|---|---|---|---|---|---|---|---|---|
| heavy df=4 | 15 | 3 | 8 | 1.696 | 1.240 | 0.862 | 0.813 | 0.813 |
| heavy df=3 | 5 | 3 | 14 | 1.226 | 0.889 | 1.161 | 0.977 | 0.977 |
| heavy df=2 | 5 | 3 | 14 | 1.250 | 0.863 | 1.350 | 0.757 | 0.757 |
| heavy df=2 | 5 | 3 | 18 | 1.545 | 0.534 | 1.019 | 0.361 | 0.361 |
| heavy df=2 | 9 | 3 | 14 | 1.377 | 0.880 | 0.843 | 0.757 | 0.757 |
| heavy df=2 | 15 | 3 | 14 | 1.678 | 0.905 | 1.246 | 0.757 | 0.757 |
| heavy df=2 | 41 | 6 | 3 | 0.549 | 0.549 | 0.512 | 0.253 | 0.253 |

Median gain over the best non-skew candidate: **0.106**. Band crossings
(`k_usable = 0.7`): **0 of 7**. So on this sweep the rescue improves the NUMBER
and never changes the VERDICT.

**Every adopted row is a target with true skewness exactly 0.** All seven are
`heavy(a=0, df=2/3/4)` -- symmetric Student-t. The gate reads the whitened
SAMPLE skewness, which on a heavy tail at 500 draws is noisy enough to clear it,
so the rescue engages on sampling noise rather than on the asymmetry it was
built for. The two genuinely skewed heavy targets in the sweep
(`skewt(a=5,df=4)` with true skew 3.509, `skewt(a=12,df=3)` with 6.742) account
for 3 of the 34 scored rows and 0 of the 7 adoptions.

This is consistent with #629 -- the rescue reads worse than the Gaussian
wherever it is scored, on the median -- and adds why it sometimes wins anyway:
on a symmetric heavy tail a skew normal has one more free parameter to fit the
realized draws with, which lowers the shape estimate without describing the
target better.

## Not a minimum-violation, and a doc correction

Row 2 ships 0.977 with a moment-matched Gaussian of 0.889 in hand, which looks
like the dispatch failing to keep the minimum. It is not: `.k_score_symmetric()`
compares the mixture to the GRID-MOMENT Gaussian, not the moment-matched one,
deliberately and with the reason written beside it -- a refined Gaussian that
got under the mixture only by widening PAST the grid is not a faithful
within-grid reading and must not mask a grid-width deficiency. The mixture (1.161)
displaced the Gaussian (0.889) under that rule, then the rescue improved on the
mixture.

CLAUDE.md said "the minimum is kept, so a candidate can never make a fit read
worse", which is true per rung and false across the four. Corrected.

## Verdict: keep, and do not read the gate as a skew detector

Nothing here says retire. The rescue costs an extra scoring pass on 1% of fits,
each rung keeps the better of itself and its input, and it is adopted rarely but
non-zero. Against that, `Suggests`-free breadth on a research-stage engine is
the house position, and there is no measured harm from keeping it.

What IS worth recording, and is the useful half of this issue:

- The rescue **never moved a band** on 3300 configurations. Anyone reading
  `pareto_k_proposal_source == "skew_normal"` as "this fit needed a skew
  correction" is over-reading it.
- The gate fires on sample-skewness noise on symmetric heavy tails, not on
  genuine asymmetry. A gate keyed on something more robust than the whitened
  sample third moment at 500 draws would fire on the right cases -- but the
  right cases in this sweep are ones the Gaussian and the mixture already
  handle, so there is no measured gain waiting behind it.

Open, and NOT answered here: whether a target exists that the rescue
legitimately wins on for the reason it was designed for. The sweep is 11 target
families on one grid geometry; it is not a search.
