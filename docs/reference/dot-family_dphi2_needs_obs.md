# Whether a family's phi Hessian additionally needs `d(W_obs)/dphi`

TRUE whenever the Newton working weight is not the observed curvature,
which is exactly when the border's `Hinv` and the mode motion's
`Hinv_mode` are different matrices. For a family whose weight is also
free of eta the two inverses multiply a zero channel and the answer
would come out the same either way, but the entry is required regardless
rather than resting on that second coincidence.

## Usage

``` r
.family_dphi2_needs_obs(family)
```
