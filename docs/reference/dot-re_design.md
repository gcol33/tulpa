# n_obs x n_re_groups indicator design for an iid RE block.

`re_idx` is the 1-based per-observation group index. Observations whose
group falls outside `1:n_re_groups` contribute no RE column, matching
the `g >= 0 && g < n_re_groups` guard in the C++ kernels.

## Usage

``` r
.re_design(re_idx, n_re_groups, n_obs)
```
