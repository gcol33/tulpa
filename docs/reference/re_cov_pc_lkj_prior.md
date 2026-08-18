# PC + LKJ hyperprior for a random-effect covariance

Construct the default weakly-informative hyperprior used by
[`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
for one covariance block: independent Penalized-Complexity (PC) priors
on the marginal standard deviations `sigma_i` together with an LKJ prior
on the correlation matrix `R` (correlated block) or no correlation
(diagonal block), returned as a `log_prior_theta` function in the
block's integration coordinates.

## Usage

``` r
re_cov_pc_lkj_prior(
  n_coefs,
  prior_sigma = c(3, 0.05),
  eta = 2,
  correlated = TRUE
)
```

## Arguments

- n_coefs:

  Number of coefficients `c` in the RE block.

- prior_sigma:

  `c(U, alpha)` giving `P(sigma_i > U) = alpha` (default `c(3, 0.05)`),
  applied independently to every marginal SD.

- eta:

  LKJ shape (default 2). `eta = 1` is uniform on correlation matrices;
  larger values favour weaker correlations. Ignored for a diagonal
  block.

- correlated:

  `TRUE` (default) for a full covariance block (log-Cholesky
  coordinates, LKJ prior); `FALSE` for a diagonal / uncorrelated block
  (log-SD coordinates, no correlation). For `n_coefs = 1` the two
  coincide.

## Value

A `function(theta)` returning the scalar log prior density in the
block's integration coordinates, suitable for one block of the
`log_prior_theta` argument of
[`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md).

## Details

For a correlated block the prior is specified on the natural scale,
`p(sigma, R) = LKJ(R | eta) * prod_i PC(sigma_i)`, then pushed to the
log-Cholesky coordinates `theta` of `Sigma = L L'` by the exact
change-of-variables Jacobian. For a diagonal (uncorrelated) block the
LKJ factor drops and `theta_i = log sigma_i` with Jacobian
`sum_i theta_i`.

PC prior (Simpson et al. 2017) on each marginal SD: exponential with
rate `lambda = -log(alpha) / U`, so `P(sigma_i > U) = alpha` – the
`prior_sigma = c(U, alpha)` convention also used by the SPDE prior in
tulpa.

LKJ prior (Lewandowski et al. 2009) on the correlation matrix: `p(R)`
proportional to `det(R)^(eta - 1)`. `eta = 1` is uniform over
correlation matrices; `eta > 1` concentrates toward the identity. The
normalizing constant is dropped (constant across the grid, so it cancels
when the integration weights are renormalized).

Jacobian (correlated block): with `theta` packing `log L_ii` on the
diagonal and the raw strict-lower entries of `L`, the change of
variables from `(sigma, R)` to `theta` adds
`sum_i (c + 2 - i) * log L_ii - c * sum_i log sigma_i` to
`log p(sigma, R)`. (Composition of the log-diagonal map, the standard
Cholesky-to-covariance Jacobian `2^c prod_i L_ii^(c+1-i)`, and the
covariance-to-`(sigma, R)` Jacobian; verified against numerical
differentiation in `test-re-cov-prior.R`.)

## See also

[`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
