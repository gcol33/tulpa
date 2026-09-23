# Outer Pareto-k-hat for a batched-target grid-integrated fit

Outer Pareto-k-hat reliability diagnostic for a grid-integrated nested
fit whose inner marginal comes from a BATCHED re-fit rather than a
per-sample closure: fits a Gaussian proposal at `(theta_hat, L_scale)`,
draws from it, evaluates the caller's batched log-target, applies a
radius cap with heavy-tail escalation, then reports `pareto_k` /
`is_ess` via the shared PSIS/GPD core. This is more than plain PSIS
smoothing over an already-computed log-ratio vector – see
[`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md)
for that – because it also builds the proposal, draws, and evaluates the
target.

## Usage

``` r
tulpa_batched_pareto_k(
  theta_hat,
  L_scale,
  log_target_batched,
  n_samples = .nl_diag("k_samples"),
  radius_cap = Inf,
  return_draws = FALSE,
  tail_points = NULL,
  Z = NULL
)
```

## Arguments

- theta_hat, L_scale:

  Mean and Cholesky scale of the Gaussian proposal
  `N(theta_hat, L_scale L_scale')`, in the integrator's own coordinate
  space (e.g. `(log range, log sigma)` for an SPDE field).

- log_target_batched:

  Function taking an `S x d` matrix of proposal draws and returning the
  integrator's unnormalized log posterior at each row, in that same
  coordinate space. Any change-of-variables Jacobian is the caller's
  responsibility.

- n_samples:

  Number of draws from the proposal.

- radius_cap:

  Whitened-radius cap on evaluated draws (`Inf` keeps every draw); draws
  past the cap are folded back in under heavy-tail escalation rather
  than dropped.

- return_draws:

  If `TRUE`, include the draws in the returned list.

- tail_points:

  Optional override for the PSIS tail size; `NULL` uses the automatic
  rule.

- Z:

  Optional pre-drawn `S x d` whitened sample; `NULL` draws it here.

## Value

A list with `pareto_k`, `is_ess`, `n_eval`, and `declined` (a reason
string, or `NULL` when the diagnostic ran to completion).

## See also

[`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md)
