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

A list with `pareto_k`, `is_ess` and `n_eval` (the number of draws whose
log importance ratio was finite). When the diagnostic ran to completion
it also carries `lr`, those `n_eval` finite log importance ratios in
draw order (what `pareto_k` was fitted on, so
`tulpa_psis(lr, tail_points)$pareto_k` reproduces it), and with
`return_draws = TRUE` the evaluated draws `U` (`n_eval x d`) and their
smoothed `log_weights`. When it declined, `pareto_k` and `is_ess` are
`NA` and `declined` names the reason (e.g. `"draws_too_few"`).

## See also

[`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md)

## Examples

``` r
set.seed(1)
# A Gaussian target a little wider than the standard-normal proposal:
lt <- function(U) -0.5 * rowSums(sweep(U, 2L, c(1.2, 0.8), "/")^2)
k <- tulpa_batched_pareto_k(theta_hat = c(0, 0), L_scale = diag(2),
                            log_target_batched = lt, n_samples = 500)
k$pareto_k
#> [1] 0.3034994
```
