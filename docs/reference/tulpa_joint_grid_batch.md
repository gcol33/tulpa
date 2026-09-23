# Fit multiple joint nested-Laplace models through one fused grid solve

Fits `B` responses that share one design (arms, latent blocks, outer
grid layout, solver settings) and differ only in their responses, their
arm dispersions and the nodes of any dispersion axis, through one fused
grid solve (`cpp_nested_laplace_joint_multi_batch`): one design pass per
outer- grid cell for all `B` responses, `B` block-diagonal Newton
solves, instead of `B` independent fits each repeating the design pass.

## Usage

``` r
tulpa_joint_grid_batch(fits)
```

## Arguments

- fits:

  A non-empty list of functions of no arguments, each performing one
  ordinary joint nested-Laplace fit.

## Value

A list of fit results, one per element of `fits`, in order.

## Details

Each element of `fits` is a function of no arguments that performs one
ordinary fit (a
[`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)
call, or a consumer front door that makes one). The fits run twice: the
first run stops each fit at its main outer-grid solve and keeps the
kernel request; the fused solve answers every request at once; the
second run replays each fit with its main grid solve served from the
fused result. Every other kernel call a fit makes (placement,
refinement, diagnostics) runs as it always does, so each returned fit is
the object its ordinary fit would have returned, built by the same code.

The fits replay one after another from the random number state the batch
was called at, so they draw the same stream the same fits called in
sequence would draw; each capture starts from that state too, and a fit
whose grid request depended on a draw taken before it is refused at its
replay rather than served. Warnings and messages a capture run raises
are muffled; the replay raises them again.

Raises a `tulpa_grid_batch_ineligible` condition (catchable with
`tryCatch(..., tulpa_grid_batch_ineligible = ...)`) when the fits do not
share a design or a request uses a setting the fused driver does not
carry. The random number state is restored to where the batch was called
before that condition leaves, so a caller that falls back to fitting the
responses one at a time draws the stream it would have drawn without the
batch.

## See also

[`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)
