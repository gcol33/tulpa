# Attach the standard `tulpa_fit` contract to a fitter's result.

The dispatch layer and the fitters that return generic-accessor-facing
objects route their return value through this helper (some
special-purpose fitters – the logpost samplers, tulpa_ep, the
categorical drivers – still stamp their class by hand), so a
directly-called fitter and a
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)-dispatched
one yield the same enriched object: the `tulpa_fit` class (so the
generic S3 methods – `coef` / `summary` / `vcov` / `confint` / `tidy` /
`glance` / `ranef` – dispatch), the fixed-effect layout (`n_fixed` /
`fixed_names` / `param_names`, which the `.fit_fixed_table` summary path
reads), and an explicit posterior-provenance tag (`draws_kind`) the
chain-vs-iid diagnostic gate (`.tulpa_draws_kind()`) reads to decide
whether Rhat/ESS apply. Each field is filled only when the fitter did
not already set it, so a fitter that knows better wins and the helper is
idempotent under
[`tulpa_dispatch()`](https://gillescolling.com/tulpa/reference/tulpa_dispatch.md).

## Usage

``` r
.finalize_fit(
  fit,
  backend = NULL,
  draws_kind = NULL,
  n_fixed = NULL,
  fixed_names = NULL,
  param_names = NULL,
  extra_class = NULL
)
```

## Arguments

- fit:

  The fitter result (a list); returned unchanged if not a list.

- backend:

  Backend key (sets `$backend`; supplies the default `draws_kind` via
  the registry `emits` property when it is a registry key).

- draws_kind:

  Explicit `"chain"` / `"iid"` / `"point"` tag; used when the backend is
  absent from `BACKEND_REGISTRY` or to override the registry.

- n_fixed, fixed_names, param_names:

  Fixed-effect layout, each filled only when the fitter left it unset.

- extra_class:

  Subclass(es) to prepend before `tulpa_fit`.

## Value

The enriched fit, classed `c(extra_class, ..., "tulpa_fit")`.

## Details

`draws_kind` precedence is: a value already on the fit, then the
explicit `draws_kind` argument, then the registry `emits` property for
`backend`. Backends that are not registry keys (the `tgmrf_*` fitters)
must pass `draws_kind` explicitly, since their `emits` cannot be looked
up.
