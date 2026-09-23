# Mark an outer-grid setting as a default rather than a pin

Declares that a setting shaping the outer hyperparameter grid carries a
*default* the caller computed, not a choice the user made. The
auto-recenter pass (`outer_grid_placement`) leaves a user-pinned setting
exactly as given, and re-centres (or, for a prior, engages its own
regularizer over) a marked one when the fit rails against its ceiling.

Four kinds of setting take the mark:

- a grid axis on a nested-Laplace `prior` block (`sigma_grid`,
  `tau_grid`, ...) – a numeric vector of nodes, or the `[n_cells x k]`
  matrix of pre-paired coordinates the families whose axis is a matrix
  take (`mcar` / `miid`'s `logchol_grid`, `tgmrf`'s `theta_grid_built`);

- an entry of
  [`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)'s
  `phi_grid` – one arm's dispersion axis;

- a scalar grid-construction knob in `control`, for a driver that builds
  its axes rather than taking them
  ([`fit_st_nested()`](https://gillescolling.com/tulpa/reference/fit_st_nested.md)'s
  `n_grid_spatial`, `tau_upper`, ...);

- a `prior_sigma` hyperprior specification – a list, e.g.
  `list("pc.prec", c(U = 3, alpha = 0.01))`.

Wrapper packages are the intended caller: one that builds a default of
its own – because it derives a second axis from it, hands the same
vector to several blocks, or exposes its own argument with a default –
would otherwise be indistinguishable from a user who pinned that setting
deliberately. Mark it and the rescue stays live. A setting whose value
is exactly the engine's own default is recognised without a mark;
anything else needs one – and a dispersion axis ALWAYS needs one, since
the engine has no default of its own to recognise it against.

The mark is an attribute, so it is dropped by
[`sort()`](https://rdrr.io/r/base/sort.html), `[`,
[`c()`](https://rdrr.io/r/base/c.html) and
[`as.numeric()`](https://rdrr.io/r/base/numeric.html): build the value
first, mark it last.

## Usage

``` r
auto_grid(x, place = TRUE)
```

## Arguments

- x:

  Numeric vector or matrix of grid nodes, a numeric scalar knob, or a
  prior-specification list.

- place:

  May the auto-placement pass move this axis onto its own posterior?
  `TRUE` (default) declares a default the engine may re-place; `FALSE`
  declares one it must integrate as written.

## Value

`x` carrying the marker attribute. Numeric input is coerced to double IN
PLACE, so everything else it carries –
[`dim()`](https://rdrr.io/r/base/dim.html) and
[`dimnames()`](https://rdrr.io/r/base/dimnames.html) above all –
survives the mark; a list is returned unchanged apart from the
attribute.

## Declaring a default the engine must integrate as written

`place = FALSE` separates the two questions the mark otherwise answers
at once. Provenance – whose choice these nodes are – and placement
policy – whether the pass may move them – are different questions, and a
package that measured its own default as the one to integrate has an
answer to the second that is not "the user pinned it". Such an axis is
treated exactly as a pin everywhere the engine ACTS on it (the placement
pass leaves it, refinement densifies within its span rather than
following the posterior past the end nodes, and no curvature is computed
for it), and differs only in what the fit REPORTS:
`"default_axis_pinned"` rather than `"axis_pinned"`, so a reader is not
told they pinned an axis they never wrote down. Leaving the mark off
instead buys the same integration and says the wrong thing about it.

`place` is read wherever the engine would otherwise RE-PLACE the
setting: a grid axis, and a scalar grid-construction knob. A
`prior_sigma` specification is not re-placed but replaced, by the
engine's own regularizer, and carries no `place`.

## See also

[`is_auto_grid()`](https://gillescolling.com/tulpa/reference/is_auto_grid.md),
[`auto_grid_place()`](https://gillescolling.com/tulpa/reference/auto_grid_place.md),
[`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md),
[`fit_st_nested()`](https://gillescolling.com/tulpa/reference/fit_st_nested.md)

## Examples

``` r
prior <- list(type = "icar", sigma_grid = auto_grid(c(0.1, 0.5, 1, 2, 3)))
is_auto_grid(prior$sigma_grid)
#> [1] TRUE
is_auto_grid(auto_grid(list("pc.prec", c(U = 3, alpha = 0.01))))
#> [1] TRUE
auto_grid_place(auto_grid(c(0.5, 1, 2), place = FALSE))
#> [1] FALSE
```
