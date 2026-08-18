# User-defined GMRF latent block, compiled C++ backend

`tgmrf_cpp()` is the compiled-C++ analogue of
[`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md). Where
[`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md) takes R
closures `Q(theta)` / `prior(theta)`, `tgmrf_cpp()` takes a user-written
`.cpp` file that defines the same kernels as templated C++ (so the same
source compiles against every AD type tulpa uses). The returned object
has the same S3 class as
[`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md) –
`c("tgmrf", "tulpa_latent_block")` – so every downstream consumer
(formula parser, inference layers, S3 methods) treats the two paths
identically. Only the dispatch on `Q(theta)` differs: registry-stored
function pointer (`tgmrf_cpp()`) vs Rcpp::Function callback
([`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md)).

The user `.cpp` file must include `<tulpa/tgmrf.h>` and call the
`TULPA_REGISTER_TGMRF(id, Q_fn, mu_fn, log_prior_fn)` macro once. See
`inst/examples/tgmrf_periodic_ar1.cpp` for a worked example.

## Usage

``` r
tgmrf_cpp(
  cpp_file,
  id,
  init,
  mu = NULL,
  graph = NULL,
  bounds = NULL,
  obs_idx = NULL,
  name = NULL,
  cache_dir = tulpa_cache_dir(),
  rebuild = FALSE
)
```

## Arguments

- cpp_file:

  Absolute path to the user's `.cpp` file. The file is compiled via
  [`Rcpp::sourceCpp()`](https://rdrr.io/pkg/Rcpp/man/sourceCpp.html)
  with caching keyed on
  `digest::sha256(file_contents) + TULPA_ABI_VERSION` so repeated calls
  with an unchanged source skip the rebuild.

- id:

  Character: the stable id used as the registry key. Must match the
  first argument of `TULPA_REGISTER_TGMRF` in the user's `.cpp`.

- init, mu, graph, bounds, obs_idx, name:

  Same arguments as
  [`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md).
  `init` is required and supplies the canonical theta names. `mu` is
  currently ignored on the C++ path – the mu kernel registered by
  `TULPA_REGISTER_TGMRF` is the source of truth; pass the argument here
  only to keep call sites symmetric with
  [`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md).

- cache_dir:

  Directory for Rcpp's `sourceCpp` cache. Defaults to a per-user
  location under `tools::R_user_dir("tulpa", "cache")`.

- rebuild:

  Force a recompile even if the cached DLL is up to date. Default
  `FALSE`.

## Value

An object of class `c("tgmrf", "tulpa_latent_block")` with the same
fields as
[`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md) plus
`backend = "cpp"` and `cpp_id = id`. The `Q` / `prior` fields are R
wrapper closures around the registered C++ kernels, so callers (e.g.
NUTS-over-theta) that hold the object can still call `block$Q(theta)`
directly.

## See also

[`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md) for the
R-closure path;
[`latent()`](https://gillescolling.com/tulpa/reference/latent.md) for
the formula slot that consumes a `tgmrf` block.

## Examples

``` r
if (FALSE) { # \dontrun{
# Compile a user latent block against tulpa's AD types; see
# inst/examples/tgmrf_periodic_ar1.cpp for a complete block.
blk <- tgmrf_cpp("my_block.cpp", id = "ar1", init = c(0, 0), graph = my_graph)
} # }
```
