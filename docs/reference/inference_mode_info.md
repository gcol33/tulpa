# Print inference mode information

Displays information about the inference modes available in tulpa,
including their tiers, guarantees, and appropriate use cases.

## Usage

``` r
inference_mode_info()
```

## Value

`NULL`, invisibly. Called for the side effect of printing the available
inference modes, their tiers, guarantees, and use cases to the console.

## Examples

``` r
inference_mode_info()
#> tulpa Inference Modes
#> ========================
#> 
#> Modes encode EPISTEMIC GUARANTEES, not just runtime characteristics.
#> 
#> TIER 1: EXACT
#>   Asymptotically correct posterior inference
#>   Guarantee: Credible intervals interpretable as posterior uncertainty
#>   Backends: hmc, ess, gibbs, re_cov_gibbs, smc, imh_laplace, mala
#>   Note: Reference standard
#> 
#> TIER 2: STRUCTURED
#>   Accurate inference conditional on structural assumptions
#>   Guarantee: Correct if model meets structural assumptions
#>   Backends: laplace, re_cov_nested, pathfinder, agq, ep, eb, nested_laplace, nested_laplace_joint, spde
#>   Note: Controlled approximation, not heuristics
#> 
#> TIER 3: OPTIMIZED
#>   Optimization-based approximate inference
#>   Guarantee: No general correctness guarantee
#>   Backends: sghmc, sgld, mclmc, vi
#>   Note: Uncertainty often underestimated; requires explicit opt-in
#> 
#> AUTO MODE
#>   Selects between Tier 1 and Tier 2 only.
#>   NEVER silently chooses Tier 3 (Optimized).
#>   Contract: 'Use the most reliable method expected to finish.'
#> 
#> Usage (by tier):
#>   tulpa(..., mode = 'auto')       # Tier 1 or 2 (default)
#>   tulpa(..., mode = 'exact')      # Tier 1
#>   tulpa(..., mode = 'structured') # Tier 2
#>   tulpa(..., mode = 'optimized')  # Tier 3 (explicit opt-in)
#> 
#> Backends ([R] = tulpa(mode =) dispatch, [direct] = has an R fitter but tulpa(mode =) refuses it -- call the fitter itself, [C-ABI] = model-package kernel only):
#>   [R]     mode = 'hmc'               # Tier 1 (Exact)
#>   [R]     mode = 'ess'               # Tier 1 (Exact)
#>   [R]     mode = 'gibbs'             # Tier 1 (Exact)
#>   [R]     mode = 're_cov_gibbs'      # Tier 1 (Exact)
#>   [R]     mode = 'smc'               # Tier 1 (Exact)
#>   [R]     mode = 'imh_laplace'       # Tier 1 (Exact)
#>   [R]     mode = 'mala'              # Tier 1 (Exact)
#>   [R]     mode = 'laplace'           # Tier 2 (Structured)
#>   [R]     mode = 're_cov_nested'     # Tier 2 (Structured)
#>   [R]     mode = 'pathfinder'        # Tier 2 (Structured)
#>   [R]     mode = 'agq'               # Tier 2 (Structured)
#>   [R]     mode = 'ep'                # Tier 2 (Structured)
#>   [R]     mode = 'eb'                # Tier 2 (Structured)
#>   [R]     mode = 'nested_laplace'    # Tier 2 (Structured)
#>   [direct] tulpa_nested_laplace_joint() # Tier 2 (Structured)
#>   [R]     mode = 'spde'              # Tier 2 (Structured)
#>   [R]     mode = 'sghmc'             # Tier 3 (Optimized)
#>   [R]     mode = 'sgld'              # Tier 3 (Optimized)
#>   [R]     mode = 'mclmc'             # Tier 3 (Optimized)
#>   [R]     mode = 'vi'                # Tier 3 (Optimized)
```
