# Simulation-Based Diagnostics for tulpa Models

Model-checking tools based on posterior predictive simulation. These are
native R implementations equivalent to DHARMa's test suite, with no
external dependencies. They work with any model that provides
[`simulate()`](https://rdrr.io/r/stats/simulate.html),
[`fitted()`](https://rdrr.io/r/stats/fitted.values.html), and
[`residuals()`](https://rdrr.io/r/stats/residuals.html) methods.

## Value

The diagnostic functions documented in this family return their
individual results (a test-statistic object, a data frame of residuals,
or a `check_model` summary); see each function's own help page.
