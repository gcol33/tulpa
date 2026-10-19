# Information: tulpa

## Terms

- Hierarchical model: models observations through predictors and unobserved effects; introduced in the README opening, developed in the quickstart.
- MCMC: Markov chain Monte Carlo; expanded at first use in the README.
- Random intercept: a group-specific change in the intercept; illustrated by spray treatment in the README example.
- Posterior calibration: assessed through repeated simulations; home is vignettes/sbc.Rmd.

## Sources and defaults

- Package version and R requirement: DESCRIPTION; the README citation reads this metadata at render time.
- Citation author, title, URL, and key: inst/CITATION.
- Compiler standard and matrix libraries: src/Makevars and src/Makevars.win.
- OpenMP flags: SHLIB_OPENMP_CXXFLAGS in the build files; no package configure script.
- Thread controls: man/tulpa_nested_laplace.Rd and man/tulpa_nested_laplace_joint.Rd.
- Example observations: datasets::InsectSprays; generated estimates are evaluated during README rendering.
- Automatic inference selection: R/tulpa.R and vignettes/inference-modes.Rmd.

## Placement

- Inference tier tables and backend choices: vignettes/inference-modes.Rmd.
- Random-slope simulation and covariance details: vignettes/random-slopes.Rmd.
- Spatial and temporal structure catalogs: their respective articles.
- Calibration numbers and methodology: vignettes/sbc.Rmd.
- Custom-process comparisons and code: vignettes/tgmrf.Rmd.
- The README contains summaries and links to these homes.
