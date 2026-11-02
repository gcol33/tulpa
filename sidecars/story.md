# Story: tulpa

## Claim

I provide a Bayesian hierarchical modelling engine that combines latent structures with several inference methods through one formula interface.

## Audience

R users fitting models with grouped, spatial, or temporal observations. The README assumes familiarity with data frames and introduces the fitting workflow.

## Relation

- to: none
- phrase: Companion packages supply observation likelihoods and reuse tulpa's inference and latent structures.
- home: vignettes/em-laplace.Rmd

## Arcs

### A1 Specify the model

- question: How do I represent grouping, smooth effects, and dependence?
- answer: Formula terms and structure specifications describe the effects.
- evidence: man/tulpa.Rd; vignettes/random-slopes.Rmd
- home: vignettes/tulpa.Rmd
- also: README, with links to spatial and temporal articles

### A2 Choose and assess inference

- question: Which inference method ran, and what can its diagnostics establish?
- answer: The fit records the method; diagnostics distinguish chains and approximations.
- evidence: vignettes/inference-modes.Rmd; vignettes/reliability-pareto-k.Rmd
- limits: Per-fit diagnostics and repeated-simulation calibration answer different questions.
- home: vignettes/inference-modes.Rmd
- also: README, with links to reliability and calibration articles

### A3 Extend the model

- question: How do I supply a custom process or observation model?
- answer: R and C++ interfaces expose custom latent processes; companion packages supply observation likelihoods.
- evidence: vignettes/tgmrf.Rmd; vignettes/em-laplace.Rmd
- home: vignettes/tgmrf.Rmd
- also: README

### A4 Check, compare and trust a fit

- question: Does the model fit the data, which model is better, and does the result agree with established packages?
- answer: Residual, sampler and approximation diagnostics, model comparison, and matched fits against lme4, glmmTMB, brms and INLA.
- evidence: vignettes/articles/diagnostics.Rmd; vignettes/articles/model-comparison.Rmd; vignettes/articles/compatibility.Rmd; vignettes/articles/identifiability.Rmd
- limits: Diagnostics on a field model depend on whether residuals condition on the field.
- home: vignettes/articles/diagnostics.Rmd
- also: vignettes/articles/troubleshooting.Rmd, vignettes/articles/case-study-sids.Rmd

## Documents

- README.Rmd / README.md: overview, installation, first model, routing, help, and citation.
- vignettes/tulpa.Rmd: Get started, a tutorial through model fitting with a taste of each feature.
- Concepts (articles/concepts): the vocabulary and the fitted object.
- Models: random-slopes, spatial-models, temporal-models, space-time, varying-coefficients, spde, spatial-confounding, joint-models, latent-factors, categorical-responses; one model type each, simulate to compare.
- Workflows: data-formatting, priors, diagnostics, model-comparison, troubleshooting, checkpoint, tgmrf, em-laplace; one task each.
- Case studies: case-study-sids, real county data end to end.
- Background: inference-modes, identifiability, reliability-pareto-k, sbc, compatibility.
- Help pages: argument and return-value contracts.
