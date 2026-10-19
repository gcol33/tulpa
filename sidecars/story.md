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
- home: vignettes/quickstart.Rmd
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

## Documents

- README.Rmd / README.md: overview, installation, first model, routing, help, and citation.
- vignettes/quickstart.Rmd: a tutorial through model fitting.
- Topic articles: complete workflows, methods, and limitations.
- Help pages: argument and return-value contracts.
