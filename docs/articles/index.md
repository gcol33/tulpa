# Articles

### Concepts

- [Concepts](https://gillescolling.com/tulpa/articles/concepts.md):

  The vocabulary the other articles use, each term defined on a small
  fit, followed by the fitted object, its accessors and the rules the
  fitting functions enforce.

### Models

- [Random
  slopes](https://gillescolling.com/tulpa/articles/random-slopes.md):

  Random slopes with a correlated or diagonal covariance: fit, check,
  extract per-group effects, predict and compare.

- [Spatial
  models](https://gillescolling.com/tulpa/articles/spatial-models.md):

  Areal and continuous spatial fields end to end: simulate, fit, check,
  map, predict and compare with a non-spatial fit.

- [Temporal
  models](https://gillescolling.com/tulpa/articles/temporal-models.md):

  Random-walk and autoregressive time trends end to end: simulate, fit,
  check, predict and compare with a fit without a time term.

- [Space-time
  models](https://gillescolling.com/tulpa/articles/space-time.md):

  Areal fields that change over time: additive space plus time,
  Knorr-Held interactions as Kronecker blocks, maps per time slice,
  forecasts, and which door fits which model.

- [Varying
  coefficients](https://gillescolling.com/tulpa/articles/varying-coefficients.md):

  Coefficients that change over space or time: spatially varying
  coefficients on a slope or the intercept, time-varying coefficients on
  a grid or at irregular times, checked against a constant coefficient,
  a parametric interaction and two priors.

- [Continuous fields with the
  SPDE](https://gillescolling.com/tulpa/articles/spde.md):

  Matern fields on a triangular mesh in depth: the SPDE model, mesh
  construction and resolution, PC priors, mean and SD maps on a grid,
  and a comparison with non-spatial, NNGP and HSGP fits.

- [Spatial confounding and restricted spatial
  regression](https://gillescolling.com/tulpa/articles/spatial-confounding.md):

  What a spatial field does to the slope of a spatially smooth
  covariate, the two slopes a model can target, and how spatial_rsr()
  switches between them on areal and NNGP fields.

- [Joint
  models](https://gillescolling.com/tulpa/articles/joint-models.md):

  Two response arms sharing one spatial field through a copy scale:
  simulate, fit, read the copy scale and its point mass at zero, compare
  with separate fits, predict where one arm has no data.

- [Multivariate areal
  fields](https://gillescolling.com/tulpa/articles/multivariate-spatial.md):

  Two responses with correlated spatial fields on one graph (MCAR) or
  correlated unit effects without a graph (MIID): simulate, fit, recover
  the cross-response correlation, predict where one response has no
  data, and compare with two independent fits.

- [Latent
  factors](https://gillescolling.com/tulpa/articles/latent-factors.md):

  Many species counted at the same sites, sharing two latent site
  factors with a loading per species: simulate a long-format data set,
  fit it with tulpa(factors = ), and read loadings, the covariance
  between species they imply, and the site scores.

- [Categorical and ordinal
  responses](https://gillescolling.com/tulpa/articles/categorical-responses.md):

  Responses that fall into one of several classes: the baseline-category
  logit for unordered classes and the cumulative logit for ordered ones,
  each simulated, fitted, checked, predicted and compared.

- [Censored, truncated and zero-inflated
  responses](https://gillescolling.com/tulpa/articles/censored-responses.md):

  Fit measurements below a detection limit as censored, samples above a
  reporting ceiling as truncated, and counts with excess zeros as
  zero-inflated, each against a known truth and a naive Gaussian or
  Poisson fit.

### Workflows

- [Data
  formatting](https://gillescolling.com/tulpa/articles/data-formatting.md):

  The input tulpa() expects, built from a data frame, with random-effect
  terms, binomial trials, offsets and the validation errors.

- [Priors](https://gillescolling.com/tulpa/articles/priors.md):

  The prior families, the defaults, priors on fixed effects and
  hyperparameters, and prior predictive checks.

- [Diagnostics](https://gillescolling.com/tulpa/articles/diagnostics.md):

  One Poisson areal model checked through every diagnostic tulpa offers:
  simulated residuals, spatial and temporal structure, sampler
  convergence, approximation reliability, model comparison and prior
  sensitivity, with a misspecified twin that shows what failing looks
  like.

- [Prediction and
  forecasting](https://gillescolling.com/tulpa/articles/prediction.md):

  Predict for fitted and new groups, forecast by masking the response,
  predict at new locations and map the result, each checked against
  held-out simulated values.

- [Model
  comparison](https://gillescolling.com/tulpa/articles/model-comparison.md):

  Compare fits by marginal likelihood, Bayes factors, WAIC and LOO, and
  average models for prediction.

- [Sequential Monte Carlo and
  WBIC](https://gillescolling.com/tulpa/articles/smc-wbic.md):

  Fit with the sequential Monte Carlo sampler, check its draws against a
  closed-form posterior and NUTS, and rank models by WBIC.

- [Troubleshooting](https://gillescolling.com/tulpa/articles/troubleshooting.md):

  The errors and warnings a tulpa fit most often stops on, grouped by
  stage, each with the message as printed and a corrected call that
  runs.

- [Checkpointing](https://gillescolling.com/tulpa/articles/checkpoint.md):

  Save the progress of a long fit to disk, resume it after an
  interruption, and see the fingerprint refuse a file written for other
  inputs.

- [Custom latent
  blocks](https://gillescolling.com/tulpa/articles/tgmrf.md):

  Write a latent block in R or C++ and fit it at every inference tier.

- [EM +
  Laplace](https://gillescolling.com/tulpa/articles/em-laplace.md):

  Fit a model with a discrete latent variable through EM + Laplace, pool
  multiple imputations, or sample its joint posterior with true-density
  MH-within-Gibbs.

### Case studies

- [Case study: sudden infant deaths in North
  Carolina](https://gillescolling.com/tulpa/articles/case-study-sids.md):

  An applied areal analysis on real county data: expected counts as an
  offset, non-spatial, ICAR and BYM2 fits, risk maps with exceedance
  probabilities, residual Moran’s I, a prior check and the limits of an
  ecological question.

### Background

- [Inference
  modes](https://gillescolling.com/tulpa/articles/inference-modes.md):

  The three inference tiers, how mode = “auto” chooses between them, and
  what each backend costs and guarantees.

- [Identifiability](https://gillescolling.com/tulpa/articles/identifiability.md):

  The maths of each alias and weakly identified hyperparameter behind
  tulpa’s fields and random effects, with stress tests that show how
  many regions, locations or groups recovery needs.

- [Reliability
  diagnostics](https://gillescolling.com/tulpa/articles/reliability-pareto-k.md):

  Read the Pareto k-hat that scores whether a deterministic
  approximation can be trusted, and what to do when it is high.

- [Simulation-based
  calibration](https://gillescolling.com/tulpa/articles/sbc.md):

  Check that an inference algorithm is calibrated across the generative
  model with simulation-based calibration.

- [Compatibility with lme4, glmmTMB, brms and
  INLA](https://gillescolling.com/tulpa/articles/compatibility.md):

  The same models fitted in tulpa and in lme4, glmmTMB, brms and INLA
  with matched settings and priors: measured agreement, a syntax map,
  the documented differences, timings, and when the other package fits
  the task better.
