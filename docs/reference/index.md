# Package index

## Front door

The main fitter and its inference-mode / prior helpers.

- [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) : Fit
  a tulpa model
- [`inference_mode_info()`](https://gillescolling.com/tulpa/reference/inference_mode_info.md)
  : Print inference mode information
- [`tulpa_priors()`](https://gillescolling.com/tulpa/reference/tulpa_priors.md)
  : Prior specification for tulpa models

## Tier 2 – structured approximations

Laplace, nested Laplace, and the free-Sigma random-effect backends.

- [`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
  : Nested Laplace approximation for latent Gaussian models
- [`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)
  : Joint multi-likelihood nested Laplace approximation
- [`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
  : Nested-Laplace integration over random-effect covariances
- [`tulpa_em_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_em_laplace.md)
  : Fit a latent-variable model via EM + Laplace approximation
- [`tulpa_em_mc()`](https://gillescolling.com/tulpa/reference/tulpa_em_mc.md)
  : Generic Monte-Carlo EM driver
- [`agq_fit()`](https://gillescolling.com/tulpa/reference/agq_fit.md) :
  Adaptive Gauss-Hermite quadrature for one-RE GLMMs
- [`fit_spde()`](https://gillescolling.com/tulpa/reference/fit_spde.md)
  : Fit a Spatial Model using SPDE Laplace Approximation
- [`imh_laplace()`](https://gillescolling.com/tulpa/reference/imh_laplace.md)
  : Independence Metropolis-Hastings with Laplace proposal

## Tier 1 – exact MCMC and debias

- [`tulpa_gibbs()`](https://gillescolling.com/tulpa/reference/tulpa_gibbs.md)
  : Fit via Polya-Gamma Gibbs sampler
- [`tulpa_re_cov_gibbs()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_gibbs.md)
  : Gibbs estimation of random-effect covariances (exact-target debias)
- [`tulpa_nuts_beta()`](https://gillescolling.com/tulpa/reference/tulpa_nuts_beta.md)
  : Fit a beta-regression model via NUTS (joint sampling of beta +
  log_phi)
- [`tulpa_nuts_spde()`](https://gillescolling.com/tulpa/reference/tulpa_nuts_spde.md)
  : Sample an SPDE GLM via NUTS, optionally jointly over Matern hypers
- [`tulpa_tgmrf()`](https://gillescolling.com/tulpa/reference/tulpa_tgmrf.md)
  : Fit a custom tgmrf latent block
- [`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md) :
  User-defined GMRF latent block
- [`tgmrf_cpp()`](https://gillescolling.com/tulpa/reference/tgmrf_cpp.md)
  : User-defined GMRF latent block, compiled C++ backend
- [`mala()`](https://gillescolling.com/tulpa/reference/mala.md) :
  Metropolis-Adjusted Langevin Algorithm (MALA)

## Other approximations

- [`pathfinder()`](https://gillescolling.com/tulpa/reference/pathfinder.md)
  : Pathfinder: variational warm-start via L-BFGS + ELBO scoring
- [`bridge_sampling()`](https://gillescolling.com/tulpa/reference/bridge_sampling.md)
  : Bridge sampling for marginal likelihood

## Latent structure

Spatial, temporal, and varying-coefficient constructors.

- [`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md)
  : CAR / ICAR spatial structure
- [`spatial_bym2()`](https://gillescolling.com/tulpa/reference/spatial_bym2.md)
  : BYM2 spatial structure
- [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
  : Gaussian process spatial structure (NNGP)
- [`spatial_svc()`](https://gillescolling.com/tulpa/reference/spatial_svc.md)
  : Spatially varying coefficient structure
- [`spatial_rsr()`](https://gillescolling.com/tulpa/reference/spatial_rsr.md)
  : Restricted Spatial Regression (RSR)
- [`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md)
  : SPDE Spatial Field (Matern via Triangular Mesh)
- [`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md)
  : RW1 temporal structure (First-order Random Walk)
- [`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md)
  : RW2 temporal structure (Second-order Random Walk)
- [`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)
  : AR1 temporal structure (First-order Autoregressive)
- [`temporal_multiscale()`](https://gillescolling.com/tulpa/reference/temporal_multiscale.md)
  : Multi-scale temporal structure
- [`temporal_tvc()`](https://gillescolling.com/tulpa/reference/temporal_tvc.md)
  : Time-varying coefficient structure
- [`temporal_rtr()`](https://gillescolling.com/tulpa/reference/temporal_rtr.md)
  : Restricted temporal regression (RTR)
- [`spatiotemporal()`](https://gillescolling.com/tulpa/reference/spatiotemporal.md)
  : Spatiotemporal interaction specifications for tulpa
- [`svc()`](https://gillescolling.com/tulpa/reference/svc.md) : Extract
  spatially-varying coefficients from a fitted model
- [`tvc()`](https://gillescolling.com/tulpa/reference/tvc.md) : Extract
  temporally-varying coefficients from a fitted model
- [`latent()`](https://gillescolling.com/tulpa/reference/latent.md) :
  Mark an expression as a latent block in a tulpa formula
- [`latent_factor()`](https://gillescolling.com/tulpa/reference/latent_factor.md)
  : Create a latent factor specification
- [`adjacency()`](https://gillescolling.com/tulpa/reference/adjacency.md)
  : Construct a spatial adjacency graph for areal models

## Methods, criteria, and prediction

- [`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
  : Model criteria from a pointwise log-likelihood
- [`tulpa_kfold()`](https://gillescolling.com/tulpa/reference/tulpa_kfold.md)
  : K-fold cross-validation for a tulpa fit
- [`tulpa_reloo()`](https://gillescolling.com/tulpa/reference/tulpa_reloo.md)
  : Selective refit of high-Pareto-k observations (reloo)
- [`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md)
  : Pareto-smoothed importance sampling
- [`bayes_R2()`](https://gillescolling.com/tulpa/reference/bayes_R2.md)
  : Bayesian R-squared
- [`tulpa_powerscale_sensitivity()`](https://gillescolling.com/tulpa/reference/tulpa_powerscale_sensitivity.md)
  : Power-scaling prior / likelihood sensitivity
- [`compare_models()`](https://gillescolling.com/tulpa/reference/compare_models.md)
  : Compare models by information criteria
- [`model_average()`](https://gillescolling.com/tulpa/reference/model_average.md)
  : Model-averaged predictions
- [`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md)
  : Posterior predictive replicates
- [`pp_check()`](https://gillescolling.com/tulpa/reference/pp_check.md)
  : Posterior predictive check
- [`prior_predict()`](https://gillescolling.com/tulpa/reference/prior_predict.md)
  : Prior predictive simulation

## Diagnostics

- [`mcmc_diagnostics()`](https://gillescolling.com/tulpa/reference/mcmc_diagnostics.md)
  **\[deprecated\]** : MCMC convergence diagnostics
- [`check_diagnostics()`](https://gillescolling.com/tulpa/reference/check_diagnostics.md)
  : Quick convergence check
- [`diagnostic_summary()`](https://gillescolling.com/tulpa/reference/diagnostic_summary.md)
  : Comprehensive Diagnostic Summary
- [`laplace_diagnostics()`](https://gillescolling.com/tulpa/reference/laplace_diagnostics.md)
  **\[deprecated\]** : Approximation-reliability diagnostics for a
  deterministic nested-Laplace fit
- [`geweke_test()`](https://gillescolling.com/tulpa/reference/geweke_test.md)
  : Geweke Convergence Test
- [`moran_i()`](https://gillescolling.com/tulpa/reference/moran_i.md) :
  Moran's I test for spatial autocorrelation in residuals
- [`durbin_watson()`](https://gillescolling.com/tulpa/reference/durbin_watson.md)
  : Durbin-Watson test for temporal autocorrelation
- [`tulpa_variogram()`](https://gillescolling.com/tulpa/reference/tulpa_variogram.md)
  : Empirical semivariogram of residuals
- [`pit_residuals()`](https://gillescolling.com/tulpa/reference/pit_residuals.md)
  : PIT (Probability Integral Transform) residuals
- [`check_model()`](https://gillescolling.com/tulpa/reference/check_model.md)
  : Diagnostic panel plot
- [`spatial_range()`](https://gillescolling.com/tulpa/reference/spatial_range.md)
  : Extract spatial range and variance from a fitted spatial model
- [`temporal_corr()`](https://gillescolling.com/tulpa/reference/temporal_corr.md)
  : Extract temporal correlation parameters from a fitted model
- [`post_hoc_lm()`](https://gillescolling.com/tulpa/reference/post_hoc_lm.md)
  : Fit a post-hoc linear model on estimated parameters
