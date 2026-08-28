# Package index

## Front door

The main fitter, its formula surface, and the mode / family helpers.

- [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) : Fit
  a tulpa model
- [`tulpa_parse_formula()`](https://gillescolling.com/tulpa/reference/tulpa_parse_formula.md)
  : Parse a mixed-model formula
- [`findbars()`](https://gillescolling.com/tulpa/reference/findbars.md)
  : Find all bar terms in a formula's parse tree
- [`nobars()`](https://gillescolling.com/tulpa/reference/nobars.md) :
  Remove all bar terms from a formula's parse tree
- [`inference_mode_info()`](https://gillescolling.com/tulpa/reference/inference_mode_info.md)
  : Print inference mode information
- [`validate_mode()`](https://gillescolling.com/tulpa/reference/validate_mode.md)
  : Validate that a fit used the expected mode
- [`tulpa_family()`](https://gillescolling.com/tulpa/reference/tulpa_family.md)
  : Construct a minimal tulpa_family for simulation
- [`tulpa_gaussian()`](https://gillescolling.com/tulpa/reference/tulpa_gaussian.md)
  : Fit a Gaussian linear model via tulpa's generic engine
- [`tulpa_build_model_data()`](https://gillescolling.com/tulpa/reference/tulpa_build_model_data.md)
  : Build model matrices from a parsed formula
- [`tulpa_simulate()`](https://gillescolling.com/tulpa/reference/tulpa_simulate.md)
  : Simulate data from a tulpa model

## Tier 2 – structured approximations

Laplace, nested Laplace, and the free-Sigma random-effect backends.

- [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  : Fit a model via Laplace approximation
- [`tulpa_laplace_beta()`](https://gillescolling.com/tulpa/reference/tulpa_laplace_beta.md)
  : Fit a beta-regression model via Laplace, estimating the precision
- [`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
  : Nested Laplace approximation for latent Gaussian models
- [`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)
  : Joint multi-likelihood nested Laplace approximation
- [`fit_st_nested()`](https://gillescolling.com/tulpa/reference/fit_st_nested.md)
  : Fit an additive spatiotemporal GLM by nested Laplace
- [`tulpa_hyper_grid()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_grid.md)
  : Outer hyperparameter-grid integration with a user-supplied inner fit
- [`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
  : Nested-Laplace integration over random-effect covariances
- [`tulpa_re_aghq()`](https://gillescolling.com/tulpa/reference/tulpa_re_aghq.md)
  : Adaptive Gauss-Hermite refinement of a grouped random-effect
  covariance
- [`tulpa_eb()`](https://gillescolling.com/tulpa/reference/tulpa_eb.md)
  : Empirical-Bayes random-effect covariances
- [`tulpa_em_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_em_laplace.md)
  : Fit a latent-variable model via EM + Laplace approximation
- [`tulpa_em_mc()`](https://gillescolling.com/tulpa/reference/tulpa_em_mc.md)
  : Generic Monte-Carlo EM driver
- [`agq_fit()`](https://gillescolling.com/tulpa/reference/agq_fit.md) :
  Adaptive Gauss-Hermite quadrature for one-RE GLMMs
- [`fit_spde()`](https://gillescolling.com/tulpa/reference/fit_spde.md)
  : Fit a Spatial Model using SPDE Laplace Approximation
- [`tulpa_ep()`](https://gillescolling.com/tulpa/reference/tulpa_ep.md)
  : Expectation-Propagation fit for a GLM
- [`tulpa_multinomial()`](https://gillescolling.com/tulpa/reference/tulpa_multinomial.md)
  : Multinomial (nominal K-class) logistic regression via Laplace
- [`tulpa_ordinal()`](https://gillescolling.com/tulpa/reference/tulpa_ordinal.md)
  : Ordinal (ordered K-class) cumulative-logit regression via Laplace
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

- [`tulpa_cache_dir()`](https://gillescolling.com/tulpa/reference/tulpa_cache_dir.md)
  :

  Default cache directory for
  [`tgmrf_cpp()`](https://gillescolling.com/tulpa/reference/tgmrf_cpp.md)-compiled
  DLLs

- [`tulpa_cache_clear()`](https://gillescolling.com/tulpa/reference/tulpa_cache_clear.md)
  : Remove compiled blocks from the tgmrf_cpp() cache

- [`mala()`](https://gillescolling.com/tulpa/reference/mala.md) :
  Metropolis-Adjusted Langevin Algorithm (MALA)

- [`tulpa_integrator()`](https://gillescolling.com/tulpa/reference/tulpa_integrator.md)
  : Select the symplectic integrator for HMC and NUTS

- [`with_tulpa_integrator()`](https://gillescolling.com/tulpa/reference/with_tulpa_integrator.md)
  : Run an expression under a chosen symplectic integrator

## Other approximations

- [`pathfinder()`](https://gillescolling.com/tulpa/reference/pathfinder.md)
  : Pathfinder: variational warm-start via L-BFGS + ELBO scoring
- [`bridge_sampling()`](https://gillescolling.com/tulpa/reference/bridge_sampling.md)
  : Bridge sampling for marginal likelihood

## Latent structure

Spatial, temporal, and varying-coefficient constructors.

- [`spatial()`](https://gillescolling.com/tulpa/reference/spatial.md) :
  Areal spatially varying coefficient field
- [`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md)
  : CAR / ICAR spatial structure
- [`spatial_car_proper()`](https://gillescolling.com/tulpa/reference/spatial_car_proper.md)
  : Proper CAR spatial structure
- [`spatial_bym2()`](https://gillescolling.com/tulpa/reference/spatial_bym2.md)
  : BYM2 spatial structure
- [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
  : Gaussian process spatial structure (NNGP)
- [`spatial_multiscale()`](https://gillescolling.com/tulpa/reference/spatial_multiscale.md)
  : Multi-Scale Gaussian Process spatial structure
- [`spatial_svc()`](https://gillescolling.com/tulpa/reference/spatial_svc.md)
  : Spatially varying coefficient structure
- [`spatial_rsr()`](https://gillescolling.com/tulpa/reference/spatial_rsr.md)
  : Restricted Spatial Regression (RSR)
- [`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md)
  : SPDE Spatial Field (Matern via Triangular Mesh)
- [`spatial_spde_custom()`](https://gillescolling.com/tulpa/reference/spatial_spde_custom.md)
  : SPDE Spatial Field from Custom Matrices
- [`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md)
  : RW1 temporal structure (First-order Random Walk)
- [`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md)
  : RW2 temporal structure (Second-order Random Walk)
- [`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)
  : AR1 temporal structure (First-order Autoregressive)
- [`temporal_ar2()`](https://gillescolling.com/tulpa/reference/temporal_ar2.md)
  : AR(2) temporal latent field (second-order autoregressive)
- [`temporal_ar()`](https://gillescolling.com/tulpa/reference/temporal_ar.md)
  : AR(p) temporal latent field (general-order autoregressive)
- [`temporal_gp()`](https://gillescolling.com/tulpa/reference/temporal_gp.md)
  : Gaussian Process temporal structure
- [`temporal_multiscale()`](https://gillescolling.com/tulpa/reference/temporal_multiscale.md)
  : Multi-scale temporal structure
- [`temporal_tvc()`](https://gillescolling.com/tulpa/reference/temporal_tvc.md)
  : Time-varying coefficient structure
- [`temporal_rtr()`](https://gillescolling.com/tulpa/reference/temporal_rtr.md)
  : Restricted temporal regression (RTR)
- [`spatiotemporal()`](https://gillescolling.com/tulpa/reference/spatiotemporal.md)
  : Spatiotemporal interaction specifications for tulpa
- [`spatiotemporal_gp()`](https://gillescolling.com/tulpa/reference/spatiotemporal_gp.md)
  : Non-separable spatiotemporal GP
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
- [`check_adjacency()`](https://gillescolling.com/tulpa/reference/check_adjacency.md)
  : Validate a spatial adjacency matrix
- [`node_index()`](https://gillescolling.com/tulpa/reference/node_index.md)
  : Map cell identifiers to graph node indices
- [`compute_nngp_neighbors()`](https://gillescolling.com/tulpa/reference/compute_nngp_neighbors.md)
  : Compute nearest neighbors for NNGP
- [`tulpa_bar_field_specs()`](https://gillescolling.com/tulpa/reference/tulpa_bar_field_specs.md)
  : Expand a varying-coefficient bar into per-column field specs
- [`tulpa_bar_field_replicate()`](https://gillescolling.com/tulpa/reference/tulpa_bar_field_replicate.md)
  : Replicate an areal graph across the levels of a factor (replicated
  CAR)
- [`tulpa_is_spatial_bar()`](https://gillescolling.com/tulpa/reference/tulpa_is_spatial_bar.md)
  : Recognize an inline varying-coefficient bar

## Priors

The prior surface and the individual prior builders.

- [`tulpa_priors()`](https://gillescolling.com/tulpa/reference/tulpa_priors.md)
  : Prior specification for tulpa models

- [`priors_default()`](https://gillescolling.com/tulpa/reference/priors_default.md)
  : Show default priors for a tulpa family

- [`prior_from_spec()`](https://gillescolling.com/tulpa/reference/prior_from_spec.md)
  :

  Build a `prior` list for
  [`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
  from a tulpa spec object

- [`prior_normal()`](https://gillescolling.com/tulpa/reference/prior_normal.md)
  : Normal prior

- [`prior_beta()`](https://gillescolling.com/tulpa/reference/prior_beta.md)
  : Beta prior

- [`prior_gamma()`](https://gillescolling.com/tulpa/reference/prior_gamma.md)
  : Gamma prior

- [`prior_exponential()`](https://gillescolling.com/tulpa/reference/prior_exponential.md)
  : Exponential prior

- [`prior_half_cauchy()`](https://gillescolling.com/tulpa/reference/prior_half_cauchy.md)
  : Half-Cauchy prior

- [`prior_half_normal()`](https://gillescolling.com/tulpa/reference/prior_half_normal.md)
  : Half-normal prior

- [`prior_pc()`](https://gillescolling.com/tulpa/reference/prior_pc.md)
  : Penalized complexity (PC) prior

- [`re_cov_pc_lkj_prior()`](https://gillescolling.com/tulpa/reference/re_cov_pc_lkj_prior.md)
  : PC + LKJ hyperprior for a random-effect covariance

## Methods, criteria, and prediction

- [`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
  : Model criteria from a pointwise log-likelihood
- [`dic()`](https://gillescolling.com/tulpa/reference/criteria_doors.md)
  [`cpo()`](https://gillescolling.com/tulpa/reference/criteria_doors.md)
  : DIC and CPO
- [`tulpa_kfold()`](https://gillescolling.com/tulpa/reference/tulpa_kfold.md)
  : K-fold cross-validation for a tulpa fit
- [`tulpa_reloo()`](https://gillescolling.com/tulpa/reference/tulpa_reloo.md)
  : Selective refit of high-Pareto-k observations (reloo)
- [`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md)
  : Pareto-smoothed importance sampling
- [`tulpa_loglik()`](https://gillescolling.com/tulpa/reference/tulpa_loglik.md)
  : Streaming pointwise log-likelihood
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
- [`reexports`](https://gillescolling.com/tulpa/reference/reexports.md)
  [`tidy`](https://gillescolling.com/tulpa/reference/reexports.md)
  [`glance`](https://gillescolling.com/tulpa/reference/reexports.md) :
  Objects exported from other packages

## Reading a fit

Accessors for a fitted model: the coefficient surface, the posterior
draws in whatever representation the backend produced, and the extracted
latent effects.

- [`fixef()`](https://gillescolling.com/tulpa/reference/fixef.md) :
  Fixed-effect coefficients (lme4-compatible)
- [`ranef()`](https://gillescolling.com/tulpa/reference/ranef.md) :
  Random-effect summaries
- [`VarCorr()`](https://gillescolling.com/tulpa/reference/VarCorr.md) :
  Random-effect variances and correlations
- [`posterior_sample()`](https://gillescolling.com/tulpa/reference/posterior_sample.md)
  : Posterior parameter sample from a fit
- [`mcmc_draws()`](https://gillescolling.com/tulpa/reference/mcmc_draws.md)
  : MCMC chain draws from a fit
- [`tulpa_posterior_draws()`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md)
  : Posterior draws from a nested-Laplace fit
- [`tulpa_draws_array()`](https://gillescolling.com/tulpa/reference/tulpa_draws_array.md)
  : Posterior draws as a 3D array
- [`as_draws()`](https://gillescolling.com/tulpa/reference/as_draws.md)
  [`as_draws_array()`](https://gillescolling.com/tulpa/reference/as_draws.md)
  [`as_draws_matrix()`](https://gillescolling.com/tulpa/reference/as_draws.md)
  [`as_draws_df()`](https://gillescolling.com/tulpa/reference/as_draws.md)
  [`as_draws_rvars()`](https://gillescolling.com/tulpa/reference/as_draws.md)
  : Posterior draws in the posterior package's format
- [`temporal()`](https://gillescolling.com/tulpa/reference/temporal.md)
  : Extract temporal effects from a fitted model
- [`spatiotemporal_effects()`](https://gillescolling.com/tulpa/reference/spatiotemporal_effects.md)
  : Extract spatiotemporal effects from fitted model
- [`smooth_effects()`](https://gillescolling.com/tulpa/reference/smooth_effects.md)
  : Extract fitted covariate smooths
- [`latent_factors()`](https://gillescolling.com/tulpa/reference/latent_factors.md)
  : Extract latent factor posteriors from fit

## Diagnostics

Convergence for sampled fits, approximation reliability for
deterministic ones, and the residual / goodness-of-fit surface both
share.

- [`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
  : Posterior diagnostics for a fitted model
- [`mcmc_diagnostics()`](https://gillescolling.com/tulpa/reference/mcmc_diagnostics.md)
  **\[deprecated\]** : MCMC convergence diagnostics
- [`laplace_diagnostics()`](https://gillescolling.com/tulpa/reference/laplace_diagnostics.md)
  **\[deprecated\]** : Approximation-reliability diagnostics for a
  deterministic nested-Laplace fit
- [`check_diagnostics()`](https://gillescolling.com/tulpa/reference/check_diagnostics.md)
  : Quick convergence check
- [`diagnostic_summary()`](https://gillescolling.com/tulpa/reference/diagnostic_summary.md)
  : Comprehensive Diagnostic Summary
- [`select_main_params()`](https://gillescolling.com/tulpa/reference/select_main_params.md)
  : Select the "main" model parameters for diagnostic display
- [`n_divergent()`](https://gillescolling.com/tulpa/reference/n_divergent.md)
  : Number of divergent transitions
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
- [`tulpa_pit()`](https://gillescolling.com/tulpa/reference/tulpa_pit.md)
  : Probability integral transform from a predictive CDF
- [`test_uniformity()`](https://gillescolling.com/tulpa/reference/test_uniformity.md)
  : Test uniformity of PIT residuals
- [`test_dispersion()`](https://gillescolling.com/tulpa/reference/test_dispersion.md)
  : Test for over- or underdispersion
- [`test_outliers()`](https://gillescolling.com/tulpa/reference/test_outliers.md)
  : Test for outliers (simulation envelope)
- [`test_zero_inflation()`](https://gillescolling.com/tulpa/reference/test_zero_inflation.md)
  : Test for zero inflation
- [`check_model()`](https://gillescolling.com/tulpa/reference/check_model.md)
  : Diagnostic panel plot
- [`spatial_range()`](https://gillescolling.com/tulpa/reference/spatial_range.md)
  : Extract spatial range and variance from a fitted spatial model
- [`temporal_corr()`](https://gillescolling.com/tulpa/reference/temporal_corr.md)
  : Extract temporal correlation parameters from a fitted model
- [`post_hoc_lm()`](https://gillescolling.com/tulpa/reference/post_hoc_lm.md)
  : Fit a post-hoc linear model on estimated parameters

## Calibration

Simulation-based calibration. Where the diagnostics above score one fit,
these score whether an inference algorithm’s posteriors are calibrated
across the generative model, by reading the whole marginal CDF.

- [`sbc()`](https://gillescolling.com/tulpa/reference/sbc.md)
  [`summary(`*`<sbc>`*`)`](https://gillescolling.com/tulpa/reference/sbc.md)
  [`plot(`*`<sbc>`*`)`](https://gillescolling.com/tulpa/reference/sbc.md)
  : Simulation-based calibration
- [`sbc_mixture()`](https://gillescolling.com/tulpa/reference/sbc_predictive.md)
  [`sbc_normal()`](https://gillescolling.com/tulpa/reference/sbc_predictive.md)
  [`sbc_discrete()`](https://gillescolling.com/tulpa/reference/sbc_predictive.md)
  [`sbc_rank()`](https://gillescolling.com/tulpa/reference/sbc_predictive.md)
  [`sbc_draws()`](https://gillescolling.com/tulpa/reference/sbc_predictive.md)
  : Predictive shapes an SBC fitter reports

## Plots

- [`plot_rhat()`](https://gillescolling.com/tulpa/reference/plot_rhat.md)
  : Plot Rhat Convergence Diagnostic
- [`plot_ess()`](https://gillescolling.com/tulpa/reference/plot_ess.md)
  : Plot Effective Sample Size Diagnostic
- [`plot_acf()`](https://gillescolling.com/tulpa/reference/plot_acf.md)
  : Plot Autocorrelation Functions
- [`plot_energy()`](https://gillescolling.com/tulpa/reference/plot_energy.md)
  : Plot Energy Diagnostic (E-BFMI)
- [`plot_divergences()`](https://gillescolling.com/tulpa/reference/plot_divergences.md)
  : Plot Divergent Transitions
- [`plot_pairs()`](https://gillescolling.com/tulpa/reference/plot_pairs.md)
  : Plot Bivariate Parameter Posteriors (Pairs Plot)
- [`plot_diagnostics()`](https://gillescolling.com/tulpa/reference/plot_diagnostics.md)
  : Diagnostic Plotting Functions for tulpa Models
- [`plot_map()`](https://gillescolling.com/tulpa/reference/plot_map.md)
  : Plot spatial predictions as a map
- [`plot_map_panel()`](https://gillescolling.com/tulpa/reference/plot_map_panel.md)
  : Plot multiple maps in a grid

## Outer-grid integration and utilities

The node designs the outer hyperparameter integration is laid on, the
axis-provenance surface, and the small numerical helpers.

- [`ccd_grid()`](https://gillescolling.com/tulpa/reference/ccd_grid.md)
  : Central Composite Design (CCD) grid for nested-Laplace integration

- [`ccd_weights()`](https://gillescolling.com/tulpa/reference/ccd_weights.md)
  : Corrected R-INLA CCD integration weights

- [`ccd_to_theta()`](https://gillescolling.com/tulpa/reference/ccd_to_theta.md)
  : Map standardised CCD coordinates to physical hyperparameters

- [`hyper_axis_spec()`](https://gillescolling.com/tulpa/reference/hyper_axis_spec.md)
  : Describe one outer-grid hyperparameter axis

- [`auto_grid()`](https://gillescolling.com/tulpa/reference/auto_grid.md)
  : Mark an outer-grid setting as a default rather than a pin

- [`is_auto_grid()`](https://gillescolling.com/tulpa/reference/is_auto_grid.md)
  : Is an outer-grid setting marked as a default?

- [`rubins_pool()`](https://gillescolling.com/tulpa/reference/rubins_pool.md)
  : Pool multiple imputation draws via Rubin's rules

- [`sn_cdf()`](https://gillescolling.com/tulpa/reference/sn_cdf.md) :
  Skew-normal CDF

- [`sn_quantile()`](https://gillescolling.com/tulpa/reference/sn_quantile.md)
  : Skew-normal quantile

- [`sn_match()`](https://gillescolling.com/tulpa/reference/sn_match.md)
  : Match three cumulants to a skew-normal parameterisation

- [`tulpa_profile()`](https://gillescolling.com/tulpa/reference/tulpa_profile.md)
  : Profile the inner Laplace solve by phase

- [`tulpa_check_control()`](https://gillescolling.com/tulpa/reference/tulpa_check_control.md)
  :

  Validate a `control = list()` surface against its canonical key set
