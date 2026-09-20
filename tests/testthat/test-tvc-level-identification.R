# A TVC term's level is CENTRED, not penalised (gcol33/tulpa#844).
#
# A temporally-varying coefficient contributes `eta_i += x_ij w_{j,g(i)}(t_i)`,
# so `w -> w + c` together with `beta_j -> beta_j - c` leaves eta exactly
# unchanged whatever the covariate. The TVC block identified that direction
# with `tvc_sum_to_zero_penalty()`, the soft instrument the SVC path was moved
# off in 34c9cb5b -- a penalty leaves the aliased direction IN the likelihood
# and stiffens it, at a precision (`s2z_precision(n_times)`) unrelated to the
# field's own `1/sigma^2`.
#
# A sampler can traverse a stiff direction; a diagonal-ish Gaussian variational
# family cannot. That is the asymmetry #844 measured: the same fixture
# recovered under HMC and collapsed to a sixth of its amplitude under VI.
#
# The construction now follows tulpa/sum_to_zero.h, as the SVC path does.
# An intrinsic block (rw1, rw2) is AUGMENTED -- its constant direction carries
# the field's own tau, and the rank rises by one pin -- and a proper one (ar1)
# is not, because its constant direction already has a prior. Both are CENTRED
# on the way into eta, which is what removes the alias.

.tvcl_sim <- function(seed = 3L, Tn = 40L, per = 4L, rho = 0.8, sd_i = 0.35) {
  set.seed(seed)
  v <- numeric(Tn); v[1] <- rnorm(1, 0, 0.5)
  for (t in 2:Tn) v[t] <- rho * v[t - 1] + rnorm(1, 0, sd_i)
  v <- v - mean(v)
  ti <- rep(seq_len(Tn), each = per)
  d <- data.frame(tidx = ti, x = rnorm(length(ti)))
  s <- v[ti]
  d$y <- rpois(nrow(d), exp(0.3 + 0.5 * d$x + s))
  list(d = d, s = s)
}

# The fitted field in eta: the eta draws with the fixed-effect part removed.
.tvcl_field <- function(fit, d) {
  colMeans(tulpa:::.tulpa_eta_draws(fit, ndraws = 200L, synth_seed = 1L)) -
    as.numeric(cbind(1, d$x) %*% coef(fit)[c("(Intercept)", "x")])
}


test_that("a TVC field under VI recovers the amplitude its RW1 sibling does", {
  skip_if_not_slow()
  # The regression this file exists for. Measured before the construction
  # changed, on this exact fixture: the TVC arm read cor 0.344 / sd 0.040 at
  # the default stopping rule and 0.694 / 0.077 with the rule relaxed, against
  # a truth of 0.472 and an RW1 comparator at 0.842 / 0.460. The gap was the
  # penalty, not the stopping rule (gcol33/tulpa#821 had already relaxed that).
  S <- .tvcl_sim()
  d <- S$d
  ctl <- list(seed = 1L)
  f_tvc <- suppressWarnings(suppressMessages(tulpa(
    y ~ x, data = d, family = "poisson", mode = "vi",
    temporal = temporal_tvc("tidx"), control = ctl)))
  f_rw1 <- suppressWarnings(suppressMessages(tulpa(
    y ~ x, data = d, family = "poisson", mode = "vi",
    temporal = temporal_rw1("tidx"), control = ctl)))

  e_tvc <- .tvcl_field(f_tvc, d)
  e_rw1 <- .tvcl_field(f_rw1, d)

  # Against the truth, and against the sibling block that never had a penalty.
  expect_gt(cor(e_tvc, S$s), 0.7)
  expect_gt(sd(e_tvc), 0.5 * sd(S$s))
  expect_lt(sd(e_tvc), 1.5 * sd(S$s))
  # The two blocks fit the same field here, so a collapse in one and not the
  # other is the signal. Before the fix this ratio was 0.087.
  expect_gt(sd(e_tvc) / sd(e_rw1), 0.7)
})


test_that("every TVC structure keeps its amplitude under VI", {
  skip_if_not_slow()
  # ar1 is PROPER, so it takes the centring and no augmentation; rw1 and rw2
  # are intrinsic and take both. All three collapsed under the penalty.
  S <- .tvcl_sim()
  d <- S$d
  for (st in c("rw1", "rw2", "ar1")) {
    f <- suppressWarnings(suppressMessages(tulpa(
      y ~ x, data = d, family = "poisson", mode = "vi",
      temporal = temporal_tvc("tidx", structure = st),
      control = list(seed = 1L))))
    e <- .tvcl_field(f, d)
    expect_gt(cor(e, S$s), 0.7, label = paste0("cor(", st, ")"))
    expect_gt(sd(e), 0.5 * sd(S$s), label = paste0("sd(", st, ")"))
  }
})


test_that("a longer VI run does not report a worse ELBO than a shorter one", {
  skip_if_not_slow()
  # The penalty term dominated the objective, so relaxing the stopping rule
  # took the reported ELBO from -689.64 to -1256.94 -- a longer run scoring
  # worse than a shorter one on the same model. With the aliased direction
  # removed from the likelihood instead of stiffened, the ordering is the one
  # a stopping rule can be read against.
  S <- .tvcl_sim()
  spec <- list(y ~ x, data = S$d, family = "poisson", mode = "vi",
               temporal = temporal_tvc("tidx"))
  short <- suppressWarnings(suppressMessages(do.call(tulpa,
    c(spec, list(control = list(seed = 1L))))))
  long <- suppressWarnings(suppressMessages(do.call(tulpa,
    c(spec, list(control = list(seed = 1L, vi_max_iter = 3000L,
                                vi_tol_rel_elbo = 0, vi_patience = 500L))))))

  expect_gt(long$vi_iterations, short$vi_iterations)
  expect_gt(long$elbo, short$elbo - 5)
})


test_that("the stored TVC draws are the field the likelihood saw", {
  skip_if_not_slow()
  # The centring happens on the way into eta, so a stored draw that was not
  # centred would report a different field than the one scored -- the same
  # contract the SVC store carries (gcol33/tulpa#822's neighbourhood).
  S <- .tvcl_sim()
  fit <- suppressWarnings(suppressMessages(tulpa(
    y ~ x, data = S$d, family = "poisson", mode = "hmc",
    temporal = temporal_tvc("tidx"),
    control = list(n_iter = 400L, n_warmup = 250L, seed = 1L))))

  cols <- grep("^tvc_w\\[", colnames(fit$draws))
  skip_if(length(cols) == 0L, "fit carries no tvc_w draws")
  # One block, one term: every draw sums to zero over the time index.
  blk <- fit$draws[, cols, drop = FALSE]
  expect_lt(max(abs(rowSums(blk))), 1e-8)
})
