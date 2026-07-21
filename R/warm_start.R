# ============================================================================
# Warm-starting the NUTS sampler from a Laplace / empirical-Bayes fit.
#
# NUTS starts at the origin with a structural inverse-mass diagonal, and spends
# warmup travelling to the typical set and learning the posterior's scaling. A
# Laplace or EB fit has already located the mode and estimated a curvature
# there, at a small fraction of the cost, so handing both over starts the chain
# where it would otherwise have to walk.
#
# Placement is by INDEX, not by name. Both the Laplace kernel and the sampler
# derive their parameter vector from the same compute_param_layout(), so a
# term's random effects occupy the same relative positions in both; the layout
# is asked for those positions (cpp_tulpa_glmm_layout) rather than reconstructed
# here. The one place the two layouts genuinely differ is zero inflation -- the
# Laplace path carries the ZI predictor as a second process appended to the
# fixed effects, while the sampler gives it its own beta_zi block -- so the ZI
# coefficients are moved across explicitly rather than assumed to line up.
#
# What is warm-started, and what is not:
#   * fixed effects, ZI coefficients, random-effect deviations -- from the mode.
#   * the inverse-mass diagonal for those slots -- from the fit's curvature
#     (solve(H_beta) for the fixed effects, the estimated variance component for
#     the random effects).
#   * variance-component hyperparameters (log_sigma_re) -- position from the
#     estimate, and inverse mass from the source fit's outer curvature when it
#     has one: tulpa_eb(marginal = TRUE) estimates H_theta at theta_hat and
#     carries its inverse as `theta_cov`, whose diagonal is the posterior
#     variance of each log sigma. Without that field the mass stays at 1 and
#     adapts, as it did before.
#   * a correlated term's Cholesky (L_re) -- neither position nor mass. Its
#     theta coordinates are log-Cholesky, where log L_ii is not log sigma_i
#     (sigma_i is the norm of row i of L), so neither the estimate nor its
#     curvature transfers to the sampler's parameterization.
#
# Chains are dispersed rather than stacked: seeding every chain at the same mode
# would leave Rhat comparing chains that started identically, which makes it
# read low for reasons that have nothing to do with convergence. Chain 1 starts
# at the mode and the rest are drawn around it at the fit's own scale, so the
# between-chain spread the diagnostic needs is present from the first iteration.
# Note that independent per-coordinate noise puts a chain at roughly
# sqrt(D) * sd from the mode, so on a 164-parameter model the dispersed chains
# start ~11 away from it -- further out than the origin a cold chain starts
# from. That costs nothing measurable (10-seed paired comparison below) and it
# is what keeps Rhat honest, which is why the dispersion is on by default.
#
# What warm-starting is and is not worth: over 10 seeds, redrawing both the data
# and the sampler seed, warm and cold were indistinguishable in effective sample
# size per second (paired Wilcoxon p = 0.32 dispersed, p = 0.49 stacked, with
# the source fit's own cost charged to the warm runs). It is offered as a
# capability, not as a speedup, and no claim of one belongs in the docs. Single
# runs of this comparison disagree with each other by 2-3x, because min(ESS) is
# an order statistic over parameters; anything measured here needs replication
# before it means anything.
#
# The kernel clamps whatever inverse mass it is handed to [1e-3, 1e3]
# (hmc_nuts_chain_setup.h), the same clamp it applies to its own structural
# metric, so a sharply identified coefficient whose posterior variance falls
# below 1e-3 starts from the clamp rather than from its true scale. That costs
# accuracy in the starting metric, not correctness: warmup adaptation runs from
# this point and converges to the same metric it would have found cold.
# ============================================================================


# Backends whose fits carry a mode and a curvature this can read.
#' @keywords internal
.WARM_START_SOURCES <- c("eb", "laplace")


#' Split a Laplace / EB mode vector into its semantic blocks
#'
#' Returns the count-side fixed effects, the zero-inflation coefficients, and
#' the per-term random-effect deviations, so the caller places each into the
#' sampler's corresponding block rather than relying on the two flat vectors
#' agreeing end to end (they do not, under zero inflation).
#'
#' @param fit A fitted `tulpa_fit` from `mode = "eb"` or `mode = "laplace"`.
#' @param re_terms The random-effect terms of the model being sampled, in
#'   `tulpa_laplace()` form. Taken from the caller rather than from the fit:
#'   `re_layout` is attached by the front door, so a source fit built directly
#'   by `tulpa_eb()` does not carry one, and the model being sampled is the
#'   authority on the block shapes in any case.
#' @keywords internal
.warm_start_blocks <- function(fit, re_terms) {
  mode_vec <- as.numeric(fit$mode)
  n_fixed <- as.integer(fit$n_fixed %||% 0L)
  fixed_names <- as.character(fit$fixed_names %||% character(0))

  if (length(mode_vec) < n_fixed) {
    stop("warm start: the source fit's mode is shorter than its fixed-effect ",
         "count; it is not a fit this can read.", call. = FALSE)
  }

  # The ZI columns are the ones the front door prefixed; everything before them
  # is the count side. They are contiguous and last within the fixed block.
  is_zi <- grepl("^zi_", fixed_names)
  beta <- mode_vec[seq_len(n_fixed)][!is_zi]
  beta_zi <- mode_vec[seq_len(n_fixed)][is_zi]

  re_flat <- if (length(mode_vec) > n_fixed) {
    mode_vec[seq.int(n_fixed + 1L, length(mode_vec))]
  } else numeric(0)

  # Per-term random effects, in the layout's term order. Each term occupies
  # n_groups * n_coefs consecutive slots.
  re_terms <- re_terms %||% list()
  re <- vector("list", length(re_terms))
  off <- 0L
  for (t in seq_along(re_terms)) {
    sz <- as.integer(re_terms[[t]]$n_groups) *
      as.integer(re_terms[[t]]$n_coefs %||% 1L)
    if (off + sz > length(re_flat)) {
      stop(sprintf(paste0(
        "warm start: the source fit carries %d random-effect value(s), but the ",
        "model being sampled needs at least %d for term %d alone. The warm ",
        "start must come from the same model."),
        length(re_flat), off + sz, t), call. = FALSE)
    }
    re[[t]] <- re_flat[seq.int(off + 1L, off + sz)]
    off <- off + sz
  }

  list(beta = beta, beta_zi = beta_zi, re = re)
}


#' Per-term random-effect SDs from a warm-start source fit
#'
#' An EB fit estimated them, so they are read off `map`. A Laplace fit
#' conditioned on values the caller supplied, so those are passed in.
#'
#' @param fit A fitted `tulpa_fit`.
#' @param sigma_re Fallback SDs, used when the fit did not estimate them.
#' @param n_terms Number of random-effect terms.
#' @keywords internal
.warm_start_sigmas <- function(fit, sigma_re, n_terms) {
  if (n_terms == 0L) return(list())

  # EB: `map` is one summary for a single block, a list of them otherwise.
  map <- fit$map
  if (!is.null(map)) {
    per_block <- if (!is.null(map$sigma)) list(map) else map
    if (length(per_block) == n_terms) {
      return(lapply(per_block, function(m) as.numeric(m$sigma)))
    }
  }

  if (is.null(sigma_re)) {
    stop("warm start: the source fit did not estimate the random-effect ",
         "standard deviations and none were supplied. Use `warm_start = \"eb\"`, ",
         "which estimates them, or pass `sigma_re`.", call. = FALSE)
  }
  s <- as.numeric(sigma_re)
  if (length(s) == 1L) s <- rep(s, n_terms)
  if (length(s) != n_terms) {
    stop(sprintf("warm start: `sigma_re` has length %d but the model has %d ",
                 length(s), n_terms), "random-effect terms.", call. = FALSE)
  }
  lapply(s, function(v) v)
}


#' Assemble NUTS initial positions and an inverse-mass diagonal
#'
#' @param fit Source fit (`mode = "eb"` or `"laplace"`).
#' @param layout The target sampler layout, from `cpp_tulpa_glmm_layout()`.
#' @param re_terms Random-effect terms of the model being sampled.
#' @param n_chains Number of chains to initialise.
#' @param sigma_re Random-effect SDs, when the source fit conditioned on them.
#' @param jitter Scale of the between-chain dispersion, as a multiple of each
#'   parameter's warm-start SD. `0` stacks every chain on the mode.
#' @param metric Hand the assembled inverse-mass diagonal to the kernel as well
#'   as the position. `FALSE` (default) because it can only be filled for some
#'   blocks today and a partial metric measurably slows the sampler; see the
#'   note at the end of this function.
#' @return `list(init = <n_chains x D matrix>, inv_metric_diag = <D vector or
#'   NULL>)`.
#' @keywords internal
.build_warm_start <- function(fit, layout, re_terms, n_chains, sigma_re = NULL,
                              jitter = 1.0, metric = FALSE) {
  D <- as.integer(layout$total_params)
  if (length(layout$unsupported) > 0L) {
    stop("warm start does not carry ", paste(layout$unsupported, collapse = " / "),
         " structure: a Laplace or EB fit does not estimate that block's ",
         "hyperparameters, so there is nothing to hand the sampler for it. ",
         "Drop `warm_start` for this model.", call. = FALSE)
  }

  blocks <- .warm_start_blocks(fit, re_terms)
  if (length(blocks$re) != length(layout$re_terms)) {
    stop(sprintf(paste0("warm start: the source fit has %d random-effect ",
                        "term(s) but the model being sampled lays out %d. The ",
                        "warm start must come from the same model."),
                 length(blocks$re), length(layout$re_terms)), call. = FALSE)
  }
  init <- rep(0.0, D)
  # Left at 1 wherever nothing is known, which is the sampler's own default, so
  # an unfilled slot adapts exactly as it would have without a warm start.
  inv_m <- rep(1.0, D)

  place <- function(vec, span, values, what) {
    if (is.null(span) || length(values) == 0L) return(vec)
    idx <- seq.int(span[1], span[2])
    if (length(idx) != length(values)) {
      stop(sprintf(paste0("warm start: the source fit supplies %d value(s) for ",
                          "the %s block, which the sampler lays out with %d ",
                          "slot(s)."), length(values), what, length(idx)),
           call. = FALSE)
    }
    vec[idx] <- values
    vec
  }

  # ---- Fixed effects ------------------------------------------------------
  # beta[[1]] is the count-side process; the sampler keeps zero inflation in its
  # own block rather than as a second process.
  beta_span <- if (length(layout$beta) > 0L) layout$beta[[1]] else NULL
  init <- place(init, beta_span, blocks$beta, "fixed-effect")

  # Marginal fixed-effect variances at the mode. H_beta is the fixed-effect
  # block of the joint precision, so its inverse is the conditional covariance
  # -- an understatement of the marginal, but the right order of magnitude for a
  # mass matrix, and warmup adapts from there.
  H_beta <- fit$H_beta
  if (!is.null(H_beta) && !is.null(beta_span)) {
    v <- tryCatch(diag(solve(H_beta)), error = function(e) NULL)
    if (!is.null(v) && all(is.finite(v)) && all(v > 0)) {
      n_beta <- length(blocks$beta)
      init_span <- seq.int(beta_span[1], beta_span[2])
      # H_beta spans the count side and, under zero inflation, the ZI tail too.
      inv_m[init_span] <- v[seq_len(min(n_beta, length(v)))]
    }
  }

  # ---- Zero-inflation coefficients ---------------------------------------
  init <- place(init, layout$beta_zi, blocks$beta_zi, "zero-inflation")
  if (!is.null(layout$beta_zi) && !is.null(H_beta)) {
    v <- tryCatch(diag(solve(H_beta)), error = function(e) NULL)
    n_beta <- length(blocks$beta)
    if (!is.null(v) && length(v) >= n_beta + length(blocks$beta_zi)) {
      zi_idx <- seq.int(layout$beta_zi[1], layout$beta_zi[2])
      vz <- v[seq.int(n_beta + 1L, n_beta + length(blocks$beta_zi))]
      if (all(is.finite(vz)) && all(vz > 0)) inv_m[zi_idx] <- vz
    }
  }

  # ---- Random effects and their variance components -----------------------
  sig <- .warm_start_sigmas(fit, sigma_re, length(layout$re_terms))
  hyper_var <- .warm_start_hyper_var(fit)
  for (t in seq_along(layout$re_terms)) {
    spec <- layout$re_terms[[t]]
    init <- place(init, spec$re, blocks$re[[t]],
                  sprintf("random-effect term %d", t))

    s_t <- as.numeric(sig[[t]])
    # The deviations are a priori N(0, sigma^2), and the posterior is tighter,
    # so the prior variance is a conservative mass scale for them.
    if (!is.null(spec$re) && length(s_t) > 0L) {
      re_idx <- seq.int(spec$re[1], spec$re[2])
      nc <- as.integer(spec$n_coefs)
      # Coefficient-minor within group, matching the layout's own striding.
      per_slot <- rep(if (length(s_t) == nc) s_t else rep(s_t[1], nc),
                      length.out = length(re_idx))
      inv_m[re_idx] <- per_slot^2
    }

    # log_sigma_re starts at the estimate, and takes its mass from the outer
    # curvature when the source fit estimated one. Without it the slot keeps the
    # adapting default.
    ls <- spec$log_sigma_re
    if (!is.null(ls)) {
      ls <- as.integer(ls)
      ls <- ls[!is.na(ls)]
      vals <- log(if (length(s_t) == length(ls)) s_t else rep(s_t[1], length(ls)))
      init[ls] <- vals
      hv <- if (!is.null(hyper_var) && length(hyper_var) >= t) hyper_var[[t]]
            else NULL
      if (!is.null(hv) && length(hv) == length(ls) && length(ls) > 0L) {
        inv_m[ls] <- hv
      }
    }
    # A correlated term's Cholesky is left at the layout default: the source
    # fit's Sigma is in log-Cholesky coordinates whose off-diagonal convention
    # is not the sampler's, and guessing it would seed a wrong correlation
    # rather than none.
  }

  # ---- Per-chain dispersion ----------------------------------------------
  # Dispersal uses the assembled scales whether or not the metric is handed to
  # the kernel: they are the best available estimate of each parameter's spread,
  # which is what a starting spread should be drawn at.
  init_mat <- matrix(init, nrow = n_chains, ncol = D, byrow = TRUE)
  if (n_chains > 1L && jitter > 0) {
    sd_vec <- sqrt(inv_m) * jitter
    for (cc in seq.int(2L, n_chains)) {
      init_mat[cc, ] <- init + stats::rnorm(D, 0, sd_vec)
    }
  }

  # The metric is deliberately NOT handed over by default, because it cannot
  # currently be composed from one kind of quantity. NUTS reads every entry as a
  # posterior variance and sets its step size from the worst-conditioned
  # direction, so the blocks have to be on a common footing. They are not:
  #
  #   beta          solve(H_beta) diagonal   a genuine posterior variance
  #   re            sigma^2                  the PRIOR variance, which overstates
  #                                          the posterior once a group has data
  #   log_sigma_re  1                        no estimate at all
  #
  # On a crossed random-intercept model that spanned 7550:1, with two of the
  # three blocks not describing what the kernel reads them as. That is a fact
  # about the vector, not a benchmark result -- replicated timing over 10 seeds
  # found no significant difference between warm and cold either way, so the
  # case for withholding it rests on the composition, not on speed.
  #
  # Both gaps now have real sources and neither is wired up here:
  # `return_joint_hessian` gives the random effects' posterior variances, and
  # `tulpa_eb(marginal = TRUE)` gives the variance components' outer curvature
  # (read by .warm_start_hyper_var below). Composing the metric from those three
  # and re-measuring is what would justify flipping this default.
  list(init = init_mat,
       inv_metric_diag = if (isTRUE(metric)) inv_m else NULL)
}


#' Resolve a `warm_start` request into sampler init / inverse-mass vectors
#'
#' Accepts the front door's `warm_start`: `"eb"` or `"laplace"` to run that fit
#' first, or an already-fitted object from either. Returns `NULL` when no warm
#' start was asked for, so the caller passes nothing and the kernel keeps its
#' own defaults.
#'
#' @param warm_start `NULL`, `"eb"`, `"laplace"`, or a fitted `tulpa_fit`.
#' @param args The assembled sampler arguments (the same values the kernel will
#'   receive), used both to run the source fit and to probe the target layout.
#' @param re_terms Random-effect terms in `tulpa_eb()` / `tulpa_laplace()` form.
#' @param sigma_re Random-effect SDs to condition on, for the Laplace source.
#' @param beta_prior Optional fixed-effect prior, threaded into the source fit.
#' @param n_chains Number of chains the sampler will run.
#' @keywords internal
.resolve_warm_start <- function(warm_start, args, re_terms, sigma_re,
                                beta_prior, n_chains) {
  if (is.null(warm_start)) return(NULL)

  # Validate the request before doing any work for it. Probing the layout first
  # would report a malformed `warm_start` as an error from the engine, about
  # arguments the caller never passed.
  if (is.character(warm_start)) {
    if (length(warm_start) != 1L || !warm_start %in% .WARM_START_SOURCES) {
      stop("`warm_start` must be NULL, ",
           paste(sprintf('"%s"', .WARM_START_SOURCES), collapse = " / "),
           ", or a fit from one of those modes; got \"",
           paste(warm_start, collapse = ", "), "\".", call. = FALSE)
    }
    if (length(re_terms) == 0L && identical(warm_start, "eb")) {
      stop("`warm_start = \"eb\"` needs at least one random-effect term to ",
           "estimate. Use `warm_start = \"laplace\"` for a fixed-effect model.",
           call. = FALSE)
    }
  } else if (inherits(warm_start, "tulpa_fit")) {
    bk <- warm_start$backend %||% ""
    if (!bk %in% .WARM_START_SOURCES) {
      stop("`warm_start` was given a fit from backend '", bk, "', which ",
           "carries no mode and curvature to hand over. Supply a fit from ",
           paste(sprintf('mode = "%s"', .WARM_START_SOURCES), collapse = " or "),
           ".", call. = FALSE)
    }
  } else {
    stop("`warm_start` must be NULL, a mode name, or a fitted tulpa_fit; got ",
         class(warm_start)[1], ".", call. = FALSE)
  }

  # The layout is asked of the engine with exactly the arguments the sampler
  # will get, so the vectors built against it cannot be a slot out.
  layout <- cpp_tulpa_glmm_layout(
    y = as.numeric(args$y),
    n_trials = as.integer(args$n_trials),
    X = args$X,
    family = args$family,
    phi = as.numeric(args$phi),
    sigma_beta = as.numeric(args$sigma_beta),
    offset_nullable = args$offset,
    re_spec = args$re_spec,
    spatial_spec = args$spatial_spec,
    temporal_spec = args$temporal_spec,
    sigma_re_scale = as.numeric(args$sigma_re_scale),
    fixed_names = args$fixed_names,
    svc_spec = args$svc_spec,
    tvc_spec = args$tvc_spec,
    zi_spec = args$zi_spec)

  if (is.character(warm_start)) {
    src <- if (identical(warm_start, "eb")) {
      tulpa_eb(y = args$y, n_trials = args$n_trials, X = args$X,
               re_terms = re_terms, family = args$family, phi = args$phi,
               beta_prior = beta_prior, offset = args$offset)
    } else {
      tulpa_laplace(y = args$y, n_trials = args$n_trials, X = args$X,
                    re_list = re_terms, family = args$family, phi = args$phi,
                    beta_prior = beta_prior, offset = args$offset,
                    X_zi = if (is.null(args$zi_spec)) NULL else args$zi_spec$X)
    }
  } else {
    src <- warm_start
  }

  .build_warm_start(src, layout, re_terms = re_terms, n_chains = n_chains,
                    sigma_re = sigma_re)
}
