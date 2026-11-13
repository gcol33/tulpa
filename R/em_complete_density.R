# Controls for the conditional MH kernel and retained data-augmentation chain.
.em_gibbs_control <- function(n_gibbs, control) {
  if (!is.numeric(n_gibbs) || length(n_gibbs) != 1L || !is.finite(n_gibbs) ||
      n_gibbs < 2 || n_gibbs != floor(n_gibbs)) {
    stop("`n_gibbs` must be an integer >= 2.", call. = FALSE)
  }
  defaults <- list(warmup = n_gibbs %/% 2L, thin = 1L, mh_steps = 1L, scale = 1)
  if (!is.list(control) || (length(control) &&
      (is.null(names(control)) || any(!nzchar(names(control))) ||
       anyDuplicated(names(control)) || any(!names(control) %in% names(defaults))))) {
    stop("`correction_control` accepts warmup, thin, mh_steps and scale.",
         call. = FALSE)
  }
  control <- utils::modifyList(defaults, control)
  .check_run_length(n_gibbs, control$warmup, "Gibbs correction")
  for (nm in c("thin", "mh_steps")) {
    value <- control[[nm]]
    if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
        value < 1 || value != floor(value)) {
      stop("`correction_control$", nm, "` must be a positive whole number.",
           call. = FALSE)
    }
  }
  if (!is.numeric(control$scale) || length(control$scale) != 1L ||
      !is.finite(control$scale) || control$scale <= 0) {
    stop("`correction_control$scale` must be positive and finite.", call. = FALSE)
  }
  control
}

.em_check_exact_callbacks <- function(log_complete, m_step_extra = NULL) {
  if (!is.null(log_complete) && !is.function(log_complete)) {
    stop("`log_complete` must be NULL or a function(fits, z, ...).", call. = FALSE)
  }
  if (!is.null(m_step_extra)) {
    stop("correction = 'gibbs' requires fixed block hyperparameters; ",
         "deterministic `m_step_extra` updates are not posterior transitions.",
         call. = FALSE)
  }
}

# Terms and coefficient positions identify the same parameters for every Z.
.em_complete_layout <- function(blocks) {
  if (is.null(names(blocks)) || anyNA(names(blocks)) ||
      any(!nzchar(names(blocks))) || anyDuplicated(names(blocks))) {
    stop("Gibbs correction needs unique non-empty block names.", call. = FALSE)
  }
  n_fixed <- vapply(blocks, function(b) ncol(b$X), integer(1))
  terms <- lapply(blocks, function(b) {
    if (!is.null(b$prior) || !is.null(b$spatial) || !is.null(b$likelihood)) {
      stop("Gibbs correction supports complete-data GLMM blocks with fixed ",
           "dispersion and RE covariance; nested, spatial and custom-likelihood ",
           "blocks need a full conditional parameter proposal.", call. = FALSE)
    }
    lapply(b$re_list, function(r) {
      list(n_groups = as.integer(r$n_groups), n_coefs = as.integer(r$n_coefs %||% 1L))
    })
  })
  widths <- n_fixed + vapply(terms, function(tt) {
    as.integer(sum(vapply(tt, function(t) t$n_groups * t$n_coefs, integer(1))))
  }, integer(1))
  coef_names <- lapply(seq_along(blocks), function(k) {
    b <- blocks[[k]]
    fixed <- if (n_fixed[k]) colnames(b$X) %||% paste0("beta", seq_len(n_fixed[k]))
      else character(0)
    random <- unlist(lapply(seq_along(terms[[k]]), function(j) {
      term <- terms[[k]][[j]]
      paste0("re", j, ":g", rep(seq_len(term$n_groups), each = term$n_coefs),
             ":c", rep(seq_len(term$n_coefs), term$n_groups))
    }), use.names = FALSE)
    paste0(names(blocks)[k], ":", c(fixed, random))
  })
  param_names <- unlist(coef_names, use.names = FALSE)
  if (anyDuplicated(param_names)) {
    stop("Gibbs correction needs unique coefficient names within each block.",
         call. = FALSE)
  }
  list(block_names = names(blocks), n_fixed = n_fixed, widths = widths,
       terms = terms, param_names = param_names,
       offsets = c(0L, cumsum(widths)))
}

# The E-step and model density read sampled values from mode and beta.
.em_fits_at <- function(theta, layout) {
  fits <- lapply(seq_along(layout$widths), function(k) {
    value <- theta[layout$offsets[k] + seq_len(layout$widths[k])]
    list(mode = unname(value), beta = unname(value[seq_len(layout$n_fixed[k])]))
  })
  stats::setNames(fits, layout$block_names)
}

# Normalized complete-data GLMM density, using the Newton engine's native
# family likelihood and the same Gaussian coefficient and RE priors.
.em_complete_density <- function(blocks, beta_prior = NULL) {
  layout <- .em_complete_layout(blocks)
  densities <- lapply(blocks, function(b) {
    p <- ncol(b$X)
    bp <- .normalize_beta_prior(b$beta_prior %||% beta_prior, p) %||%
      list(mean = rep(0, p), sd = rep(.tulpa_prior_sd("laplace"), p))
    if (any(!is.finite(bp$sd))) {
      stop("The normalized complete-data density requires proper finite beta prior SDs.",
           call. = FALSE)
    }
    weight <- b$weights %||% rep(1, length(b$y))
    if (anyNA(weight) || any(!weight %in% c(0, 1))) {
      stop("The complete-data density requires hard 0/1 observation weights; ",
           "supply `log_complete` for a different normalized joint model.", call. = FALSE)
    }
    active <- which(weight == 1)
    design <- .laplace_count_design(b$X, NULL, b$re_list, length(b$y))
    design <- design[active, , drop = FALSE]
    offset <- (b$offset %||% rep(0, length(b$y)))[active]
    nt <- rep_len(b$n_trials %||% 1L, length(b$y))[active]
    family <- .canonical_family(b$family)
    phi <- .phi_to_kernel(family, b$phi %||% 1)
    re_priors <- lapply(b$re_list, function(r) {
      nc <- r$n_coefs %||% 1L
      packed <- .re_cov_pack(r)
      if (length(packed) == nc) {
        L <- diag(packed, nc)
      } else {
        L <- matrix(0, nc, nc)
        L[lower.tri(L, diag = TRUE)] <- packed
      }
      if (any(!is.finite(L)) || any(diag(L) <= 0)) {
        stop("Complete-data RE priors need a finite positive-definite covariance.",
             call. = FALSE)
      }
      list(L = L, n_groups = r$n_groups, width = nc * r$n_groups,
           log_normalizer = -nc / 2 * log(2 * pi) - sum(log(diag(L))))
    })
    function(value) {
      eta <- as.matrix(design %*% value) + offset
      ll <- cpp_family_loglik_batch(as.numeric(b$y[active]), as.integer(nt),
                                     eta, family, phi)
      lp <- sum(stats::dnorm(value[seq_len(p)], bp$mean, bp$sd, log = TRUE))
      start <- p
      for (r in re_priors) {
        u <- matrix(value[start + seq_len(r$width)], nrow(r$L), r$n_groups)
        standardized <- forwardsolve(r$L, u)
        lp <- lp + r$n_groups * r$log_normalizer - sum(standardized^2) / 2
        start <- start + r$width
      }
      as.numeric(ll + lp)
    }
  })
  function(theta) {
    if (!is.numeric(theta) || length(theta) != sum(layout$widths) ||
        any(!is.finite(theta))) return(-Inf)
    sum(vapply(seq_along(densities), function(k) {
      densities[[k]](theta[layout$offsets[k] + seq_len(layout$widths[k])])
    }, numeric(1)))
  }
}
