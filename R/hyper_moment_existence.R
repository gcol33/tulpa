# If pi(r) is proportional to r^(-1-a) and L(r) to r^(-b),
# E[r^p] is finite exactly when p < a + b. Equality diverges.
.hyper_moment_rule <- function(model, power = 1, logarithm = FALSE) {
  limit <- model$prior_power + model$likelihood_power
  finite_support <- isTRUE(model$fixed) || isTRUE(model$bounded)
  exists <- function(p) {
    if (finite_support || logarithm || p <= 0) TRUE
    else if (length(limit) != 1L || is.na(limit)) NA else p < limit
  }
  m <- exists(power); s <- exists(2 * power)
  reason <- if (isTRUE(model$fixed)) "fixed parameter"
    else if (isTRUE(model$bounded)) "bounded prior support"
    else if (logarithm) "finite log-range moments under the range prior"
    else if (length(limit) != 1L || is.na(limit)) "likelihood boundary limit unclassified"
    else paste0("range prior tail r^(-1-", model$prior_power,
      "), likelihood boundary decay r^(-", model$likelihood_power,
      "); moment order must be < ", limit)
  c(model, list(moment_limit = limit, power = power, logarithm = logarithm,
                mean_exists = m, sd_exists = s,
                reason = reason))
}

.hyper_moment_read <- function(model, mean, sd, power = 1, logarithm = FALSE) {
  if (is.null(model)) return(list(mean = mean, sd = sd))
  rule <- .hyper_moment_rule(model, power, logarithm)
  if (identical(rule$mean_exists, FALSE)) mean <- Inf
  if (identical(rule$sd_exists, FALSE)) sd <- Inf
  list(mean = mean, sd = sd, existence = rule)
}

.hyper_summary_attributes <- function(out, rows) {
  known <- vapply(rows, function(r) !is.null(r$existence), TRUE)
  if (any(known)) {
    for (key in c("existence", "log_moments")) {
      label <- if (key == "existence") "moment_existence" else key
      attr(out, label) <- stats::setNames(lapply(rows[known], `[[`, key), rownames(out)[known])
    }
  }
  out
}

# Standard observation densities are positive at finite linear predictors.
# NNGP covariances have a finite limit; each finite HSGP basis vanishes there.
.hyper_positive_boundary <- function(object) {
  families <- if (length(object$responses))
    vapply(object$responses, function(a) a$family %||% "", character(1))
    else object$family %||% ""
  all(nzchar(families)) && all(vapply(families, .family_base, character(1)) %in%
                               union(family_names(), .COUNT_FAMILIES))
}

# Null modes of the finite-mesh stiffness matrix are independent component
# levels. A Gaussian observation on a component contributes one inverse range
# to the marginal likelihood as that level's prior SD grows with range.
.hyper_spde_boundary <- function(object, sp) {
  n <- length(sp$C0_diag)
  if (!n || is.null(sp$G1_p) || is.null(sp$A_p)) return(NA_real_)
  arms <- object$responses %||% list(object)
  projectors <- if (is.list(sp$A_p)) lapply(seq_along(sp$A_p), function(k)
    list(p = sp$A_p[[k]], i = sp$A_i[[k]], x = sp$A_x[[k]]))
    else list(list(p = sp$A_p, i = sp$A_i, x = sp$A_x))
  parent <- seq_len(n)
  height <- integer(n)
  root <- function(i) {
    while (parent[i] != i) {
      parent[i] <<- parent[parent[i]]
      i <- parent[i]
    }
    i
  }
  for (j in seq_len(n)) {
    k <- seq.int(sp$G1_p[j] + 1L, sp$G1_p[j + 1L])
    if (sp$G1_p[j] == sp$G1_p[j + 1L]) next
    for (i in sp$G1_i[k[sp$G1_x[k] != 0]] + 1L) {
      ri <- root(i); rj <- root(j)
      if (ri == rj) next
      if (height[ri] < height[rj]) parent[ri] <- rj
      else {
        parent[rj] <- ri
        if (height[ri] == height[rj]) height[ri] <- height[ri] + 1L
      }
    }
  }
  orphan <- cpp_spde_orphan_mass(sp$C0_diag)
  observed <- which(Reduce(`|`, lapply(projectors, function(A) diff(A$p) > 0)) & !orphan)
  roots <- vapply(seq_len(n), root, integer(1))
  components <- split(observed, roots[observed])
  null_component <- function(vertices) {
    if (any(orphan[roots == roots[vertices[1L]]])) return(FALSE)
    sums <- vapply(vertices, function(j) {
      if (sp$G1_p[j] == sp$G1_p[j + 1L]) return(c(sum = 0, magnitude = 0))
      k <- seq.int(sp$G1_p[j] + 1L, sp$G1_p[j + 1L])
      c(sum = sum(sp$G1_x[k]), magnitude = sum(abs(sp$G1_x[k])))
    }, numeric(2))
    all(abs(sums[1L, ]) <= 64 * .Machine$double.eps * n *
          pmax(sums[2L, ], .Machine$double.xmin))
  }
  components <- Filter(null_component, components)
  if (!length(components)) return(0)
  if (all(vapply(arms, function(a) .family_base(a$family) %in% c("gaussian", "lognormal", "t"), TRUE))) {
    loadings <- lapply(seq_along(arms), function(a) {
      A <- projectors[[a]]
      ii <- jj <- integer(length(A$x)); xx <- numeric(length(A$x)); pos <- 0L
      for (b in seq_along(components)) for (j in components[[b]]) {
        if (A$p[j] == A$p[j + 1L]) next
        k <- seq.int(A$p[j] + 1L, A$p[j + 1L])
        ix <- pos + seq_along(k)
        ii[ix] <- A$i[k] + 1L; jj[ix] <- b; xx[ix] <- A$x[k]
        pos <- pos + length(k)
      }
      keep <- seq_len(pos)
      Matrix::sparseMatrix(i = ii[keep], j = jj[keep], x = xx[keep],
        dims = c(length(arms[[a]]$y), length(components)))
    })
    flat <- lapply(arms, function(a) {
      X <- a$X %||% a$model_matrix
      if (is.null(X)) return(matrix(0, length(a$y), 0))
      free <- if (length(a$beta_prior_prec)) a$beta_prior_prec == 0
        else rep_len(is.infinite(a$beta_prior$sd %||% 2.5), ncol(X))
      X[, free, drop = FALSE]
    })
    X <- Matrix::bdiag(flat)
    Z <- do.call(rbind, loadings)
    rank <- function(M) if (ncol(M)) as.numeric(Matrix::rankMatrix(M, method = "qr")) else 0L
    return(rank(cbind(X, Z)) - rank(X))
  }
  sum(vapply(components, function(vertices) {
    cuts <- c(FALSE, FALSE)
    for (a in seq_along(arms)) {
      A <- projectors[[a]]
      rows <- unique(unlist(lapply(vertices, function(j) {
        if (A$p[j] == A$p[j + 1L]) return(integer())
        k <- seq.int(A$p[j] + 1L, A$p[j + 1L])
        A$i[k[A$x[k] != 0]] + 1L
      })))
      if (!length(rows)) next
      arm <- arms[[a]]
      cuts <- cuts | .hyper_level_tails(arm$family, arm$y[rows],
        (arm$n_trials %||% rep(1, length(arm$y)))[rows])
    }
    as.numeric(all(cuts))
  }, numeric(1)))
}

# Which tails of an additive field level the observation likelihood removes.
.hyper_level_tails <- function(family, y, trials) {
  switch(.family_base(family),
    gaussian =, lognormal =, t =, gamma =, beta = c(TRUE, TRUE),
    binomial =, beta_binomial = c(any(y > 0), any(y < trials)),
    poisson =, neg_binomial_1 =, neg_binomial_2 =, tweedie = c(any(y > 0), TRUE),
    truncated_poisson =, truncated_neg_binomial_2 = c(any(y > 1), TRUE),
    inverse_gaussian = c(TRUE, FALSE), c(NA, NA))
}

.hyper_range_models <- function(object) {
  out <- object$range_tail %||% list()
  sp <- object$spatial
  specs <- object$outer_axis_specs %||% object$hyper_specs
  for (s in specs %||% list()) {
    if (!sub("^b[0-9]+[.]", "", s$name) %in% .SPATIAL_LENGTH_AXES) next
    bounded <- (length(s$bounds) == 2L && all(is.finite(s$bounds))) ||
      (length(s$slab_bounds) == 2L && is.null(out[[s$name]]))
    if (is.null(out[[s$name]]) && (bounded || length(s$grid) == 1L)) {
      out[[s$name]] <- list(prior_power = NA_real_, field = "",
        fixed = length(s$grid) == 1L, bounded = bounded)
    }
  }
  if (is.list(sp) && is.null(object$theta_grid)) {
    type <- tolower(sp$type %||% "")
    axes <- switch(type,
      gp = c("log_phi_gp", "phi_gp"), nngp = c("log_phi_gp", "phi_gp"),
      hsgp = "log_lengthscale_hsgp",
      multiscale = if (identical(sp$approx, "hsgp"))
        c("log_lengthscale_hsgp_local", "log_lengthscale_hsgp_regional")
        else c("log_phi_gp_local", "log_phi_gp_regional"),
      svc = paste0("log_phi_svc[", seq_len(sp$n_svc %||% 1L), "]"),
      spde = if (isTRUE(object$joint) || length(object$range_draws)) "log_kappa",
      character())
    layout <- object$param_names %||% colnames(.fit_draws(object))
    for (a in intersect(axes, layout)) {
      if (is.null(out[[a]])) out[[a]] <- list(prior_power = 1, field = type, fixed = FALSE)
    }
  }
  for (a in names(out)) {
    model <- out[[a]]
    spec <- Filter(function(s) identical(s$name, a), specs %||% list())
    if (length(spec)) {
      bd <- spec[[1]]$bounds
      model$bounded <- isTRUE(model$bounded) || (length(bd) == 2L && all(is.finite(bd)))
    }
    if (is.null(model$likelihood_power)) model$likelihood_power <- if (model$field %in% c("spde", "rsr_spde")) {
      blocks <- object$prior %||% object$blocks
      if (!is.null(blocks$type)) blocks <- list(blocks)
      bi <- if (grepl("^b[0-9]+[.]", a)) as.integer(sub("^b([0-9]+)[.].*", "\\1", a)) else 1L
      block <- if (length(blocks) >= bi) blocks[[bi]] else sp
      b <- .hyper_spde_boundary(object, block)
      if (isTRUE(model$scale_fixed)) b else min(b, 1)
    } else if (.hyper_positive_boundary(object)) 0 else NA_real_
    out[[a]] <- model
  }
  out
}

# The model declaration travels separately from the numerical integration
# window. Every reporting route reads this contract, including derived ranges.
.hyper_attach_moments <- function(object) {
  if (is.list(object$nested) && !is.null(object$nested$theta_grid)) {
    rec <- object$nested
    rec$spatial <- object$spatial
    rec$family <- object$family
    rec$y <- object$y
    rec$n_trials <- object$n_trials
    rec <- .hyper_attach_moments(rec)
    rec$range_mean <- rec$theta_mean[["range"]]
    rec$range_sd <- rec$theta_sd[["range"]]
    object$nested <- rec
    object$moment_existence <- rec$moment_existence
    object$theta_log_mean <- rec$theta_log_mean
    object$theta_log_sd <- rec$theta_log_sd
  }
  models <- .hyper_range_models(object)
  if (!length(models)) return(object)
  if (is.null(object$theta_log_mean)) {
    object$theta_log_mean <- stats::setNames(rep(NA_real_, length(models)), names(models))
    object$theta_log_sd <- object$theta_log_mean
  }
  tg <- .nl_theta_matrix(object)
  dr <- if (is.null(tg)) .fit_draws(object)
  if (!is.null(tg) && is.null(object$theta_mean)) {
    object$theta_mean <- colSums(tg * object$weights)
    object$theta_sd <- sqrt(pmax(0, colSums(tg^2 * object$weights) - object$theta_mean^2))
  }
  for (a in names(models)) {
    model <- models[[a]]
    object$moment_existence[[a]] <- .hyper_moment_rule(model)
    if (!is.null(tg) && a %in% colnames(tg)) {
      mm <- .hyper_moment_read(model, object$theta_mean[[a]], object$theta_sd[[a]])
      object$theta_mean[[a]] <- mm$mean
      object$theta_sd[[a]] <- mm$sd
      if (is.infinite(mm$sd)) object$theta_sd_source[[a]] <- "nonexistent"
      z <- log(tg[, a]); w <- object$weights / sum(object$weights)
      lm <- sum(w * z)
      object$theta_log_mean[[a]] <- lm
      object$theta_log_sd[[a]] <- sqrt(sum(w * (z - lm)^2))
      for (b in seq_along(object$block_moments)) {
        bm <- object$block_moments[[b]]
        j <- match(match(a, colnames(tg)), bm$axis_cols)
        if (is.na(j)) next
        object$block_moments[[b]]$mean[j] <- mm$mean
        object$block_moments[[b]]$sd[j] <- mm$sd
      }
    } else if (a %in% colnames(dr)) {
      z <- if (identical(a, "log_kappa")) log(object$range_draws)
        else if (identical(a, "phi_gp")) log(dr[, a]) else dr[, a]
      object$theta_log_mean[[a]] <- mean(z)
      object$theta_log_sd[[a]] <- stats::sd(z)
      if (identical(a, "log_kappa") && length(object$range_summary)) {
        mm <- .hyper_moment_read(model, object$range_summary[["mean"]],
                                stats::sd(object$range_draws))
        object$range_summary[c("mean", "sd")] <- c(mm$mean, mm$sd)
      }
    }
  }
  object
}

