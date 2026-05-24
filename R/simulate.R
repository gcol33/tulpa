#' @importFrom stats rbinom rpois rgamma rexp terms rnorm rnbinom rbeta rcauchy
NULL

#' Simulate data from a tulpa model
#'
#' @description
#' Generic simulator that dispatches through a `tulpa_family`'s `simulate_fn`.
#' Given a model spec (formula + family + data) and a parameter vector or a
#' fitted model, generate one or more synthetic response datasets.
#'
#' Used internally by [prior_predict()] and exposed for posterior predictive
#' checks, simulation-based calibration, and what-if analyses with fixed
#' parameters.
#'
#' @param formula A model formula, or list of formulas keyed by process name.
#' @param family A `tulpa_family` object (see [tulpa_family()]).
#' @param data Data frame with covariates and grouping factors.
#' @param theta One of:
#'   - A named list with `beta` (numeric or list per process), `u` (list of RE
#'     coefficient vectors, one per RE term per process), `extras` (named list
#'     of family-specific extras like `phi`, `sigma_y`).
#'   - A `tulpa_fit` object: posterior draws are sampled from `$draws`.
#'   - `NULL`: equivalent to a single prior draw (shortcut).
#' @param n_sims Number of simulated datasets. When `theta` is a fit, draws are
#'   subsampled (or recycled) to `n_sims`. Default 1.
#' @param priors Used only if `theta = NULL`; default `tulpa_priors()`.
#' @param seed Optional integer seed.
#' @param ... Passed to `family$simulate_fn`.
#'
#' @return A `tulpa_simulate` object: list with `y` (length-`n_sims` list of
#'   simulated responses), `theta` (parameters used per sim), `linpred`,
#'   `family`, `n_sims`, `n_obs`.
#'
#' @examples
#' fam <- tulpa_family(
#'   name = "gaussian",
#'   simulate_fn = function(eta, params, n_obs, ...) {
#'     rnorm(n_obs, eta[[1]], params$sigma_y)
#'   },
#'   extra_params = list(sigma_y = prior_half_normal(1))
#' )
#' df <- data.frame(y = rep(0, 20), x = rnorm(20))
#' theta <- list(
#'   beta = list(y = c(0.5, 1.0)),
#'   u = list(y = list()),
#'   extras = list(sigma_y = 0.5)
#' )
#' sim <- tulpa_simulate(y ~ x, fam, df, theta = theta, n_sims = 3, seed = 1)
#' length(sim$y)  # 3
#'
#' @export
tulpa_simulate <- function(formula, family, data,
                           theta = NULL, n_sims = 1L,
                           priors = NULL, seed = NULL, ...) {

  if (!is.null(seed)) {
    old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) {
      get(".Random.seed", envir = .GlobalEnv)
    } else NULL
    set.seed(seed)
    on.exit({
      if (is.null(old_seed)) {
        rm(".Random.seed", envir = .GlobalEnv)
      } else {
        assign(".Random.seed", old_seed, envir = .GlobalEnv)
      }
    }, add = TRUE)
  }

  validate_family(family)
  n_sims <- as.integer(n_sims)
  if (length(n_sims) != 1L || is.na(n_sims) || n_sims < 1L) {
    stop("`n_sims` must be a positive integer", call. = FALSE)
  }

  process_names <- family$process_names
  formulas <- normalize_formulas(formula, process_names)
  parsed <- lapply(formulas, tulpa_parse_formula)
  built  <- lapply(parsed, tulpa_build_model_data, data = data)
  n_obs <- built[[1]]$n_obs

  # Resolve theta source
  theta_source <- classify_theta(theta)

  y_list <- vector("list", n_sims)
  theta_list <- vector("list", n_sims)
  linpred_list <- vector("list", n_sims)

  for (s in seq_len(n_sims)) {
    th <- switch(theta_source,
      "list" = validate_theta(theta, built, family),
      "fit"  = theta_from_fit(theta, s, built, family),
      "null" = {
        if (is.null(priors)) priors <- tulpa_priors()
        draw_theta_from_priors(built, priors, family)
      }
    )
    eta <- build_eta(built, th, family$link_inv, apply_link = FALSE)
    sim_args <- c(list(eta = eta, params = th$extras, n_obs = n_obs), list(...))
    y_list[[s]] <- do.call(family$simulate_fn, sim_args)
    theta_list[[s]] <- th
    linpred_list[[s]] <- eta
  }

  structure(
    list(
      y = y_list,
      theta = theta_list,
      linpred = linpred_list,
      family = family,
      n_sims = n_sims,
      n_obs = n_obs,
      process_names = process_names
    ),
    class = "tulpa_simulate"
  )
}


#' Print method for tulpa_simulate
#' @param x A tulpa_simulate object
#' @param ... Ignored
#' @export
print.tulpa_simulate <- function(x, ...) {
  cat("tulpa simulated datasets\n")
  cat("========================\n")
  cat("Family:    ", x$family$name, "\n")
  cat("Processes: ", paste(x$process_names, collapse = ", "), "\n")
  cat("Sims:      ", x$n_sims, "\n")
  cat("Obs:       ", x$n_obs, "\n")
  invisible(x)
}


# ============================================================================
# Family helper: minimal R-side family for tests and simple cases
#
# Real model packages register richer families (linking to C++ LikelihoodSpec
# for sampling). For prior_predict / tulpa_simulate we only need the simulator
# plus structural metadata. Keeping the R-side contract small means model
# packages can attach `simulate_fn` to whatever family object they already use.
# ============================================================================

#' Construct a minimal tulpa_family for simulation
#'
#' @description
#' Lightweight constructor for a `tulpa_family` object that exposes the
#' contract required by [prior_predict()] and [tulpa_simulate()]. Model
#' packages (numdenom, tulpaObs) register richer families that also link to
#' C++ likelihoods; this helper is for tests and simple custom families that
#' only need simulation.
#'
#' @param name Family name (character, length 1).
#' @param simulate_fn `function(eta, params, n_obs, ...)` returning a numeric
#'   vector of length `n_obs` (single-process) or a list of such vectors keyed
#'   by `process_names` (multi-process). `eta` is a list of linear predictors,
#'   one per process.
#' @param process_names Character vector. Defaults to `"y"` (single-process).
#' @param extra_params Named list of `tulpa_prior` objects for likelihood-
#'   specific scalar parameters (e.g., dispersion `phi`). Drawn at each
#'   prior-predictive iteration. Defaults to empty.
#' @param link_inv List of inverse-link functions per process; defaults to
#'   identity for every process. tulpa passes the raw linear predictor to
#'   `simulate_fn`, so most families implement the link inside `simulate_fn`
#'   (e.g., `mu = exp(eta)` for Poisson). The `link_inv` slot exists for
#'   families that prefer to keep the inverse-link separate.
#'
#' @return A `tulpa_family` object.
#'
#' @examples
#' fam <- tulpa_family(
#'   name = "poisson",
#'   simulate_fn = function(eta, params, n_obs, ...) {
#'     rpois(n_obs, exp(eta[[1]]))
#'   }
#' )
#'
#' @export
tulpa_family <- function(name,
                         simulate_fn,
                         process_names = "y",
                         extra_params = list(),
                         link_inv = NULL) {

  if (!is.character(name) || length(name) != 1L) {
    stop("`name` must be a single character string", call. = FALSE)
  }
  if (!is.function(simulate_fn)) {
    stop("`simulate_fn` must be a function", call. = FALSE)
  }
  if (!is.character(process_names) || length(process_names) < 1L) {
    stop("`process_names` must be a non-empty character vector", call. = FALSE)
  }
  if (length(extra_params) > 0L) {
    if (is.null(names(extra_params)) || any(names(extra_params) == "")) {
      stop("`extra_params` must be a named list", call. = FALSE)
    }
    for (nm in names(extra_params)) {
      if (!inherits(extra_params[[nm]], "tulpa_prior")) {
        stop(sprintf("extra_params$%s must be a tulpa_prior object", nm),
             call. = FALSE)
      }
    }
  }
  if (is.null(link_inv)) {
    link_inv <- stats::setNames(rep(list(identity), length(process_names)),
                                process_names)
  } else {
    if (length(link_inv) != length(process_names)) {
      stop("`link_inv` must have one entry per process", call. = FALSE)
    }
    for (f in link_inv) {
      if (!is.function(f)) stop("`link_inv` entries must be functions",
                                call. = FALSE)
    }
    if (is.null(names(link_inv))) names(link_inv) <- process_names
  }

  structure(
    list(
      name = name,
      simulate_fn = simulate_fn,
      process_names = process_names,
      extra_params = extra_params,
      link_inv = link_inv
    ),
    class = "tulpa_family"
  )
}


#' Validate a tulpa_family
#'
#' @param family Object to validate
#' @keywords internal
validate_family <- function(family) {
  if (!inherits(family, "tulpa_family")) {
    stop("`family` must be a tulpa_family object (see ?tulpa_family)",
         call. = FALSE)
  }
  required <- c("name", "simulate_fn", "process_names")
  miss <- setdiff(required, names(family))
  if (length(miss) > 0L) {
    stop("tulpa_family missing required slots: ",
         paste(miss, collapse = ", "), call. = FALSE)
  }
  if (!is.function(family$simulate_fn)) {
    stop("family$simulate_fn must be a function", call. = FALSE)
  }
  invisible(TRUE)
}


# ============================================================================
# Internal helpers shared by prior_predict and tulpa_simulate
# ============================================================================

# Normalize a formula or list-of-formulas argument against expected processes.
# Returns a named list of formulas keyed by process_names.
#' @keywords internal
normalize_formulas <- function(formula, process_names) {
  n_proc <- length(process_names)
  if (inherits(formula, "formula")) {
    if (n_proc == 1L) {
      out <- list(formula)
      names(out) <- process_names
      return(out)
    }
    # Single formula for multi-process: replicate; model packages can override
    out <- rep(list(formula), n_proc)
    names(out) <- process_names
    return(out)
  }
  if (is.list(formula)) {
    if (length(formula) != n_proc) {
      stop(sprintf("formula list has %d entries but family has %d processes",
                   length(formula), n_proc), call. = FALSE)
    }
    if (is.null(names(formula))) {
      names(formula) <- process_names
    } else if (!setequal(names(formula), process_names)) {
      stop("formula list names must match family$process_names", call. = FALSE)
    }
    if (!all(vapply(formula, inherits, logical(1), "formula"))) {
      stop("All entries in formula list must be formulas", call. = FALSE)
    }
    return(formula[process_names])
  }
  stop("`formula` must be a formula or a named list of formulas", call. = FALSE)
}


# Sample a parameter vector from priors for one prior-predictive draw.
# Returns: list(beta = list per process, u = list of RE blocks per process,
#               sigma = list of sigma_re per process, extras = named list).
#' @keywords internal
draw_theta_from_priors <- function(built, priors, family) {
  process_names <- names(built)

  beta_list  <- vector("list", length(process_names))
  u_list     <- vector("list", length(process_names))
  sigma_list <- vector("list", length(process_names))
  names(beta_list) <- names(u_list) <- names(sigma_list) <- process_names

  for (k in seq_along(built)) {
    b <- built[[k]]
    beta_list[[k]] <- rprior(priors$beta, b$n_fixed)

    re_u <- vector("list", length(b$re_terms))
    re_sigma <- numeric(length(b$re_terms))
    for (j in seq_along(b$re_terms)) {
      sigma_j <- rprior(priors$sigma, 1)
      re_sigma[j] <- sigma_j
      n_groups <- b$re_terms[[j]]$n_groups
      n_coefs  <- b$re_terms[[j]]$n_coefs
      re_u[[j]] <- matrix(rnorm(n_groups * n_coefs, 0, sigma_j),
                          nrow = n_groups, ncol = n_coefs)
    }
    u_list[[k]] <- re_u
    sigma_list[[k]] <- re_sigma
  }

  extras <- list()
  for (nm in names(family$extra_params)) {
    extras[[nm]] <- rprior(family$extra_params[[nm]], 1)
  }

  list(beta = beta_list, u = u_list, sigma = sigma_list, extras = extras)
}


# Sample n iid values from a tulpa_prior. Centralizes the prior-distribution
# dispatch so prior_predict and tulpa_simulate stay in lockstep with priors.R.
#' @keywords internal
rprior <- function(prior, n) {
  d <- prior$distribution
  switch(d,
    normal      = rnorm(n, prior$mean, prior$sd),
    half_normal = abs(rnorm(n, 0, prior$sd)),
    half_cauchy = abs(stats::rcauchy(n, 0, prior$scale)),
    gamma       = rgamma(n, shape = prior$shape, rate = prior$rate),
    exponential = rexp(n, rate = prior$rate),
    beta        = stats::rbeta(n, prior$alpha, prior$beta),
    pc          = rexp(n, rate = prior$rate),
    stop("Unsupported prior distribution: ", d, call. = FALSE)
  )
}


# Build per-process linear predictors eta = X %*% beta + sum_j Z_j u_j
# Returns a named list of numeric vectors (one per process).
#' @keywords internal
build_eta <- function(built, theta, link_inv, apply_link = FALSE) {
  process_names <- names(built)
  eta <- vector("list", length(process_names))
  names(eta) <- process_names

  for (k in seq_along(built)) {
    b <- built[[k]]
    pname <- process_names[[k]]
    beta_k <- theta$beta[[pname]]
    eta_k <- as.numeric(b$X %*% beta_k)

    for (j in seq_along(b$re_terms)) {
      re <- b$re_terms[[j]]
      u_jk <- theta$u[[pname]][[j]]
      contrib <- numeric(b$n_obs)
      if (re$has_intercept) {
        contrib <- contrib + u_jk[re$group_idx, 1L]
      }
      if (!is.null(re$slope_matrix)) {
        slope_offset <- if (re$has_intercept) 1L else 0L
        for (s in seq_len(ncol(re$slope_matrix))) {
          contrib <- contrib +
            re$slope_matrix[, s] * u_jk[re$group_idx, slope_offset + s]
        }
      }
      eta_k <- eta_k + contrib
    }

    if (apply_link && !is.null(link_inv) && !is.null(link_inv[[pname]])) {
      eta_k <- link_inv[[pname]](eta_k)
    }
    eta[[k]] <- eta_k
  }
  eta
}


# Classify the source of theta to drive the dispatch in tulpa_simulate.
#' @keywords internal
classify_theta <- function(theta) {
  if (is.null(theta)) return("null")
  if (inherits(theta, "tulpa_fit")) return("fit")
  if (is.list(theta)) return("list")
  stop("`theta` must be NULL, a list, or a tulpa_fit object", call. = FALSE)
}


# Validate a user-supplied theta list against the parsed model structure.
#' @keywords internal
validate_theta <- function(theta, built, family) {
  required <- c("beta", "u")
  miss <- setdiff(required, names(theta))
  if (length(miss) > 0L) {
    stop("`theta` missing required slots: ",
         paste(miss, collapse = ", "), call. = FALSE)
  }
  process_names <- names(built)
  if (is.numeric(theta$beta) && length(process_names) == 1L) {
    theta$beta <- stats::setNames(list(theta$beta), process_names)
  }
  if (is.list(theta$u) && length(process_names) == 1L &&
      !all(process_names %in% names(theta$u))) {
    theta$u <- stats::setNames(list(theta$u), process_names)
  }
  for (pname in process_names) {
    if (is.null(theta$beta[[pname]])) {
      stop(sprintf("theta$beta missing entry for process '%s'", pname),
           call. = FALSE)
    }
    if (length(theta$beta[[pname]]) != built[[pname]]$n_fixed) {
      stop(sprintf("theta$beta$%s has length %d, expected %d",
                   pname, length(theta$beta[[pname]]),
                   built[[pname]]$n_fixed), call. = FALSE)
    }
  }
  if (is.null(theta$extras)) theta$extras <- list()
  for (nm in names(family$extra_params)) {
    if (is.null(theta$extras[[nm]])) {
      stop(sprintf("theta$extras missing required family extra '%s'", nm),
           call. = FALSE)
    }
  }
  theta
}


# Pull a parameter draw from a fitted tulpa_fit object. Posterior draws are
# expected on $draws$beta etc. (the layout used by tulpa NUTS / Laplace).
#' @keywords internal
theta_from_fit <- function(fit, sim_index, built, family) {
  draws <- fit$draws
  if (is.null(draws)) {
    stop("tulpa_fit has no $draws slot to simulate from", call. = FALSE)
  }

  pick_row <- function(mat) {
    if (is.null(mat)) return(NULL)
    if (is.matrix(mat)) {
      idx <- ((sim_index - 1L) %% nrow(mat)) + 1L
      mat[idx, , drop = TRUE]
    } else {
      idx <- ((sim_index - 1L) %% length(mat)) + 1L
      mat[idx]
    }
  }

  process_names <- names(built)
  beta <- vector("list", length(process_names))
  names(beta) <- process_names
  u    <- vector("list", length(process_names))
  names(u) <- process_names

  for (k in seq_along(process_names)) {
    pname <- process_names[[k]]
    beta_slot <- draws[[paste0("beta_", pname)]] %||% draws$beta
    beta_row <- pick_row(beta_slot)
    if (is.null(beta_row)) {
      stop(sprintf("Cannot find beta draws for process '%s' in fit", pname),
           call. = FALSE)
    }
    if (length(beta_row) < built[[k]]$n_fixed) {
      stop(sprintf("beta row has %d entries, expected %d for process '%s'",
                   length(beta_row), built[[k]]$n_fixed, pname),
           call. = FALSE)
    }
    beta[[k]] <- as.numeric(beta_row[seq_len(built[[k]]$n_fixed)])

    u_blocks <- vector("list", length(built[[k]]$re_terms))
    for (j in seq_along(built[[k]]$re_terms)) {
      re <- built[[k]]$re_terms[[j]]
      u_slot_name <- paste0("u_", pname, "_", j)
      u_slot <- draws[[u_slot_name]]
      if (is.null(u_slot)) {
        u_blocks[[j]] <- matrix(0, nrow = re$n_groups, ncol = re$n_coefs)
      } else {
        u_row <- as.numeric(pick_row(u_slot))
        u_blocks[[j]] <- matrix(u_row, nrow = re$n_groups, ncol = re$n_coefs)
      }
    }
    u[[k]] <- u_blocks
  }

  extras <- list()
  for (nm in names(family$extra_params)) {
    extras[[nm]] <- pick_row(draws[[nm]])
    if (is.null(extras[[nm]])) {
      extras[[nm]] <- rprior(family$extra_params[[nm]], 1)
    }
  }

  list(beta = beta, u = u, sigma = NULL, extras = extras)
}


#' Generate random effects for simulation
#'
#' @param n_groups Number of groups
#' @param sigma Standard deviation
#' @param type Type of random effects: "iid" (default), "car", "ar1"
#'
#' @return Numeric vector of random effects
#' @keywords internal
sim_random_effects <- function(n_groups, sigma = 0.5, type = "iid") {

  if (type == "iid") {
    return(rnorm(n_groups, 0, sigma))
  }

  if (type == "ar1") {
    # Generate AR(1) process
    re <- numeric(n_groups)
    rho <- 0.7
    re[1] <- rnorm(1, 0, sigma / sqrt(1 - rho^2))
    for (i in 2:n_groups) {
      re[i] <- rho * re[i - 1] + rnorm(1, 0, sigma)
    }
    return(re)
  }

  # Default: IID
  rnorm(n_groups, 0, sigma)
}


#' Generate spatial random effects for simulation
#'
#' @param adjacency Adjacency matrix
#' @param sigma Standard deviation
#' @param type Spatial type: "icar" (default), "bym2"
#'
#' @return Numeric vector of spatial effects
#' @keywords internal
sim_spatial_effects <- function(adjacency, sigma = 0.5, type = "icar") {
  n <- nrow(adjacency)

  if (type == "icar") {
    # Generate ICAR effects via conditional simulation
    adj <- as.matrix(adjacency)
    n_neighbors <- rowSums(adj)
    Q <- diag(n_neighbors) - adj

    # Regularize for sampling
    Q_reg <- Q + 0.01 * diag(n)
    L <- chol(Q_reg)

    # Generate from precision matrix
    z <- rnorm(n)
    phi <- backsolve(L, z) * sigma

    # Center
    phi <- phi - mean(phi)
    return(phi)
  }

  # Default: IID
  rnorm(n, 0, sigma)
}


#' Generate temporal random effects for simulation
#'
#' @param n_times Number of time points
#' @param sigma Standard deviation
#' @param type Temporal type: "rw1", "rw2", "ar1"
#' @param rho Autocorrelation for AR(1). Default 0.7.
#'
#' @return Numeric vector of temporal effects
#' @keywords internal
sim_temporal_effects <- function(n_times, sigma = 0.5, type = "rw1",
                                  rho = 0.7) {

  if (type == "rw1") {
    # First-order random walk
    phi <- cumsum(rnorm(n_times, 0, sigma))
    phi <- phi - mean(phi)  # Center
    return(phi)
  }

  if (type == "rw2") {
    # Second-order random walk (smoother)
    phi <- numeric(n_times)
    phi[1] <- rnorm(1, 0, sigma)
    if (n_times >= 2) phi[2] <- phi[1] + rnorm(1, 0, sigma)
    for (t in 3:n_times) {
      phi[t] <- 2 * phi[t - 1] - phi[t - 2] + rnorm(1, 0, sigma)
    }
    phi <- phi - mean(phi)
    return(phi)
  }

  if (type == "ar1") {
    return(sim_random_effects(n_times, sigma = sigma, type = "ar1"))
  }

  # Default: IID
  rnorm(n_times, 0, sigma)
}
