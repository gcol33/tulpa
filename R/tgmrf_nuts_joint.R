#' Joint NUTS over (beta, z, theta) for a tgmrf C++-backend block
#'
#' @description
#' No-U-Turn Sampler (Hoffman & Gelman 2014) running directly on the full
#' joint vector \eqn{(\beta, z, \theta)} in C++. Unlike
#' [tulpa_tgmrf_nuts()], which targets the marginal posterior
#' \eqn{p(\theta\mid y)} via a finite-difference gradient on the inner
#' Laplace log-marginal, this sampler integrates the joint posterior in one
#' Hamiltonian system using closed-form gradients for \eqn{(\beta, z)} and a
#' central finite-difference gradient on \eqn{(Q, \mu, \log p(\theta))} with
#' respect to \eqn{\theta}.
#'
#' Scope: **C++-backend tgmrf blocks only** (`block$backend == "cpp"`,
#' created via [tgmrf_cpp()]). Calling R for `Q(theta)` at every leapfrog
#' step would dwarf any compute-time advantage the joint sampler has over
#' the marginal-theta path. Users with R-closure blocks already have
#' [tulpa_tgmrf_imh()] (independence-MH over theta) and
#' [tulpa_tgmrf_nuts()] (outer-theta NUTS) as exact-MCMC options.
#'
#' @details
#' Math (mirroring `src/tgmrf_nuts.cpp`):
#'
#' \deqn{\log p(\beta, z, \theta \mid y) = \sum_i \log p(y_i \mid \eta_i)
#'   + \tfrac{1}{2}\log\det Q(\theta) - \tfrac{1}{2}(z - \mu)^\top Q(\theta)(z - \mu)
#'   - \tfrac{n_{\mathrm{lat}}}{2}\log(2\pi) + \log p(\theta).}
#'
#' Gradients used by the leapfrog:
#' \itemize{
#'   \item \eqn{\partial / \partial \beta_k = \sum_i X_{ik} s_i} with
#'     \eqn{s_i = d\log p(y_i\mid\eta_i)/d\eta_i};
#'   \item \eqn{\partial / \partial z_j = s_{i(j)} - (Q(z-\mu))_j};
#'   \item \eqn{\partial / \partial \theta_m = \tfrac{1}{2}\mathrm{tr}(Q^{-1}\partial_mQ)
#'     - \tfrac{1}{2}(z-\mu)^\top \partial_m Q (z-\mu) + (\partial_m \mu)^\top Q(z-\mu)
#'     + d\log p(\theta)/d\theta_m}.
#' }
#'
#' \eqn{\partial_m Q}, \eqn{\partial_m \mu}, \eqn{d\log p(\theta)/d\theta_m} are
#' obtained by central finite differences on the registered C++ kernels --
#' \eqn{2\,\mathrm{dim}(\theta)} extra `Q` evaluations per leapfrog step.
#' For typical \eqn{\dim(\theta) \le 5} this is cheap. AD-on-theta (via the
#' reserved arena/forward instantiation slots in `TgmrfSpec`) is a planned
#' upgrade; the wire format already carries the slots.
#'
#' Mass matrix: diagonal, initialised from the inverse-variance of the
#' pilot Laplace posterior. Step size: dual-averaging warmup targeting
#' `target_accept = 0.8` (higher than the default 0.65 for plain NUTS
#' because \eqn{(\beta, z, \theta)} typically has dimension \eqn{\ge 50}).
#'
#' @inheritParams tulpa_tgmrf_nuts
#' @param block A tgmrf block with `backend == "cpp"`. R-closure blocks are
#'   rejected with an error pointing to [tgmrf_cpp()] or
#'   [tulpa_tgmrf_nuts()].
#' @param target_accept Target acceptance for dual-averaging step-size
#'   adaptation. Default 0.8 (higher than plain NUTS's 0.65 because joint
#'   dimensionality is large).
#' @param fd_step Central finite-difference step on theta for
#'   \eqn{\partial_m Q}, \eqn{\partial_m \mu}, \eqn{d\log p(\theta)/d\theta_m}.
#'   Default 1e-3.
#' @param debug_gradient_check If `TRUE`, runs a one-shot numerical-gradient
#'   sanity check at the initial \eqn{q} and prints the max relative error
#'   across all `D = p + n_lat + theta_dim` components.
#' @param seed Integer seed for the C++ RNG. Default uses `sample.int(.Machine$integer.max, 1L)`.
#'
#' @return A list with class `c("tulpa_tgmrf_nuts_joint", "tulpa_fit")`:
#'   \itemize{
#'     \item `draws_beta`, `draws_z`, `draws_theta` -- post-warmup draws.
#'     \item `means_beta`, `means_z`, `means_theta`,
#'       `sds_beta`, `sds_z`, `sds_theta` -- column-wise moments.
#'     \item `mean_accept`, `tree_depth`, `divergent`, `epsilon` --
#'       sampler diagnostics.
#'     \item `pilot`, `mode_beta`, `mode_z`, `mode_theta`, `mass_diag` --
#'       pilot Laplace outputs used for init / mass matrix.
#'     \item `inference_mode = "exact"`, `inference_tier = 1L`,
#'       `backend = "tgmrf_nuts_joint"`.
#'   }
#'
#' @references Hoffman & Gelman (2014). The No-U-Turn Sampler. JMLR
#'   15:1593-1623.
#' @seealso [tulpa_tgmrf_nuts()] for the marginal-theta NUTS;
#'   [tulpa_tgmrf_imh()] for the IMH composition; [tgmrf_cpp()] for the
#'   C++-backend block constructor.
#' @export
tulpa_tgmrf_nuts_joint <- function(y, n_trials, X, block,
                                   family = "binomial",
                                   phi = 1.0,
                                   re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                                   n_iter = 500L, warmup = n_iter %/% 2L,
                                   epsilon = NULL,
                                   max_depth = 6L,
                                   target_accept = 0.8,
                                   pilot_axis_points = 5L,
                                   fd_step = 1e-3,
                                   max_iter = 50L, tol = 1e-7,
                                   n_threads = 1L,
                                   verbose = FALSE,
                                   debug_gradient_check = FALSE,
                                   seed = NULL) {

  if (!inherits(block, "tgmrf")) {
    stop("`block` must be a tgmrf object.", call. = FALSE)
  }
  if (!identical(block$backend, "cpp")) {
    stop("tulpa_tgmrf_nuts_joint requires a C++-backend block ",
         "(see `tgmrf_cpp()`). For R-closure blocks, use ",
         "`tulpa_tgmrf_nuts()` (outer-theta NUTS) or `tulpa_tgmrf_imh()` ",
         "(independence-MH).", call. = FALSE)
  }
  if (!is.null(re_idx) || n_re_groups != 0L) {
    stop("tulpa_tgmrf_nuts_joint does not support random effects in v1. ",
         "Use `tulpa_tgmrf_nuts()` instead.", call. = FALSE)
  }
  if (max_depth < 1L || max_depth > 12L) {
    stop("`max_depth` must be in [1, 12].", call. = FALSE)
  }
  if (n_iter < 2L || warmup < 1L || warmup >= n_iter) {
    stop("Need 1 <= warmup < n_iter and n_iter >= 2.", call. = FALSE)
  }

  N <- length(y)
  p <- ncol(X)
  n_lat <- block$n_latent
  d <- block$theta_dim
  obs_idx <- block$obs_idx %||% seq_len(N)
  if (length(obs_idx) != N) {
    stop("obs_idx length (", length(obs_idx),
         ") does not match N = ", N, ".", call. = FALSE)
  }

  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)

  # -- Pilot Laplace ----------------------------------------------------------
  pilot_block <- block
  pilot_block$obs_idx <- obs_idx
  if (!is.null(pilot_axis_points) && pilot_axis_points != 5L) {
    axes <- vector("list", d)
    for (j in seq_len(d)) {
      lo <- if (!is.null(block$bounds)) block$bounds$lower[j] else block$init[j] - 2
      hi <- if (!is.null(block$bounds)) block$bounds$upper[j] else block$init[j] + 2
      axes[[j]] <- seq(lo, hi, length.out = pilot_axis_points)
    }
    names(axes) <- block$theta_names
    pilot_block$theta_grid_built <- as.matrix(do.call(expand.grid, axes))
  }

  pilot <- tulpa_nested_laplace(
    y = y, n_trials = n_trials, X = X,
    prior = pilot_block,
    family = family, phi = phi,
    max_iter = max_iter, tol = tol, n_threads = n_threads
  )

  k_star <- which.max(pilot$log_marginal)
  theta_init <- as.numeric(pilot$theta_grid[k_star, ])
  names(theta_init) <- block$theta_names

  # Mode at the grid argmax: pilot$modes is n_grid x n_x where n_x = p + n_lat.
  if (is.null(pilot$modes)) {
    stop("Pilot Laplace did not return modes; joint NUTS init requires them.",
         call. = FALSE)
  }
  mode_full <- as.numeric(pilot$modes[k_star, ])
  if (length(mode_full) < p + n_lat) {
    stop("Pilot modes row has length ", length(mode_full),
         " but expected at least p + n_lat = ", p + n_lat, ".", call. = FALSE)
  }
  beta_init <- mode_full[seq_len(p)]
  z_init    <- mode_full[p + seq_len(n_lat)]

  # -- Mass matrix diagonal ---------------------------------------------------
  # beta: from pilot$beta_sd (posterior SD at grid argmax)^2 if available;
  # else use a moderate default.
  # z: from inverse of the diagonal of the joint Hessian at the mode if
  #    available (pilot$grid_hessians stores only the beta block by default);
  #    otherwise default to 1.0 (z is unit-variance after the prior centring).
  # theta: from pilot$theta_sd^2 (posterior SD of theta marginals from the
  #    grid weighting) -- same scale source as outer-theta NUTS.
  beta_var <- rep(1.0, p)
  if (!is.null(pilot$beta_sd)) {
    bs <- as.numeric(pilot$beta_sd)
    if (length(bs) == p && all(is.finite(bs)) && all(bs > 0)) {
      beta_var <- bs^2
    }
  }
  z_var <- rep(1.0, n_lat)
  theta_var <- pmax(as.numeric(pilot$theta_sd)^2, 1e-3)
  if (length(theta_var) != d || !all(is.finite(theta_var))) {
    theta_var <- rep(0.25, d)
  }
  M_inv_diag <- c(beta_var, z_var, theta_var)

  if (is.null(epsilon)) {
    # Conservative initial step. The dual-averaging warmup will adapt.
    epsilon <- 0.05
  }

  # n_trials must be an integer vector of length N. For Poisson / Gaussian
  # families nested_laplace uses a length-1 sentinel; the C++ joint sampler
  # wants a vector of length N -- replicate if needed.
  if (length(n_trials) == 1L) {
    n_trials_full <- rep(as.integer(n_trials), N)
  } else if (length(n_trials) == N) {
    n_trials_full <- as.integer(n_trials)
  } else {
    stop("`n_trials` must have length 1 or length N (= ", N, "); got ",
         length(n_trials), ".", call. = FALSE)
  }

  raw <- cpp_tgmrf_nuts_joint(
    y           = as.numeric(y),
    n_trials    = n_trials_full,
    X           = as.matrix(X),
    obs_idx     = as.integer(obs_idx),
    family      = as.character(family),
    phi         = as.numeric(phi),
    cpp_id      = block$cpp_id,
    theta_dim   = as.integer(d),
    n_latent    = as.integer(n_lat),
    beta_init   = as.numeric(beta_init),
    z_init      = as.numeric(z_init),
    theta_init  = as.numeric(theta_init),
    M_inv_diag  = as.numeric(M_inv_diag),
    epsilon0    = as.numeric(epsilon),
    n_iter      = as.integer(n_iter),
    n_warmup    = as.integer(warmup),
    max_depth   = as.integer(max_depth),
    target_accept = as.numeric(target_accept),
    fd_step     = as.numeric(fd_step),
    verbose     = isTRUE(verbose),
    seed        = as.integer(seed),
    debug_gradient_check = isTRUE(debug_gradient_check)
  )

  beta_names  <- colnames(X)
  if (is.null(beta_names)) beta_names <- paste0("beta_", seq_len(p))
  z_names     <- paste0("z_", seq_len(n_lat))
  theta_names <- block$theta_names

  draws_beta  <- raw$draws_beta
  draws_z     <- raw$draws_z
  draws_theta <- raw$draws_theta
  colnames(draws_beta)  <- beta_names
  colnames(draws_z)     <- z_names
  colnames(draws_theta) <- theta_names

  n_keep <- nrow(draws_theta)
  keep <- seq_len(n_keep)
  tree_depth_keep <- raw$tree_depth[(warmup + 1L):length(raw$tree_depth)]
  accept_keep     <- raw$accept_prob[(warmup + 1L):length(raw$accept_prob)]
  divergent_keep  <- raw$divergent[(warmup + 1L):length(raw$divergent)]

  fit <- list(
    draws_beta     = draws_beta,
    draws_z        = draws_z,
    draws_theta    = draws_theta,
    means_beta     = colMeans(draws_beta),
    means_z        = colMeans(draws_z),
    means_theta    = colMeans(draws_theta),
    sds_beta       = apply(draws_beta,  2L, stats::sd),
    sds_z          = apply(draws_z,     2L, stats::sd),
    sds_theta      = apply(draws_theta, 2L, stats::sd),
    n_samples      = n_keep,
    mean_accept    = mean(accept_keep),
    tree_depth     = tree_depth_keep,
    divergent      = divergent_keep,
    n_divergent    = sum(divergent_keep),
    epsilon        = raw$epsilon,
    pilot          = pilot,
    mode_beta      = beta_init,
    mode_z         = z_init,
    mode_theta     = theta_init,
    mass_diag      = M_inv_diag,
    inference_mode = "exact",
    inference_tier = 1L,
    backend        = "tgmrf_nuts_joint"
  )
  class(fit) <- c("tulpa_tgmrf_nuts_joint", "tulpa_fit")
  fit
}
