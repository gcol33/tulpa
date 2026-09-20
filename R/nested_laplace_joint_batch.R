# Batched multi-response joint nested-Laplace fits.
#
# B species that share one design (arms, latent blocks, outer grid layout,
# solver settings) and differ only in their responses, their arm dispersions and
# the nodes of any dispersion axis are fit through one fused grid solve
# (`cpp_nested_laplace_joint_multi_batch`): one design pass per cell for all
# species, B block-diagonal Newton solves. Each species' grid result is the list
# `cpp_nested_laplace_joint_multi` returns for that species alone.
#
# `tulpa_joint_grid_batch()` runs whole fits that way. Each fit is a function of
# no arguments that performs one ordinary fit (a `tulpa_nested_laplace_joint()`
# call, or a consumer front door that makes one). The fits are run twice. The
# first run stops each fit at its main grid solve and keeps the kernel request;
# the fused solve answers all requests at once; the second run replays each fit
# with its main grid solve served from the fused result. Every other kernel call
# a fit makes (placement, refinement, diagnostics) runs as it always does, so a
# species' fit is the object its ordinary fit returns, built by the same code.

# Run `solve` as the fit's main outer-grid solve: the one kernel call a grid
# batch captures and serves.
.joint_main_grid_solve <- function(solve) {
  op <- options(tulpa.nl_grid_main = TRUE)
  on.exit(options(op), add = TRUE)
  solve()
}

# The grid-batch hook on the kernel door. Outside a batch, and for every kernel
# call that is not the fit's first main grid solve, returns NULL and the kernel
# runs. In capture mode the request is signalled as a `tulpa_grid_request`
# condition, which the batch runner turns into an exit from the fit; in serve
# mode the request must be the one captured, and the fused result is returned.
.joint_grid_batch_intercept <- function(args) {
  gb <- getOption("tulpa.nl_grid_batch")
  if (!is.environment(gb) || isTRUE(gb$consumed) ||
      !isTRUE(getOption("tulpa.nl_grid_main"))) {
    return(NULL)
  }
  gb$consumed <- TRUE
  if (identical(gb$mode, "capture")) {
    cp <- getOption("tulpa.nl_checkpoint", NULL)
    gb$checkpoint <- if (is.list(cp)) as.character(cp$path) else ""
    signalCondition(structure(
      class = c("tulpa_grid_request", "condition"),
      list(message = "joint grid request", call = NULL, args = args)))
    stop("A captured joint grid request reached no batch runner.",
         call. = FALSE)
  }
  if (!identical(args, gb$request)) {
    gb$mismatch <- TRUE
    .joint_grid_batch_ineligible(paste0(
      "a replayed fit issued a different grid request from the one it issued ",
      "when captured, so the batched result does not belong to it."))
  }
  gb$result
}

# Signal that a set of fits cannot share one fused grid solve.
.joint_grid_batch_ineligible <- function(reason) {
  stop(structure(
    class = c("tulpa_grid_batch_ineligible", "error", "condition"),
    list(message = paste0("joint grid batch: ", reason), call = NULL)))
}

# Run `fn` with the grid-batch state `gb` in scope.
.joint_with_grid_batch <- function(gb, fn) {
  op <- options(tulpa.nl_grid_batch = gb)
  on.exit(options(op), add = TRUE)
  fn()
}

#' Fit multiple joint nested-Laplace models through one fused grid solve
#'
#' Fits `B` responses that share one design (arms, latent blocks, outer grid
#' layout, solver settings) and differ only in their responses, their arm
#' dispersions and the nodes of any dispersion axis, through one fused grid
#' solve (`cpp_nested_laplace_joint_multi_batch`): one design pass per outer-
#' grid cell for all `B` responses, `B` block-diagonal Newton solves, instead
#' of `B` independent fits each repeating the design pass.
#'
#' Each element of `fits` is a function of no arguments that performs one
#' ordinary fit (a [tulpa_nested_laplace_joint()] call, or a consumer front
#' door that makes one). The fits run twice: the first run stops each fit at
#' its main outer-grid solve and keeps the kernel request; the fused solve
#' answers every request at once; the second run replays each fit with its
#' main grid solve served from the fused result. Every other kernel call a
#' fit makes (placement, refinement, diagnostics) runs as it always does, so
#' each returned fit is the object its ordinary fit would have returned,
#' built by the same code.
#'
#' The fits replay one after another from the random number state the batch
#' was called at, so they draw the same stream the same fits called in
#' sequence would draw; each capture starts from that state too, and a fit
#' whose grid request depended on a draw taken before it is refused at its
#' replay rather than served. Warnings and messages a capture run raises are
#' muffled; the replay raises them again.
#'
#' Raises a `tulpa_grid_batch_ineligible` condition (catchable with
#' `tryCatch(..., tulpa_grid_batch_ineligible = ...)`) when the fits do not
#' share a design or a request uses a setting the fused driver does not
#' carry. The random number state is restored to where the batch was called
#' before that condition leaves, so a caller that falls back to fitting the
#' responses one at a time draws the stream it would have drawn without the
#' batch.
#'
#' @param fits A non-empty list of functions of no arguments, each performing
#'   one ordinary joint nested-Laplace fit.
#' @return A list of fit results, one per element of `fits`, in order.
#' @seealso [tulpa_nested_laplace_joint()]
#' @export
tulpa_joint_grid_batch <- function(fits) {
  if (!is.list(fits) || length(fits) < 1L ||
      !all(vapply(fits, is.function, logical(1)))) {
    stop("`fits` must be a non-empty list of functions of no arguments.",
         call. = FALSE)
  }
  seed0 <- .joint_grid_batch_seed()
  withCallingHandlers(
    .joint_grid_batch_run(fits, seed0),
    tulpa_grid_batch_ineligible = function(e)
      .joint_grid_batch_restore_seed(seed0))
}

.joint_grid_batch_run <- function(fits, seed0) {
  B <- length(fits)
  requests <- vector("list", B)
  for (s in seq_len(B)) {
    .joint_grid_batch_restore_seed(seed0)
    gb <- new.env(parent = emptyenv())
    gb$mode <- "capture"
    gb$consumed <- FALSE
    captured <- withRestarts(
      withCallingHandlers(
        .joint_with_grid_batch(gb, function() { fits[[s]](); NULL }),
        tulpa_grid_request = function(cond)
          invokeRestart("tulpa_grid_captured", cond$args),
        warning = function(w) invokeRestart("muffleWarning"),
        message = function(m) invokeRestart("muffleMessage")),
      tulpa_grid_captured = function(args) args)
    if (is.null(captured)) {
      .joint_grid_batch_ineligible(sprintf(
        "fit %d made no joint nested-Laplace grid solve.", s))
    }
    if (length(gb$checkpoint) && any(nzchar(gb$checkpoint))) {
      .joint_grid_batch_ineligible(sprintf(
        "fit %d writes a grid checkpoint, which the fused solve does not.", s))
    }
    requests[[s]] <- captured
  }

  results <- .cpp_joint_multi_batch(requests)

  .joint_grid_batch_restore_seed(seed0)
  lapply(seq_len(B), function(s) {
    gb <- new.env(parent = emptyenv())
    gb$mode <- "serve"
    gb$consumed <- FALSE
    gb$request <- requests[[s]]
    gb$result <- results[[s]]
    gb$mismatch <- FALSE
    fit <- tryCatch(.joint_with_grid_batch(gb, fits[[s]]), error = function(e) {
      if (isTRUE(gb$mismatch)) {
        .joint_grid_batch_ineligible(sprintf(paste0(
          "fit %d issued a different grid request when replayed than when ",
          "captured."), s))
      }
      stop(e)
    })
    if (isTRUE(gb$mismatch)) {
      .joint_grid_batch_ineligible(sprintf(paste0(
        "fit %d issued a different grid request when replayed than when ",
        "captured."), s))
    }
    if (!isTRUE(gb$consumed)) {
      stop(sprintf(paste0("joint grid batch: fit %d did not reach its grid ",
                          "solve when replayed."), s), call. = FALSE)
    }
    fit
  })
}

.joint_grid_batch_seed <- function() {
  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
    get(".Random.seed", envir = globalenv(), inherits = FALSE) else NULL
}

.joint_grid_batch_restore_seed <- function(seed) {
  if (is.null(seed)) {
    if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
      rm(".Random.seed", envir = globalenv())
  } else {
    assign(".Random.seed", seed, envir = globalenv())
  }
}

# A request's arguments under the kernel entry's own formal names, which callers
# may abbreviate (R matches a unique prefix).
.joint_grid_request_canonical <- function(req) {
  formal <- names(formals(cpp_nested_laplace_joint_multi))
  names(req) <- vapply(names(req), function(n) {
    i <- pmatch(n, formal)
    if (is.na(i)) n else formal[[i]]
  }, character(1), USE.NAMES = FALSE)
  req
}

# The kernel arguments a fused solve carries per species. Everything else in a
# request is design and must agree across the batch.
.joint_grid_request_design <- function(req) {
  req$arms_list <- lapply(req$arms_list, function(a) {
    a$y <- NULL
    a$phi <- NULL
    a
  })
  if (!is.null(req$phi_grid_per_arm)) {
    req$phi_grid_per_arm <- lapply(req$phi_grid_per_arm, function(v)
      if (is.null(v)) NULL else length(v))
  }
  req
}

# One fused grid solve over captured `.cpp_joint_multi()` requests. Returns one
# grid result per request, each the list the single-species kernel returns for
# it.
.cpp_joint_multi_batch <- function(requests) {
  requests <- lapply(requests, .joint_grid_request_canonical)
  B <- length(requests)
  req1 <- requests[[1L]]
  unsupported <- c(
    x_init            = length(req1$x_init_nullable) > 0L,
    x_init_per_cell   = !is.null(req1$x_init_per_cell),
    n_threads_outer   = as.integer(req1$n_threads_outer %||% 1L) != 1L,
    tile_ids          = length(req1$tile_ids) > 0L,
    prune_tol         = as.numeric(req1$prune_tol %||% 0) > 0,
    compute_skew      = isTRUE(req1$compute_skew),
    debias            = !is.null(req1$debias),
    cila              = !is.null(req1$cila),
    inner_refresh     = as.integer(req1$inner_refresh %||% 1L) != 1L,
    inner_sparse_override = as.integer(req1$inner_sparse_override %||% 0L) != 0L,
    screen_log_offset = !is.null(req1$screen_log_offset),
    uncoupled_arm     = !all(vapply(req1$arms_list, function(a)
      isTRUE(a$coupled), logical(1))))
  if (any(unsupported)) {
    .joint_grid_batch_ineligible(paste0(
      "the grid request uses ", paste(names(unsupported)[unsupported],
                                      collapse = ", "),
      ", which the fused solve does not carry."))
  }
  design1 <- .joint_grid_request_design(req1)
  for (s in seq_len(B)[-1L]) {
    ds <- .joint_grid_request_design(requests[[s]])
    if (!identical(ds, design1)) {
      differ <- union(setdiff(names(ds), names(design1)),
                      names(design1)[!vapply(names(design1), function(nm)
                        identical(ds[[nm]], design1[[nm]]), logical(1))])
      .joint_grid_batch_ineligible(sprintf(
        "fit %d does not share the design of fit 1 (differs in %s).",
        s, paste(differ, collapse = ", ")))
    }
  }

  kernel <- lapply(requests, .joint_phi_args_to_kernel)
  arm_y <- function(k) lapply(kernel, function(r) as.numeric(r$arms_list[[k]]$y))
  n_arms <- length(req1$arms_list)
  y_batch <- lapply(seq_len(n_arms), function(k) {
    ys <- arm_y(k)
    if (length(ys[[1L]]) == 0L) NULL else do.call(cbind, ys)
  })
  phi_batch <- matrix(
    vapply(kernel, function(r) vapply(r$arms_list, function(a)
      as.numeric(a$phi), numeric(1)), numeric(n_arms)),
    nrow = n_arms, ncol = B)
  phi_grid_per_arm <- if (is.null(req1$phi_grid_per_arm)) NULL else
    lapply(seq_along(req1$phi_grid_per_arm), function(k) {
      if (is.null(req1$phi_grid_per_arm[[k]])) return(NULL)
      do.call(cbind, lapply(kernel, function(r)
        as.numeric(r$phi_grid_per_arm[[k]])))
    })

  cpp_nested_laplace_joint_multi_batch(
    arms_list          = kernel[[1L]]$arms_list,
    copy_arms          = req1$copy_arms,
    copy_blocks        = req1$copy_blocks,
    blocks_spec        = req1$blocks_spec,
    theta_grid         = req1$theta_grid,
    axis_offsets       = req1$axis_offsets,
    n_batch            = as.integer(B),
    y_batch            = y_batch,
    phi_batch          = phi_batch,
    max_iter           = as.integer(req1$max_iter %||% 50L),
    tol                = as.numeric(req1$tol %||% 1e-6),
    cell_coupling_name = as.character(req1$cell_coupling_name %||% "separable"),
    store_Q            = isTRUE(req1$store_Q),
    phi_grid_per_arm   = phi_grid_per_arm,
    hessian_pd_mode    = as.integer(req1$hessian_pd_mode %||% 0L),
    step_curvature_mode = as.integer(req1$step_curvature_mode %||% 0L),
    force_sparse       = isTRUE(req1$force_sparse),
    fixed_block_p      = as.integer(req1$fixed_block_p %||% 0L),
    fixed_block_constraints = req1$fixed_block_constraints,
    # Read where the single-species kernel call reads it: the switch is added at
    # the kernel call rather than captured on the request, so a batched species
    # that read it anywhere else could integrate under a different answer from
    # the fit it reproduces.
    compute_fitted_var = .nl_want_fitted_var())
}

# Shared marshaling for the kernel-level entries below: responses + prior ->
# arms / copy spec / blocks_spec / axis_offsets / outer grid, with any
# dispersion axis in `phi_grid` crossed on as the multi-block driver crosses it.
# Returns the `.cpp_joint_multi()` arguments of that solve.
# @keywords internal
.tulpa_nl_joint_request <- function(responses, prior, copy = NULL,
                                    phi_grid = NULL, max_iter = 200L,
                                    tol = 1e-6, cell_coupling = "separable",
                                    store_Q = FALSE) {
  prior_list <- if (.is_multi_block_prior(prior)) prior else list(prior)
  n_arms <- length(responses)
  arms <- lapply(seq_along(responses),
                 function(k) .normalise_joint_arm_multi(responses[[k]], k))
  arm_names <- names(responses) %||% paste0("arm", seq_along(responses))
  cp <- .resolve_copy_multi(copy, responses, prior_list)
  B <- length(prior_list)

  per_block <- lapply(seq_len(B), function(b) {
    copy_pos <- if (cp$has_copy) match((b - 1L), cp$copy_blocks_zero) else NA_integer_
    is_copy  <- !is.na(copy_pos)
    alpha_grid_b <- if (is_copy) cp$alpha_grids[[copy_pos]] else numeric(0)
    .joint_block_axis_grid(prior_list[[b]], is_copy, alpha_grid_b, b)
  })
  block_grids <- lapply(per_block, function(x) x$grid)
  prepared    <- lapply(per_block, function(x) x$prepared)
  axis_counts  <- vapply(block_grids, ncol, integer(1))
  axis_offsets <- as.integer(c(0L, cumsum(axis_counts)))
  axis_names   <- unlist(lapply(seq_along(block_grids), function(b) {
    cn <- colnames(block_grids[[b]])
    if (is.null(cn) || length(cn) == 0L) character(0) else paste0("b", b, ".", cn)
  }))
  blocks_spec <- lapply(seq_along(prepared), function(b) {
    .joint_block_spec_for_cpp(prepared[[b]], n_arms, b, arms = arms)
  })

  row_counts <- vapply(block_grids, nrow, integer(1))
  idx <- do.call(expand.grid, lapply(row_counts, seq_len))
  joint_grid <- do.call(cbind, lapply(seq_along(block_grids), function(b) {
    block_grids[[b]][idx[[b]], , drop = FALSE]
  }))
  if (ncol(joint_grid) > 0L) colnames(joint_grid) <- axis_names
  phi_axes <- .normalise_phi_grid(phi_grid, arm_names)
  if (!is.null(phi_axes) && any(lengths(phi_axes) > 0L)) {
    joint_grid <- .joint_multi_cross_phi(joint_grid, phi_axes, axis_names)$grid
  }

  list(
    arms_list          = arms,
    copy_arms          = as.integer(cp$copy_arms_zero),
    copy_blocks        = as.integer(cp$copy_blocks_zero),
    blocks_spec        = blocks_spec,
    theta_grid         = .joint_multi_cpp_grid(joint_grid, axis_offsets, B, cp),
    axis_offsets       = axis_offsets,
    max_iter           = as.integer(max_iter),
    tol                = as.numeric(tol),
    n_threads          = 1L,
    x_init_nullable    = NULL,
    store_Q            = isTRUE(store_Q),
    phi_grid_per_arm   = .joint_multi_phi_per_arm(joint_grid, arm_names),
    n_threads_outer    = 1L,
    tile_ids           = NULL,
    tile_pilot_cells   = NULL,
    prune_tol          = 0.0,
    force_sparse       = FALSE,
    cell_coupling_name = as.character(cell_coupling),
    inner_refresh      = 1L
  )
}

# The fused kernel on B species of one design: species s takes column s of each
# `y_batch` arm matrix (NULL keeps the template arm's response), column s of
# `phi_batch` as its arm dispersions and `phi_grid_batch[[s]]` as its dispersion
# axes. Returns `per_species`, each the grid result the single-species kernel
# returns for that species (`tulpa_nl_joint_single()`).
# @keywords internal
tulpa_nl_joint_batch <- function(responses, prior, copy = NULL,
                                 n_batch, y_batch, phi_batch,
                                 max_iter = 200L, tol = 1e-6,
                                 cell_coupling = "separable",
                                 store_Q = TRUE, phi_grid_batch = NULL) {
  n_arms <- length(responses)
  phi_batch <- as.matrix(phi_batch)
  if (nrow(phi_batch) != n_arms || ncol(phi_batch) != n_batch) {
    stop("`phi_batch` must be [n_arms x n_batch].", call. = FALSE)
  }
  if (!is.list(y_batch) || length(y_batch) != n_arms) {
    stop("`y_batch` must be a list of length n_arms.", call. = FALSE)
  }
  if (!is.null(phi_grid_batch) &&
      (!is.list(phi_grid_batch) || length(phi_grid_batch) != n_batch)) {
    stop("`phi_grid_batch` must be NULL or a list of length n_batch.",
         call. = FALSE)
  }
  requests <- lapply(seq_len(n_batch), function(s) {
    resp <- responses
    for (k in seq_len(n_arms)) {
      yk <- y_batch[[k]]
      if (!is.null(yk)) {
        yk <- as.matrix(yk)
        if (nrow(yk) != length(resp[[k]]$y) || ncol(yk) != n_batch) {
          stop("`y_batch[[", k, "]]` must be [N_k x n_batch].", call. = FALSE)
        }
        resp[[k]]$y <- as.numeric(yk[, s])
      }
      resp[[k]]$phi <- phi_batch[k, s]
    }
    .tulpa_nl_joint_request(resp, prior, copy,
                            phi_grid = phi_grid_batch[[s]],
                            max_iter = max_iter, tol = tol,
                            cell_coupling = cell_coupling, store_Q = store_Q)
  })
  list(per_species = .cpp_joint_multi_batch(requests),
       theta_grid  = requests[[1L]]$theta_grid,
       axis_offsets = requests[[1L]]$axis_offsets)
}

# The single-species kernel at the grid `tulpa_nl_joint_batch()` builds.
# @keywords internal
tulpa_nl_joint_single <- function(responses, prior, copy = NULL,
                                  max_iter = 200L, tol = 1e-6,
                                  cell_coupling = "separable", store_Q = FALSE,
                                  phi_grid = NULL) {
  do.call(.cpp_joint_multi, .tulpa_nl_joint_request(
    responses, prior, copy, phi_grid = phi_grid, max_iter = max_iter,
    tol = tol, cell_coupling = cell_coupling, store_Q = store_Q))
}
