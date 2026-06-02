# Multi-block joint dispatch (list-of-blocks prior).
#
# Part of the joint nested-Laplace driver; the public entry point
# tulpa_nested_laplace_joint() lives in nested_laplace_joint.R.

# =============================================================================
# Multi-block joint dispatch (Phase J-B)
# =============================================================================
#
# Activated when `prior` is a list-of-blocks (each element has a `type`
# field). Routes to cpp_nested_laplace_joint_multi, which builds a
# std::vector<LatentBlock> on the joint side. At most one block can be
# the copy block; first ship restricts the copy block to spatial types
# (icar / bym2 / car_proper). See dev_notes/plan_multi_block_joint.md.

# Multi-block detection. Same shape as the single-arm side: a list whose
# elements are themselves named lists carrying a `type` field, and whose
# top-level entry does NOT carry a `type` (which would mark a single-block
# prior).
.is_multi_block_prior_joint <- function(p) {
    is.list(p) && is.null(p$type) && length(p) > 0 &&
        all(vapply(p, function(x) is.list(x) && !is.null(x$type), logical(1)))
}

# Validate one arm spec for the multi-block path. Same shape as
# `.normalise_joint_arm` *except* `spatial_idx` is no longer required at
# the arm level — per-arm idx vectors live inside each block spec instead.
# A trailing `spatial_idx = integer(0)` placeholder is set so the existing
# JointArm packaging path doesn't complain.
.normalise_joint_arm_multi <- function(a, k) {
    if (!is.list(a)) {
        stop("Arm ", k, ": expected a list of arm spec fields.", call. = FALSE)
    }
    must_have <- c("y", "X", "family")
    missing <- setdiff(must_have, names(a))
    if (length(missing)) {
        stop("Arm ", k, ": missing fields ",
             paste(shQuote(missing), collapse = ", "), ".", call. = FALSE)
    }
    N <- length(a$y)
    a$y <- as.numeric(a$y)
    if (!is.matrix(a$X)) {
        stop("Arm ", k, ": `X` must be a numeric matrix.", call. = FALSE)
    }
    if (nrow(a$X) != N) {
        stop("Arm ", k, ": nrow(X) (", nrow(a$X), ") must equal length(y) (",
             N, ").", call. = FALSE)
    }
    # spatial_idx is OPTIONAL in multi-block (per-arm idx lives on the block).
    # The legacy joint kernels still reach into parsed[k_arm].spatial_idx, but
    # the multi-block kernel does NOT, so a length-N placeholder of zeros is
    # safe -- it shuts up parse_joint_arms' length check without contributing
    # to eta.
    a$spatial_idx <- if (is.null(a$spatial_idx)) rep(0L, N)
                     else as.integer(a$spatial_idx)
    if (length(a$spatial_idx) != N) {
        stop("Arm ", k, ": length(spatial_idx) (", length(a$spatial_idx),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$n_trials <- if (is.null(a$n_trials)) rep(1L, N) else as.integer(a$n_trials)
    if (length(a$n_trials) != N) {
        stop("Arm ", k, ": length(n_trials) (", length(a$n_trials),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$re_idx <- if (is.null(a$re_idx)) rep(0, N) else as.numeric(a$re_idx)
    if (length(a$re_idx) != N) {
        stop("Arm ", k, ": length(re_idx) (", length(a$re_idx),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$n_re_groups <- as.integer(a$n_re_groups %||% 0L)
    a$sigma_re    <- as.numeric(a$sigma_re %||% 1.0)
    a$family      <- as.character(a$family)
    a$phi         <- as.numeric(a$phi %||% 1.0)
    a <- .normalise_arm_field_coef(a, k)
    a <- .normalise_arm_cell_coupling(a, k, N)
    a
}

# Detect whether `copy` is a LIST of copy specs (one coupled field each)
# rather than a single spec. A list-of-specs is an unnamed list whose
# elements are themselves lists carrying a `block` field; a single spec is
# itself a list with a `block` (or `arm`) entry at the top level.
.is_copy_spec_list <- function(copy) {
    if (!is.list(copy)) return(FALSE)
    if (!is.null(copy$block) || !is.null(copy$arm)) return(FALSE)
    nm <- names(copy)
    has_no_names <- is.null(nm) || all(!nzchar(nm))
    has_no_names && length(copy) > 0L &&
        all(vapply(copy, function(s) is.list(s) && !is.null(s$block),
                   logical(1)))
}

# Resolve and validate one copy spec into a (arm_zero, block_zero,
# alpha_grid) triple. The per-spec validation (arm in range, block in
# range, block type spatial, alpha_grid non-negative non-empty) lives here
# so the single-spec and list-of-specs paths share one source of truth.
.resolve_one_copy_spec <- function(spec, responses, prior_list) {
    arm_id <- spec$arm
    if (is.null(arm_id)) {
        stop("`copy$arm` must be a name or 1-based index.", call. = FALSE)
    }
    if (is.character(arm_id)) {
        nm <- names(responses)
        if (is.null(nm) || !arm_id %in% nm) {
            stop("`copy$arm` = '", arm_id, "' not found in names(responses).",
                 call. = FALSE)
        }
        arm_zero <- match(arm_id, nm) - 1L
    } else {
        arm_zero <- as.integer(arm_id) - 1L
        if (arm_zero < 0L || arm_zero >= length(responses)) {
            stop("`copy$arm` index out of range.", call. = FALSE)
        }
    }
    block_id <- spec$block
    if (is.null(block_id)) {
        stop("`copy$block` must be a 1-based index into `prior` for the ",
             "multi-block joint driver. (Copy semantics are restricted to ",
             "spatial blocks; see dev_notes/plan_multi_block_joint.md.)",
             call. = FALSE)
    }
    block_zero <- as.integer(block_id) - 1L
    if (block_zero < 0L || block_zero >= length(prior_list)) {
        stop("`copy$block` index (", block_id, ") out of range for length(prior) (",
             length(prior_list), ").", call. = FALSE)
    }
    block_type <- tolower(prior_list[[block_zero + 1L]]$type)
    spatial_types <- c("icar", "bym2", "car_proper")
    if (!block_type %in% spatial_types) {
        stop("`copy$block` points at type '", block_type, "'. Copy semantics ",
             "are restricted to spatial types (",
             paste(shQuote(spatial_types), collapse = ", "), "). ",
             "See dev_notes/plan_multi_block_joint.md.", call. = FALSE)
    }
    if (!is.null(spec$alpha_grid)) {
        alpha_axis <- as.numeric(spec$alpha_grid)
    } else {
        # Default: a small log-spaced alpha grid with 0 included so the
        # "no copy" base model carries posterior mass when supported.
        alpha_axis <- c(0, exp(seq(log(0.1), log(3), length.out = 5)))
    }
    if (length(alpha_axis) == 0L) {
        stop("`copy$alpha_grid` must have at least one non-negative value.",
             call. = FALSE)
    }
    if (any(alpha_axis < 0)) {
        stop("`copy$alpha_grid` values must be non-negative.",
             call. = FALSE)
    }
    list(arm_zero = arm_zero, block_zero = block_zero, alpha_grid = alpha_axis)
}

# Resolve the copy specification for the multi-block joint driver. `copy`
# may be a single spec `list(arm, block, alpha_grid)` (back-compat) OR an
# unnamed list of such specs (N coupled fields, one alpha axis each). Each
# spec must name a distinct spatial block. Returns vectors parallel over
# the copy blocks:
#   has_copy, copy_arms_zero (int vec), copy_blocks_zero (int vec),
#   alpha_grids (list of numeric, one per copy block).
.resolve_copy_multi <- function(copy, responses, prior_list) {
    if (is.null(copy)) {
        return(list(has_copy = FALSE, copy_arms_zero = integer(0),
                    copy_blocks_zero = integer(0), alpha_grids = list()))
    }
    specs <- if (.is_copy_spec_list(copy)) copy else list(copy)
    resolved <- lapply(specs, .resolve_one_copy_spec,
                       responses = responses, prior_list = prior_list)
    copy_blocks_zero <- vapply(resolved, function(r) r$block_zero, integer(1))
    copy_arms_zero   <- vapply(resolved, function(r) r$arm_zero,   integer(1))
    alpha_grids      <- lapply(resolved, function(r) r$alpha_grid)
    if (anyDuplicated(copy_blocks_zero)) {
        stop("Two copy specs target the same block (1-based index ",
             paste(unique(copy_blocks_zero[duplicated(copy_blocks_zero)]) + 1L,
                   collapse = ", "),
             "). Each coupled field must name a distinct spatial block.",
             call. = FALSE)
    }
    list(has_copy = TRUE, copy_arms_zero = copy_arms_zero,
         copy_blocks_zero = copy_blocks_zero, alpha_grids = alpha_grids)
}

# Per-block axis grid for the multi-block joint driver. When the block is
# the copy block (spatial only for first ship), the parameterisation
# uses (sigma, alpha[, rho/rho_car]) directly. Non-copy blocks use the
# standard single-arm conventions and reuse the `.NL_REGISTRY` defaults.
#
# Returns:
#   $grid    : matrix [n_block_cells x n_axes_for_block]
#   $names   : axis names
#   $prepared: block spec with defaults filled in (so downstream code can
#              read prior$adj_row_ptr etc. without re-checking presence)
.joint_block_axis_grid <- function(p, is_copy, alpha_grid,
                                    block_index) {
    type <- tolower(p$type)
    if (is_copy) {
        sigma_axis <- p$sigma_grid
        if (is.null(sigma_axis)) {
            sigma_axis <- exp(seq(log(0.1), log(3), length.out = 5))
        }
        sigma_axis <- as.numeric(sigma_axis)
        if (type == "icar") {
            gr <- expand.grid(sigma = sigma_axis,
                              alpha = alpha_grid,
                              KEEP.OUT.ATTRS = FALSE,
                              stringsAsFactors = FALSE)
        } else if (type == "bym2") {
            rho <- p$rho_grid %||% c(0.2, 0.5, 0.8, 0.95)
            gr <- expand.grid(sigma = sigma_axis,
                              alpha = alpha_grid,
                              rho   = as.numeric(rho),
                              KEEP.OUT.ATTRS = FALSE,
                              stringsAsFactors = FALSE)
        } else if (type == "car_proper") {
            rho_car <- p$rho_car_grid %||% c(0.5, 0.8, 0.95, 0.99)
            gr <- expand.grid(sigma   = sigma_axis,
                              alpha   = alpha_grid,
                              rho_car = as.numeric(rho_car),
                              KEEP.OUT.ATTRS = FALSE,
                              stringsAsFactors = FALSE)
        } else {
            stop("Block ", block_index, ": copy semantics not supported for ",
                 "type '", type, "' in first ship.", call. = FALSE)
        }
        return(list(grid     = as.matrix(gr),
                    names    = colnames(gr),
                    prepared = p))
    }
    # Non-copy: reuse the single-arm registry's grid construction. The
    # registry returns the standard (tau, [rho]) / (sigma, [rho]) axes.
    .nl_block_axis_grid(p)
}

# Convert one R-side block to the C++ `cpp_nested_laplace_joint_multi`
# format. Per-arm idx vectors live in the BLOCK spec (not the arm spec);
# the user supplies `spatial_idx` / `temporal_idx` / `obs_idx` as a list
# of length n_arms.
#
# `arms` is the list of normalised arm specs produced by
# `.normalise_joint_arm_multi()`. It is consumed by block types that need
# to read covariate columns at fit-time (HSGP with `svc_column`). Pass
# NULL only from contexts where no such block can appear.
.joint_block_spec_for_cpp <- function(p, n_arms, block_index, arms = NULL) {
    type <- tolower(p$type)
    if (type %in% c("icar", "bym2", "car_proper")) {
        if (is.null(p$spatial_idx)) {
            stop("Block ", block_index, " (type '", type, "'): ",
                 "`spatial_idx` is required as a list of length n_arms.",
                 call. = FALSE)
        }
        spatial_idx <- .multi_block_per_arm_idx(p$spatial_idx, n_arms,
                                                  block_index, "spatial_idx")
        out <- list(
            type            = type,
            spatial_idx     = spatial_idx,
            n_spatial_units = as.integer(p$n_spatial_units),
            adj_row_ptr     = as.integer(p$adj_row_ptr),
            adj_col_idx     = as.integer(p$adj_col_idx),
            n_neighbors     = as.integer(p$n_neighbors)
        )
        if (type == "bym2") {
            out$scale_factor <- as.numeric(p$scale_factor %||% 1.0)
        }
        # Optional per-arm per-row design weight (areal SVC). When set, the
        # field's contribution to arm k row i is row-scaled by svc_weight[[k]][i]:
        #   eta_i += svc_weight[[k]][i] * amplitude * z[cell_i].
        # Parallel to `spatial_idx`: a list of length n_arms, each a numeric
        # vector matching that arm's obs count and the per-arm spatial_idx
        # length. Composes multiplicatively with copy `arm_scale`.
        if (!is.null(p$svc_weight)) {
            out$svc_weight <- .multi_block_svc_weight(
                p$svc_weight, spatial_idx, n_arms, block_index)
        }
        out
    } else if (type %in% c("rw1", "rw2", "ar1")) {
        if (is.null(p$temporal_idx)) {
            stop("Block ", block_index, " (type '", type, "'): ",
                 "`temporal_idx` is required as a list of length n_arms.",
                 call. = FALSE)
        }
        temporal_idx <- .multi_block_per_arm_idx(p$temporal_idx, n_arms,
                                                   block_index, "temporal_idx")
        out <- list(
            type         = type,
            temporal_idx = temporal_idx,
            n_times      = as.integer(p$n_times)
        )
        if (type == "rw1") out$cyclic <- isTRUE(p$cyclic)
        out
    } else if (type == "iid") {
        if (is.null(p$obs_idx)) {
            stop("Block ", block_index, " (type 'iid'): ",
                 "`obs_idx` is required as a list of length n_arms.",
                 call. = FALSE)
        }
        obs_idx <- .multi_block_per_arm_idx(p$obs_idx, n_arms,
                                              block_index, "obs_idx")
        list(
            type    = "iid",
            obs_idx = obs_idx,
            n_units = as.integer(p$n_units)
        )
    } else if (type == "tgmrf") {
        # User-supplied GMRF: per-grid CSC Q + logdet + log p(theta).
        for (req in c("n_latent", "obs_idx", "Q_csc_p_per_grid",
                      "Q_csc_i_per_grid", "Q_csc_x_per_grid",
                      "logdet_Q_per_grid", "log_prior_theta_per_grid")) {
            if (is.null(p[[req]])) {
                stop("Block ", block_index, " (type 'tgmrf'): `", req,
                     "` is required.", call. = FALSE)
            }
        }
        obs_idx <- .multi_block_per_arm_idx(p$obs_idx, n_arms,
                                              block_index, "obs_idx")
        list(
            type                       = "tgmrf",
            n_latent                   = as.integer(p$n_latent),
            obs_idx                    = obs_idx,
            Q_csc_p_per_grid           = lapply(p$Q_csc_p_per_grid, as.integer),
            Q_csc_i_per_grid           = lapply(p$Q_csc_i_per_grid, as.integer),
            Q_csc_x_per_grid           = lapply(p$Q_csc_x_per_grid, as.numeric),
            logdet_Q_per_grid          = as.numeric(p$logdet_Q_per_grid),
            log_prior_theta_per_grid   = as.numeric(p$log_prior_theta_per_grid)
        )
    } else if (type == "hsgp") {
        # HSGP block: shared eigenvalues + per-arm Phi basis matrices.
        # Axes are (log_sigma2, log_lengthscale).
        #
        # Optional `svc_column`: 1-based index into each arm's `X`. When
        # set, the block becomes a spatially-varying coefficient:
        #     eta_i += X[i, svc_column] * f(s_i)
        # Implemented by row-scaling the per-arm basis at fit-time
        #     phi_scaled[k][i, m] = phi[k][i, m] * X[k][i, svc_column]
        # and passing the scaled phi to the existing DENSE_BASIS factory.
        # Multi-scale composition (1.6c) falls out by declaring multiple
        # type='hsgp' blocks in the same prior_list.
        for (req in c("m_total", "phi", "n_obs_per_arm", "eigenvalues")) {
            if (is.null(p[[req]])) {
                stop("Block ", block_index, " (type 'hsgp'): `", req,
                     "` is required.", call. = FALSE)
            }
        }
        if (!is.list(p$phi) || length(p$phi) != n_arms) {
            stop("Block ", block_index, " (type 'hsgp'): `phi` must be a ",
                 "list of length n_arms (", n_arms, ").", call. = FALSE)
        }
        phi <- lapply(p$phi, function(M) {
            M <- as.matrix(M); storage.mode(M) <- "double"; M
        })
        if (!is.null(p$svc_column)) {
            if (is.null(arms)) {
                stop("Block ", block_index, " (type 'hsgp'): `svc_column` ",
                     "requires the joint multi-block dispatcher path ",
                     "(arms not available here).", call. = FALSE)
            }
            svc <- as.integer(p$svc_column)
            if (length(svc) != 1L || is.na(svc) || svc < 1L) {
                stop("Block ", block_index, " (type 'hsgp'): `svc_column` ",
                     "must be a single positive integer (1-based column ",
                     "index into each arm's X).", call. = FALSE)
            }
            for (k in seq_len(n_arms)) {
                Xk <- arms[[k]]$X
                if (svc > ncol(Xk)) {
                    stop("Block ", block_index, " (type 'hsgp'): ",
                         "`svc_column` = ", svc, " exceeds ncol(arm ", k,
                         "$X) = ", ncol(Xk), ".", call. = FALSE)
                }
                if (nrow(phi[[k]]) != nrow(Xk)) {
                    stop("Block ", block_index, " (type 'hsgp'): ",
                         "nrow(phi[[", k, "]]) (", nrow(phi[[k]]),
                         ") must equal nrow(arm ", k, "$X) (", nrow(Xk),
                         ") when svc_column is set.", call. = FALSE)
                }
                phi[[k]] <- phi[[k]] * as.numeric(Xk[, svc])
            }
        }
        list(
            type           = "hsgp",
            m_total        = as.integer(p$m_total),
            phi            = phi,
            n_obs_per_arm  = as.integer(p$n_obs_per_arm),
            eigenvalues    = as.numeric(p$eigenvalues)
        )
    } else if (type == "hsgp_mo") {
        # Multi-output (co-regionalization) HSGP block (Stage 1.7). K = n_arms
        # correlated latent fields share basis + eigenvalues; coefficients are
        # cross-output-correlated via Sigma. First ship restricts K == 2 with
        # axes (sigma_1, sigma_2, rho, ell), all raw.
        for (req in c("m_total", "phi", "n_obs_per_arm", "eigenvalues")) {
            if (is.null(p[[req]])) {
                stop("Block ", block_index, " (type 'hsgp_mo'): `", req,
                     "` is required.", call. = FALSE)
            }
        }
        if (n_arms != 2L) {
            stop("Block ", block_index, " (type 'hsgp_mo'): first ship ",
                 "requires n_arms == 2 (got ", n_arms, "). Multi-output HSGP ",
                 "with K > 2 needs an LKJ-cholesky correlation ",
                 "parameterization on the outer grid (deferred follow-up).",
                 call. = FALSE)
        }
        if (!is.list(p$phi) || length(p$phi) != n_arms) {
            stop("Block ", block_index, " (type 'hsgp_mo'): `phi` must be a ",
                 "list of length n_arms (", n_arms, ").", call. = FALSE)
        }
        phi <- lapply(p$phi, function(M) {
            M <- as.matrix(M); storage.mode(M) <- "double"; M
        })
        list(
            type           = "hsgp_mo",
            m_total        = as.integer(p$m_total),
            phi            = phi,
            n_obs_per_arm  = as.integer(p$n_obs_per_arm),
            eigenvalues    = as.numeric(p$eigenvalues)
        )
    } else if (type == "spde") {
        # SPDE block: per-arm sparse barycentric projector A and shared FEM
        # mesh (C0_diag, G1). Axes are (range, sigma). Optional rational
        # coefficients for fractional nu.
        for (req in c("n_mesh", "A_x", "A_i", "A_p", "n_obs_per_arm",
                      "C0_diag", "G1_x", "G1_i", "G1_p", "nu")) {
            if (is.null(p[[req]])) {
                stop("Block ", block_index, " (type 'spde'): `", req,
                     "` is required.", call. = FALSE)
            }
        }
        per_arm <- function(field) {
            v <- p[[field]]
            if (!is.list(v) || length(v) != n_arms) {
                stop("Block ", block_index, " (type 'spde'): `", field,
                     "` must be a list of length n_arms (", n_arms, ").",
                     call. = FALSE)
            }
            v
        }
        out <- list(
            type           = "spde",
            n_mesh         = as.integer(p$n_mesh),
            A_x            = lapply(per_arm("A_x"), as.numeric),
            A_i            = lapply(per_arm("A_i"), as.integer),
            A_p            = lapply(per_arm("A_p"), as.integer),
            n_obs_per_arm  = as.integer(p$n_obs_per_arm),
            C0_diag        = as.numeric(p$C0_diag),
            G1_x           = as.numeric(p$G1_x),
            G1_i           = as.integer(p$G1_i),
            G1_p           = as.integer(p$G1_p),
            nu             = as.numeric(p$nu)
        )
        if (!is.null(p$rational_poles))
            out$rational_poles <- as.numeric(p$rational_poles)
        if (!is.null(p$rational_weights))
            out$rational_weights <- as.numeric(p$rational_weights)
        out
    } else if (type == "lf") {
        # Latent factor block (Stage 1.6a). Per-arm `obs_idx` maps each
        # obs to a slot in the factor field `u` (length n_latent); the
        # per-arm loadings `lambda` (length n_arms) are part of the joint
        # latent vector and co-optimized by the inner Newton. No outer-
        # grid axes — identifiability handled by tight Gaussian anchors
        # on (u_1, lambda_1) inside the C++ factory.
        if (is.null(p$obs_idx) || is.null(p$n_latent)) {
            stop("Block ", block_index, " (type 'lf'): both `obs_idx` ",
                 "(list of length n_arms) and `n_latent` are required.",
                 call. = FALSE)
        }
        obs_idx <- .multi_block_per_arm_idx(p$obs_idx, n_arms,
                                              block_index, "obs_idx")
        out <- list(
            type     = "lf",
            n_latent = as.integer(p$n_latent),
            obs_idx  = obs_idx
        )
        if (!is.null(p$sigma_u))      out$sigma_u      <- as.numeric(p$sigma_u)
        if (!is.null(p$sigma_lambda)) out$sigma_lambda <- as.numeric(p$sigma_lambda)
        if (!is.null(p$anchor_eps))   out$anchor_eps   <- as.numeric(p$anchor_eps)
        out
    } else {
        stop("Block type '", type, "' is not supported in multi-block joint priors.",
             call. = FALSE)
    }
}

# Coerce + validate the optional `svc_weight` for an areal block into a
# length-n_arms list of numeric vectors. Each entry must match that arm's
# obs count, which is the length of that arm's resolved `spatial_idx`.
# Accepts either a list (already per-arm) or a single numeric vector
# (replicated across arms when every arm shares the same per-row weight).
.multi_block_svc_weight <- function(svc_weight, spatial_idx, n_arms,
                                    block_index) {
    if (is.list(svc_weight)) {
        if (length(svc_weight) != n_arms) {
            stop("Block ", block_index, ": `svc_weight` is a list of ",
                 length(svc_weight), " entries but n_arms = ", n_arms, ".",
                 call. = FALSE)
        }
        w_list <- lapply(svc_weight, as.numeric)
    } else {
        w_list <- rep(list(as.numeric(svc_weight)), n_arms)
    }
    for (k in seq_len(n_arms)) {
        n_k <- length(spatial_idx[[k]])
        if (length(w_list[[k]]) != n_k) {
            stop("Block ", block_index, ": `svc_weight[[", k, "]]` has length ",
                 length(w_list[[k]]), " but arm ", k, " has ", n_k,
                 " observations (matching its `spatial_idx`).", call. = FALSE)
        }
        if (any(!is.finite(w_list[[k]]))) {
            stop("Block ", block_index, ": `svc_weight[[", k, "]]` contains ",
                 "non-finite values.", call. = FALSE)
        }
    }
    w_list
}

# Coerce a per-arm idx field into a length-n_arms list of integer vectors.
# Accepts either a list (already per-arm) or a single vector (replicated
# across arms; useful when every arm shares the same indexing — e.g. years
# 1..T mapped one-to-one).
.multi_block_per_arm_idx <- function(idx, n_arms, block_index, field_name) {
    if (is.list(idx)) {
        if (length(idx) != n_arms) {
            stop("Block ", block_index, ": `", field_name, "` is a list of ",
                 length(idx), " entries but n_arms = ", n_arms, ".",
                 call. = FALSE)
        }
        lapply(idx, as.integer)
    } else {
        rep(list(as.integer(idx)), n_arms)
    }
}

# Materialize the C++-facing theta_grid from a user-facing multi-block
# `joint_grid` (the latent-block axis columns; any trailing phi_<arm>
# dispersion columns are excluded here -- they ride on `phi_grid_per_arm`).
# The kernel reads `sigma_occ` / `sigma_pos` on a copy block; the R-side grid
# lives in (sigma, alpha), so this materializes sigma_occ = sigma and
# sigma_pos = alpha * sigma per copy block. Non-copy block columns pass
# through unchanged. Single source of truth for the main dispatch and the
# Pareto-k re-evaluation.
.joint_multi_cpp_grid <- function(joint_grid, axis_offsets, B, cp) {
    cpp_grid <- joint_grid[, seq_len(axis_offsets[B + 1L]), drop = FALSE]
    if (!cp$has_copy) return(cpp_grid)
    new_names <- colnames(cpp_grid)
    for (b_copy in (cp$copy_blocks_zero + 1L)) {
        cols_b  <- (axis_offsets[b_copy] + 1L):axis_offsets[b_copy + 1L]
        bare_b  <- sub("^b[0-9]+\\.", "", colnames(cpp_grid)[cols_b])
        i_sigma <- match("sigma", bare_b)
        i_alpha <- match("alpha", bare_b)
        if (is.na(i_sigma) || is.na(i_alpha)) {
            stop(".joint_multi_cpp_grid: copy block missing 'sigma' or 'alpha' axis.",
                 call. = FALSE)
        }
        sigma_col <- as.numeric(cpp_grid[, cols_b[i_sigma]])
        alpha_col <- as.numeric(cpp_grid[, cols_b[i_alpha]])
        cpp_grid[, cols_b[i_sigma]] <- sigma_col             # -> sigma_occ
        cpp_grid[, cols_b[i_alpha]] <- alpha_col * sigma_col  # -> sigma_pos
        new_names[cols_b[i_sigma]] <- paste0("b", b_copy, ".sigma_occ")
        new_names[cols_b[i_alpha]] <- paste0("b", b_copy, ".sigma_pos")
    }
    colnames(cpp_grid) <- new_names
    cpp_grid
}

# Extract the per-arm dispersion override list from a `joint_grid` carrying
# `phi_<arm>` columns. Returns NULL when no such column exists (the kernel
# then uses each arm's parse-time scalar phi); otherwise a length-n_arms list
# with entry k the column `phi_<arm_names[k]>` or NULL.
.joint_multi_phi_per_arm <- function(joint_grid, arm_names) {
    cn <- colnames(joint_grid)
    if (is.null(cn) || !any(startsWith(cn, "phi_"))) return(NULL)
    out <- vector("list", length(arm_names))
    for (k in seq_along(arm_names)) {
        col <- paste0("phi_", arm_names[k])
        if (col %in% cn) out[[k]] <- as.numeric(joint_grid[, col])
    }
    out
}

# Add the regularizing (sigma, alpha) hyperprior to a per-cell log-marginal
# vector for a multi-block `joint_grid`. With multiple copy blocks the prior
# is baked on the first block carrying a (sigma, alpha) axis pair. Single
# source of truth for the main dispatch and the Pareto-k re-evaluation.
.joint_multi_add_hp <- function(log_marginal, joint_grid, axis_offsets, B,
                                fn_sigma, fn_alpha) {
    if (is.null(fn_sigma) && is.null(fn_alpha)) return(log_marginal)
    view_map <- integer(0)
    for (b_idx in seq_len(B)) {
        cols_b  <- (axis_offsets[b_idx] + 1L):axis_offsets[b_idx + 1L]
        bare_b  <- sub("^b[0-9]+\\.", "", colnames(joint_grid)[cols_b])
        i_sigma <- match("sigma", bare_b)
        i_alpha <- match("alpha", bare_b)
        if (!is.na(i_sigma) && is.na(view_map["sigma"])) {
            view_map["sigma"] <- cols_b[i_sigma]
        }
        if (!is.na(i_alpha) && is.na(view_map["alpha"])) {
            view_map["alpha"] <- cols_b[i_alpha]
        }
    }
    if (length(view_map) == 0L) return(log_marginal)
    view <- joint_grid[, view_map, drop = FALSE]
    colnames(view) <- names(view_map)
    hp <- .joint_hp_vec_for_grids(view, fn_sigma, fn_alpha)
    if (!is.null(hp) && length(hp) == length(log_marginal)) {
        log_marginal <- log_marginal + hp
    }
    log_marginal
}

# Multi-block joint outer Pareto-k-hat. Builds the re-evaluation closure
# (round-trips a sampled user-facing `joint_grid` through the SAME kernel via
# the shared cpp-grid / phi / hyperprior helpers) and defers to the
# block-type-aware driver `.joint_pareto_k`. n_threads_outer = 1 / no tiling
# on the re-eval path, so no tile-partition reconstruction is needed.
#
# The diagnostic solves are bounded (gcol33/tulpa#51): each is warm-started
# from the modal cell's converged latent mode (the bulk of the importance
# weight sits near it, so those draws converge in a few Newton steps) and
# capped at `.K_DIAG_MAX_ITER` iterations. A draw at an implausible
# hyperparameter where the inner Newton would otherwise stall to `max_iter`
# carries negligible importance weight, so the cap bounds its cost without
# moving the k-hat (converged draws keep their exact log-marginal; only the
# negligible-weight tail is truncated).
.joint_attach_pareto_k_multi <- function(res, arms, cp, blocks_spec,
                                         axis_offsets, B, arm_names,
                                         fn_sigma, fn_alpha,
                                         max_iter, tol, n_threads,
                                         force_sparse, cell_coupling,
                                         diagnose_k = TRUE, k_samples = 200L) {
    res$pareto_k        <- NA_real_
    res$pareto_k_is_ess <- NA_real_
    res$pareto_k_scope  <- "outer (hyperparameter) Gaussian proposal"
    if (!isTRUE(diagnose_k)) return(res)

    warm_mode  <- .joint_modal_mode(res)
    k_max_iter <- min(as.integer(max_iter), .K_DIAG_MAX_ITER)

    refit <- function(theta_mat) {
        cpp_grid <- .joint_multi_cpp_grid(theta_mat, axis_offsets, B, cp)
        phi_ppa  <- .joint_multi_phi_per_arm(theta_mat, arm_names)
        r <- .cpp_joint_multi(
            arms_list    = arms,
            copy_arms    = as.integer(cp$copy_arms_zero),
            copy_blocks  = as.integer(cp$copy_blocks_zero),
            blocks_spec  = blocks_spec,
            theta_grid   = cpp_grid,
            axis_offsets = axis_offsets,
            max_iter     = as.integer(k_max_iter),
            tol          = as.numeric(tol),
            n_threads    = as.integer(n_threads),
            x_init_nullable = warm_mode,
            store_Q      = FALSE,
            phi_grid_per_arm = phi_ppa,
            n_threads_outer = 1L,
            tile_ids        = NULL,
            tile_pilot_cells = NULL,
            prune_tol       = 0.0,
            force_sparse    = isTRUE(force_sparse),
            cell_coupling_name = as.character(cell_coupling)
        )
        .joint_multi_add_hp(r$log_marginal, theta_mat, axis_offsets, B,
                            fn_sigma, fn_alpha)
    }
    kd <- .joint_pareto_k(res, refit, k_samples)
    res$pareto_k        <- kd$pareto_k
    res$pareto_k_is_ess <- kd$is_ess
    res
}

# Main multi-block joint dispatch. Mirrors the structure of the
# single-block path (build grid, call C++, post-process) but accepts a
# list-of-blocks `prior_list` and a `copy` spec that points at a
# specific block.
.joint_dispatch_multi <- function(responses, prior_list, copy,
                                  phi_grid,
                                  fn_sigma = NULL,
                                  fn_alpha = NULL,
                                  max_iter, tol, n_threads,
                                  x_init, verbose, store_Q,
                                  n_threads_outer = 1L,
                                  tile_warm = TRUE,
                                  prune_tol = 0.0,
                                  force_sparse = FALSE,
                                  cell_coupling = "separable",
                                  diagnose_k = TRUE,
                                  k_samples = 200L,
                                  inner_refresh = 1L,
                                  integration = "grid",
                                  timer = NULL) {
    tm <- timer %||% .tulpa_timer()                    # gcol33/tulpa#48
    integration <- match.arg(integration, c("grid", "ccd"))
    n_arms <- length(responses)
    arms <- lapply(seq_along(responses), function(k) {
        a <- responses[[k]]
        .normalise_joint_arm_multi(a, k)
    })
    arm_names <- names(responses) %||% paste0("arm", seq_along(responses))

    cp <- .resolve_copy_multi(copy, responses, prior_list)

    # Per-block axis grids (with copy-block parameterisation if applicable).
    B <- length(prior_list)
    per_block <- lapply(seq_len(B), function(b) {
        copy_pos <- if (cp$has_copy) match((b - 1L), cp$copy_blocks_zero)
                    else NA_integer_
        is_copy  <- !is.na(copy_pos)
        alpha_grid_b <- if (is_copy) cp$alpha_grids[[copy_pos]] else numeric(0)
        .joint_block_axis_grid(prior_list[[b]], is_copy, alpha_grid_b, b)
    })
    block_grids <- lapply(per_block, function(x) x$grid)
    prepared    <- lapply(per_block, function(x) x$prepared)

    # Axis layout (independent of node placement: tensor or CCD).
    axis_counts  <- vapply(block_grids, ncol, integer(1))
    axis_offsets <- as.integer(c(0L, cumsum(axis_counts)))
    axis_names <- unlist(lapply(seq_along(block_grids), function(b) {
        cn <- colnames(block_grids[[b]])
        if (is.null(cn) || length(cn) == 0L) return(character(0))
        paste0("b", b, ".", cn)
    }))
    d_axes <- as.integer(axis_offsets[B + 1L])

    blocks_spec <- lapply(seq_along(prepared), function(b) {
        .joint_block_spec_for_cpp(prepared[[b]], n_arms, b, arms = arms)
    })

    # Per-arm dispersion overrides on the outer grid. Resolved up front because
    # CCD declines when a phi axis is active (the phi/latent Cartesian product
    # is a tensor-only construction).
    phi_axes <- .normalise_phi_grid(phi_grid, arm_names)
    has_phi  <- !is.null(phi_axes) &&
                any(vapply(phi_axes, length, integer(1)) > 0L)

    # Node placement. CCD (gcol33/tulpa#59) integrates on a central composite
    # design around the joint hyperparameter mode for d >= 3 transformable
    # axes; it declines (-> tensor grid) for d <= 2, an active phi axis, an
    # unguessable axis (CAR_proper rho_car / non-BYM2 rho), or a degenerate
    # outer mode-find / Hessian.
    dnode                 <- NULL
    tile_partition        <- NULL
    phi_grid_per_arm_list <- NULL
    integration_used      <- "grid"
    joint_grid            <- NULL

    use_ccd <- identical(integration, "ccd") && d_axes >= 3L && !has_phi
    if (use_ccd) {
        block_of_axis <- rep(seq_len(B), times = axis_counts)
        col_within    <- unlist(lapply(axis_counts, seq_len))
        axis_values <- lapply(seq_len(d_axes), function(j) {
            sort(unique(as.numeric(
                block_grids[[block_of_axis[j]]][, col_within[j]])))
        })

        # Warm latent mode for the mode-find / node solves: one solve at the
        # box-centre coordinate, broadcast as x_init so each subsequent inner
        # Newton converges in a few steps. Checkpoint/progress are stripped for
        # every CCD probe so they neither pollute the fit's checkpoint file nor
        # tick its progress bar.
        ccd_warm <- x_init
        with_quiet_opts <- function(expr) {
            op <- options(
                tulpa.nl_checkpoint = list(path = "", resume = TRUE),
                tulpa.nl_progress   = .nl_progress_args(list()))
            on.exit(options(op), add = TRUE)
            force(expr)
        }
        centre_row <- matrix(vapply(axis_values, stats::median, numeric(1)),
                             nrow = 1L, dimnames = list(NULL, axis_names))
        init_fit <- tryCatch(with_quiet_opts(.cpp_joint_multi(
            arms_list = arms, copy_arms = as.integer(cp$copy_arms_zero),
            copy_blocks = as.integer(cp$copy_blocks_zero),
            blocks_spec = blocks_spec,
            theta_grid = .joint_multi_cpp_grid(centre_row, axis_offsets, B, cp),
            axis_offsets = axis_offsets, max_iter = as.integer(max_iter),
            tol = as.numeric(tol), n_threads = as.integer(n_threads),
            x_init_nullable = x_init, store_Q = FALSE, phi_grid_per_arm = NULL,
            n_threads_outer = 1L, tile_ids = NULL, tile_pilot_cells = NULL,
            prune_tol = 0.0, force_sparse = isTRUE(force_sparse),
            cell_coupling_name = as.character(cell_coupling),
            inner_refresh = as.integer(inner_refresh))),
            error = function(e) NULL)
        if (!is.null(init_fit) && is.matrix(init_fit$modes) &&
            nrow(init_fit$modes) >= 1L &&
            is.finite(init_fit$log_marginal[1L])) {
            m1 <- as.numeric(init_fit$modes[1L, ])
            if (all(is.finite(m1))) ccd_warm <- m1
        }

        eval_logpost <- function(theta_mat) {
            r <- with_quiet_opts(.cpp_joint_multi(
                arms_list = arms, copy_arms = as.integer(cp$copy_arms_zero),
                copy_blocks = as.integer(cp$copy_blocks_zero),
                blocks_spec = blocks_spec,
                theta_grid = .joint_multi_cpp_grid(theta_mat, axis_offsets, B, cp),
                axis_offsets = axis_offsets, max_iter = as.integer(max_iter),
                tol = as.numeric(tol), n_threads = as.integer(n_threads),
                x_init_nullable = ccd_warm, store_Q = FALSE,
                phi_grid_per_arm = NULL,
                n_threads_outer = as.integer(n_threads_outer),
                tile_ids = NULL, tile_pilot_cells = NULL, prune_tol = 0.0,
                force_sparse = isTRUE(force_sparse),
                cell_coupling_name = as.character(cell_coupling),
                inner_refresh = as.integer(inner_refresh)))
            .joint_multi_add_hp(r$log_marginal, theta_mat, axis_offsets, B,
                                fn_sigma, fn_alpha)
        }

        ccd <- .joint_ccd_grid(axis_names, axis_offsets, prepared, axis_values,
                               eval_logpost, verbose = verbose)
        if (is.null(ccd)) {
            use_ccd <- FALSE
        } else {
            joint_grid       <- ccd$grid
            dnode            <- ccd$dnode
            integration_used <- "ccd"
        }
    }

    if (!use_ccd) {
        # Cartesian product of per-block axis grids.
        row_counts <- vapply(block_grids, nrow, integer(1))
        idx <- do.call(expand.grid, lapply(row_counts, seq_len))
        n_cells <- nrow(idx)
        if (n_cells > .NL_MULTI_GRID_HARD_CAP) {
            stop(sprintf(
                "Joint multi-block grid has %d cells (hard cap %d). Reduce per-block grid sizes or set control$integration = \"ccd\".",
                n_cells, .NL_MULTI_GRID_HARD_CAP
            ), call. = FALSE)
        }
        if (n_cells > .NL_MULTI_GRID_WARN) {
            warning(sprintf(
                "Joint multi-block grid has %d cells (>%d). Each cell costs one inner Newton solve; reduce per-block grid sizes or set control$integration = \"ccd\".",
                n_cells, .NL_MULTI_GRID_WARN
            ), call. = FALSE)
        }

        joint_grid <- do.call(cbind, lapply(seq_along(block_grids), function(b) {
            block_grids[[b]][idx[[b]], , drop = FALSE]
        }))
        if (ncol(joint_grid) > 0L) colnames(joint_grid) <- axis_names

        # phi_grid (per-arm dispersion overrides on the outer grid): each phi
        # axis varies independently of the latent-block grid, so it forms a
        # Cartesian product with the latent grid and multiplies n_cells. A phi
        # axis tied to a latent-block axis is not expressible on this path; fold
        # it into the relevant block's own axis grid instead.
        if (has_phi) {
            active <- phi_axes[vapply(phi_axes, length, integer(1)) > 0L]
            phi_extra <- do.call(expand.grid,
                                  c(active, list(KEEP.OUT.ATTRS = FALSE,
                                                  stringsAsFactors = FALSE)))
            joint_idx <- expand.grid(joint = seq_len(nrow(joint_grid)),
                                       phi   = seq_len(nrow(phi_extra)),
                                       KEEP.OUT.ATTRS = FALSE)
            joint_grid <- cbind(joint_grid[joint_idx$joint, , drop = FALSE],
                                  as.matrix(phi_extra[joint_idx$phi, , drop = FALSE]))
            phi_cols <- paste0("phi_", names(active))
            colnames(joint_grid) <- c(axis_names, phi_cols)
            # phi columns don't belong to any latent block; the C++ entry
            # consumes them via phi_grid_per_arm, not the block axes.
            phi_grid_per_arm_list <- .joint_multi_phi_per_arm(joint_grid, arm_names)
        }

        # Tile partition for the three-tier warm-start (Phase 2 of
        # dev_notes/speedup.md). Tile axis = every joint_grid column EXCEPT the
        # copy block's alpha column. Built from the *user-facing* (sigma,
        # alpha) grid (before sigma_pos materialisation) so the partition
        # reflects what is constant across alpha at fixed (sigma, rho, ...
        # other-block axes).
        if (isTRUE(tile_warm) && cp$has_copy &&
            length(cp$copy_blocks_zero) == 1L &&
            as.integer(n_threads_outer) > 1L) {
            b_copy_R <- cp$copy_blocks_zero[[1L]] + 1L
            cols_bc  <- (axis_offsets[b_copy_R] + 1L):axis_offsets[b_copy_R + 1L]
            bare_bc  <- sub("^b[0-9]+\\.", "", colnames(joint_grid)[cols_bc])
            i_alpha  <- match("alpha", bare_bc)
            if (!is.na(i_alpha)) {
                alpha_col_idx <- cols_bc[i_alpha]
                non_alpha_idx <- setdiff(seq_len(ncol(joint_grid)), alpha_col_idx)
                tile_partition <- .joint_compute_tile_partition(
                    joint_grid[, non_alpha_idx, drop = FALSE],
                    as.numeric(joint_grid[, alpha_col_idx]),
                    nrow(joint_grid)
                )
            }
        }
    }

    # Build the C++-facing theta_grid (sigma_occ / sigma_pos materialised on
    # each copy block from the user-facing (sigma, alpha) outer grid).
    cpp_grid <- .joint_multi_cpp_grid(joint_grid, axis_offsets, B, cp)

    call_kernel_with_tol <- function(tol_prune) {
        .cpp_joint_multi(
            arms_list    = arms,
            copy_arms    = as.integer(cp$copy_arms_zero),
            copy_blocks  = as.integer(cp$copy_blocks_zero),
            blocks_spec  = blocks_spec,
            theta_grid   = cpp_grid,
            axis_offsets = axis_offsets,
            max_iter     = as.integer(max_iter),
            tol          = as.numeric(tol),
            n_threads    = as.integer(n_threads),
            x_init_nullable = x_init,
            store_Q      = isTRUE(store_Q),
            phi_grid_per_arm = phi_grid_per_arm_list,
            n_threads_outer = as.integer(n_threads_outer),
            tile_ids        = tile_partition$tile_ids,
            tile_pilot_cells = tile_partition$tile_pilot_cells,
            prune_tol       = as.numeric(tol_prune),
            force_sparse    = isTRUE(force_sparse),
            cell_coupling_name = as.character(cell_coupling),
            inner_refresh   = as.integer(inner_refresh)
        )
    }
    tm$mark("setup")
    res <- call_kernel_with_tol(prune_tol)
    # Safety gate (gcol33/tulpa#43): fall back to the full grid when the
    # cheap-pass ranking is unreliable rather than silently returning a
    # pruned answer.
    if (as.numeric(prune_tol) > 0) {
        res <- .joint_prune_safety_gate(
            res, resolve_full = function() call_kernel_with_tol(0.0))
    }
    tm$mark("grid")

    # Bake the regularizing hyperprior on (sigma, alpha) into log_marginal
    # (gcol33/tulpa#22). Multi-block has no in-package refinement passes,
    # so one apply at the kernel-call boundary suffices.
    res$log_marginal <- .joint_multi_add_hp(res$log_marginal, joint_grid,
                                            axis_offsets, B, fn_sigma, fn_alpha)

    res$theta_grid   <- joint_grid
    res$theta_names  <- colnames(joint_grid)
    res$axis_offsets <- axis_offsets
    res$integration  <- integration_used
    # Integration weights fold in the CCD design weights (`dnode`); for the
    # tensor grid `dnode` is NULL and this is the plain log-marginal softmax.
    res$weights      <- .joint_integration_weights(res$log_marginal, dnode)
    is_ccd <- identical(integration_used, "ccd")
    res <- .joint_posterior_moments_multi(
        res, prepared, axis_offsets, joint_grid, cp,
        int_weights = if (is_ccd) res$weights else NULL)
    if (!is_ccd) {
        # Replace per-axis var-of-means SDs with Laplace-at-mode SDs at the
        # modal cell (gcol33/tulpa#20). The 3-point grid-profile fit needs the
        # regular per-axis lattice; on the scattered CCD design the weighted
        # var-of-means over the design IS the calibrated SD (the corrected
        # design weights reproduce the Gaussian moments), so it is kept as is.
        res <- .nl_refit_axis_sd_laplace(res)
    }
    res$arm_layout    <- .joint_multi_layout(arms, prepared)
    res$blocks        <- prepared
    res$prior         <- prior_list
    res$responses     <- responses
    res$copy          <- copy
    res$cell_coupling <- cell_coupling
    tm$mark("postproc")
    # Outer Pareto-k-hat: re-evaluate the inner joint marginal at sampled
    # hyperparameters via the shared cpp-grid / phi / hyperprior helpers and
    # PSIS-smooth. Declines (NA -> quad-ESS) when a block carries an axis with
    # unguessable support (CAR_proper's rho_car, a non-BYM2 rho, ...).
    res <- .joint_attach_pareto_k_multi(res, arms, cp, blocks_spec,
                                        axis_offsets, B, arm_names,
                                        fn_sigma, fn_alpha,
                                        max_iter, tol, n_threads,
                                        force_sparse, cell_coupling,
                                        diagnose_k = diagnose_k,
                                        k_samples  = k_samples)
    tm$mark("diagnostics")
    res$timing <- tm$timing()
    class(res) <- c("tulpa_nested_laplace_joint_multi",
                    "tulpa_nested_laplace_joint",
                    "tulpa_nested_laplace", "list")
    res
}

# Latent-vector layout for the multi-block joint result. Mirrors the
# single-block `.joint_layout()` but generalised over `prepared` blocks:
# per-arm beta, per-arm RE, then each prepared block in order. For
# back-compat with single-block consumers (e.g. tulpaObs cover-hurdle's
# `.joint_field_at_obs_copy_multi`), the first spatial-like block also
# emits `phi_start` / `theta_start` aliases:
#
#   * BYM2  -> phi_start, theta_start (length-2 sub-block).
#   * ICAR / CAR_proper -> phi_start (length-1).
#
# Non-spatial blocks expose `block_start[b]` so callers can index into
# any block by ordinal position.
.joint_multi_layout <- function(arms, prepared) {
    n_arms <- length(arms)
    p_arm  <- vapply(arms, function(a) ncol(a$X),     integer(1))
    n_re   <- vapply(arms, function(a) a$n_re_groups, integer(1))

    beta_start <- integer(n_arms)
    cur <- 0L
    for (k in seq_len(n_arms)) {
        beta_start[k] <- cur
        cur <- cur + p_arm[k]
    }
    re_start <- integer(n_arms)
    for (k in seq_len(n_arms)) {
        re_start[k] <- cur
        cur <- cur + n_re[k]
    }

    B <- length(prepared)
    block_start <- integer(B)
    block_size  <- integer(B)
    phi_start   <- NULL
    theta_start <- NULL
    # Per-spatial-block latent offset (0-based) of the structured field, in
    # block order: ICAR / CAR_proper at the block start, BYM2 at its phi
    # sub-block start. `phi_start` (back-compat) is field_starts[1].
    field_starts      <- integer(0)
    field_block_types <- character(0)
    for (b in seq_len(B)) {
        block_start[b] <- cur
        type <- tolower(prepared[[b]]$type %||% "")
        # Latent size per block. BYM2 is length-2 (phi ICAR + theta IID)
        # on n_spatial_units; all other blocks are length-1 on the
        # block's `n_spatial_units` / `n_times` / `n_units`.
        n_units <- prepared[[b]]$n_spatial_units %||%
                   prepared[[b]]$n_times         %||%
                   prepared[[b]]$n_units         %||%
                   prepared[[b]]$n_latent        %||%
                   prepared[[b]]$n_mesh          %||%
                   prepared[[b]]$m_total         %||% 0L
        n_units <- as.integer(n_units)
        if (type == "bym2") {
            block_size[b] <- 2L * n_units
            field_starts      <- c(field_starts, cur)
            field_block_types <- c(field_block_types, "bym2")
            if (is.null(phi_start)) {
                phi_start   <- cur
                theta_start <- cur + n_units
            }
        } else if (type == "lf") {
            # Latent factor block stores u (n_latent) followed by lambda
            # (n_arms). Total size = n_latent + n_arms.
            block_size[b] <- n_units + n_arms
        } else if (type == "hsgp_mo") {
            # Multi-output HSGP stores K * M coefficients in output-major
            # order: beta_{k, m} at offset k * M + m. K = n_arms.
            block_size[b] <- n_arms * n_units
        } else {
            block_size[b] <- n_units
            if (type %in% c("icar", "car_proper")) {
                field_starts      <- c(field_starts, cur)
                field_block_types <- c(field_block_types, type)
                if (is.null(phi_start)) {
                    phi_start <- cur
                }
            }
        }
        cur <- cur + block_size[b]
    }
    n_x <- cur

    out <- list(
        n_arms      = n_arms,
        p           = p_arm,
        n_re        = n_re,
        beta_start  = beta_start,
        re_start    = re_start,
        block_start = block_start,
        block_size  = block_size,
        n_x         = n_x
    )
    if (!is.null(phi_start))   out$phi_start   <- phi_start
    if (!is.null(theta_start)) out$theta_start <- theta_start
    if (length(field_starts) > 0L) {
        out$field_starts      <- field_starts
        out$field_block_types <- field_block_types
    }
    out
}

# Posterior moments for the multi-block joint grid. Mirrors the single-arm
# multi-block helper (`.nl_posterior_moments_multi`) and additionally
# attaches the weighted-quantile median + 2.5/97.5 CI for every joint-grid
# axis -- the calibrated headline summary for right-skewed scale-like
# hyperparameters (alpha at small n_pos, sigma/range/phi near boundary).
.joint_posterior_moments_multi <- function(res, prepared, axis_offsets,
                                            joint_grid, cp,
                                            int_weights = NULL) {
    w <- res$weights
    # Joint moments across every column of joint_grid (including phi
    # columns appended for per-arm dispersion overrides).
    res$theta_mean <- as.numeric(crossprod(w, joint_grid))
    names(res$theta_mean) <- colnames(joint_grid)
    res$theta_sd <- sqrt(pmax(0, as.numeric(crossprod(w, joint_grid^2)) -
                                  res$theta_mean^2))
    names(res$theta_sd) <- colnames(joint_grid)

    B <- length(prepared)
    per_block_moments <- vector("list", B)
    for (b in seq_len(B)) {
        n_axes_b <- axis_offsets[b + 1L] - axis_offsets[b]
        if (n_axes_b == 0L) {
            # Block contributes no outer-grid axes (e.g. type = "lf"):
            # empty moments, empty axis_cols.
            per_block_moments[[b]] <- list(
                type      = tolower(prepared[[b]]$type),
                mean      = numeric(0),
                sd        = numeric(0),
                axis_cols = integer(0)
            )
            next
        }
        cols <- seq.int(axis_offsets[b] + 1L, axis_offsets[b + 1L])
        sub <- joint_grid[, cols, drop = FALSE]
        block_mean <- as.numeric(crossprod(w, sub))
        block_sd   <- sqrt(pmax(0, as.numeric(crossprod(w, sub^2)) - block_mean^2))
        # Strip the "b<N>." prefix on per-block moment names — block index
        # is implicit in the list position.
        bare_names <- sub("^b[0-9]+\\.", "", colnames(sub))
        names(block_mean) <- bare_names
        names(block_sd)   <- bare_names
        per_block_moments[[b]] <- list(
            type      = tolower(prepared[[b]]$type),
            mean      = block_mean,
            sd        = block_sd,
            axis_cols = cols
        )
    }
    res$block_moments <- per_block_moments

    # Weighted-quantile median + 2.5/97.5 CI per axis. Generic helper
    # filters foreign-axis slice cells per axis name. After the (sigma,
    # alpha) reparameterization, alpha is a column of joint_grid like
    # any other axis; this attaches the calibrated summary for every
    # axis, not just alpha.
    qs <- .nl_axis_quantiles(joint_grid, res$log_marginal,
                              res$refining_axis, weights = int_weights)
    res$theta_median <- qs$median
    res$theta_ci_lo  <- qs$ci_lo
    res$theta_ci_hi  <- qs$ci_hi
    res
}
