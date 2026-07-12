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
# std::vector<LatentBlock> on the joint side. Each block can be a copy
# block; copy (alpha-coupling) is supported on areal spatial (icar / bym2 /
# car_proper), temporal (rw1 / rw2 / ar1), and unstructured (iid) blocks
# (gcol33/tulpa#76). See dev_notes/plan_multi_block_joint.md.

# Validate one arm spec for the multi-block path. Same core as
# `.normalise_joint_arm` *except* `spatial_idx` is no longer required at
# the arm level — per-arm idx vectors live inside each block spec instead,
# and a missing spatial_idx gets a length-N placeholder of zeros so the
# existing JointArm packaging path doesn't complain.
.normalise_joint_arm_multi <- function(a, k) {
    .normalise_joint_arm_core(a, k, spatial_idx = "optional")
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
             "multi-block joint driver.", call. = FALSE)
    }
    block_zero <- as.integer(block_id) - 1L
    if (block_zero < 0L || block_zero >= length(prior_list)) {
        stop("`copy$block` index (", block_id, ") out of range for length(prior) (",
             length(prior_list), ").", call. = FALSE)
    }
    block_type <- tolower(prior_list[[block_zero + 1L]]$type)
    # Copy (alpha-coupling) is supported on any block whose amplitude is a
    # single scalar the donor / copy arms scale independently: areal spatial
    # (icar / bym2 / car_proper), temporal (rw1 / rw2 / ar1), unstructured
    # (iid), and the separable correlated areal field (mcar). For mcar the copy
    # scales the WHOLE correlated (intercept, slope) field by one alpha onto the
    # copy arm (the cross-arm transfer); the free cross-covariance Sigma stays
    # the within-arm covariance among the fields, integrated over the outer grid
    # (gcol33/tulpaObs#64). Blocks whose per-arm scaling lives elsewhere (lf via
    # lambda, hsgp_mo via Sigma) or whose precision is precomputed (tgmrf) do
    # not take a copy (gcol33/tulpa#76).
    copy_types <- c("icar", "bym2", "car_proper", "rw1", "rw2", "ar1", "iid",
                    "mcar", "miid")
    if (!block_type %in% copy_types) {
        stop("`copy$block` points at type '", block_type, "'. Copy semantics ",
             "are supported on types (",
             paste(shQuote(copy_types), collapse = ", "), "). ",
             "Blocks with their own per-arm scaling (lf, hsgp_mo) or a ",
             "precomputed precision (tgmrf) do not take a copy.", call. = FALSE)
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
# spec must name a distinct copy-supported block. Returns vectors parallel
# over the copy blocks:
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
             "). Each coupled field must name a distinct block.",
             call. = FALSE)
    }
    list(has_copy = TRUE, copy_arms_zero = copy_arms_zero,
         copy_blocks_zero = copy_blocks_zero, alpha_grids = alpha_grids)
}

# Per-block axis grid for the multi-block joint driver. When the block is a
# copy block (any of icar / bym2 / car_proper / rw1 / rw2 / ar1 / iid), the
# parameterisation uses (sigma, alpha[, rho/rho_car]) directly. Non-copy
# blocks use the standard single-arm conventions and reuse the `.NL_REGISTRY`
# defaults.
#
# Returns:
#   $grid    : matrix [n_block_cells x n_axes_for_block]
#   $names   : axis names
#   $prepared: block spec with defaults filled in (so downstream code can
#              read prior$adj_row_ptr etc. without re-checking presence)
.joint_block_axis_grid <- function(p, is_copy, alpha_grid,
                                    block_index) {
    type <- tolower(p$type)
    if (is_copy && type %in% c("mcar", "miid")) {
        # Copied correlated field (mcar: areal; miid: per-group RE) -- the
        # block's own axes are the p(p+1)/2 log-Cholesky coordinates of Sigma
        # (the within-arm covariance among the fields); the copy appends ONE
        # trailing `alpha`
        # axis carrying the cross-arm copy amplitude. The kernel reads the
        # log-Cholesky axes at axis0 and alpha at axis0 + p(p+1)/2 (the last
        # column), so alpha MUST be appended last. Sigma stays a free outer-grid
        # quantity (NOT a copy-materialized sigma_occ/sigma_pos pair); alpha is
        # applied directly via arm_scale, not multiplied by a sigma.
        base <- .nl_block_axis_grid(p)
        bg   <- base$grid
        ag   <- as.numeric(alpha_grid)
        idx  <- expand.grid(b = seq_len(nrow(bg)), a = seq_along(ag),
                            KEEP.OUT.ATTRS = FALSE)
        grid <- cbind(bg[idx$b, , drop = FALSE], alpha = ag[idx$a])
        colnames(grid) <- c(colnames(bg), "alpha")
        return(list(grid = as.matrix(grid), names = colnames(grid),
                    prepared = base$prepared))
    }
    if (is_copy) {
        sigma_axis <- p$sigma_grid
        if (is.null(sigma_axis)) {
            sigma_axis <- exp(seq(log(0.1), log(3), length.out = 5))
        }
        sigma_axis <- as.numeric(sigma_axis)
        # Copy axes always lead with (sigma, alpha); the kernel materializes
        # sigma_donor = sigma and sigma_copy = alpha * sigma in those two
        # leading columns (see .joint_multi_cpp_grid), so any trailing
        # correlation axis (rho / rho_car) must follow them. A unit-precision
        # block (icar / rw1 / rw2 / iid) has just (sigma, alpha); a block with a
        # second hyperparameter (bym2 rho, car_proper rho_car, ar1 rho) appends
        # it (gcol33/tulpa#76).
        if (type %in% c("icar", "rw1", "rw2", "iid")) {
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
        } else if (type == "ar1") {
            rho <- p$rho_grid %||% c(0.0, 0.4, 0.7, 0.9, 0.97)
            gr <- expand.grid(sigma = sigma_axis,
                              alpha = alpha_grid,
                              rho   = as.numeric(rho),
                              KEEP.OUT.ATTRS = FALSE,
                              stringsAsFactors = FALSE)
        } else {
            stop("Block ", block_index, ": copy semantics are not supported ",
                 "for type '", type, "'.", call. = FALSE)
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
        # Replicated CAR (`by =`): number of disjoint equal-size components in the
        # block-diagonal graph (the intrinsic ICAR kernel pins each per component
        # and uses (n - L) in its rank-deficiency normalizer). Absent = connected.
        if (!is.null(p$n_components)) {
            out$n_components <- as.integer(p$n_components)
        }
        out
    } else if (type == "mcar") {
        # Separable multivariate CAR: p coupled areal fields sharing Sigma (x)
        # Q^-1. One block over p * n_spatial_units latent. spatial_idx is the
        # per-arm 1-based cell index (shared by all fields); field_weight is the
        # per-field design column (the intercept's is all-ones, a covariate
        # column is the per-row value) -- outer length p, inner length n_arms.
        if (is.null(p$spatial_idx)) {
            stop("Block ", block_index, " (type 'mcar'): `spatial_idx` is ",
                 "required as a list of length n_arms.", call. = FALSE)
        }
        spatial_idx <- .multi_block_per_arm_idx(p$spatial_idx, n_arms,
                                                  block_index, "spatial_idx")
        n_fields <- as.integer(p$n_fields)
        if (is.null(p$field_weight) || length(p$field_weight) != n_fields) {
            stop("Block ", block_index, " (type 'mcar'): `field_weight` must be ",
                 "a list of length n_fields (", n_fields, ").", call. = FALSE)
        }
        field_weight <- lapply(seq_len(n_fields), function(a) {
            .multi_block_svc_weight(p$field_weight[[a]], spatial_idx, n_arms,
                                    block_index)
        })
        out <- list(
            type            = "mcar",
            spatial_idx     = spatial_idx,
            n_spatial_units = as.integer(p$n_spatial_units),
            n_fields        = n_fields,
            adj_row_ptr     = as.integer(p$adj_row_ptr),
            adj_col_idx     = as.integer(p$adj_col_idx),
            n_neighbors     = as.integer(p$n_neighbors),
            field_weight    = field_weight
        )
        # Replicated MCAR (`by =`): L equal-size components per field.
        if (!is.null(p$n_components)) {
            out$n_components <- as.integer(p$n_components)
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
        # Optional per-arm per-row design weight (temporal SVC). Parallel to the
        # areal branch: the field's contribution to arm k row i is row-scaled by
        # svc_weight[[k]][i] (eta_i += svc_weight[[k]][i] * amplitude *
        # f[time_i]), used for a temporally varying slope on a covariate column.
        if (!is.null(p$svc_weight)) {
            out$svc_weight <- .multi_block_svc_weight(
                p$svc_weight, temporal_idx, n_arms, block_index)
        }
        out
    } else if (type == "iid") {
        if (is.null(p$obs_idx)) {
            stop("Block ", block_index, " (type 'iid'): ",
                 "`obs_idx` is required as a list of length n_arms.",
                 call. = FALSE)
        }
        obs_idx <- .multi_block_per_arm_idx(p$obs_idx, n_arms,
                                              block_index, "obs_idx")
        out <- list(
            type    = "iid",
            obs_idx = obs_idx,
            n_units = as.integer(p$n_units)
        )
        # Optional per-arm per-row design weight (random slope on a covariate
        # column). When set, the field's contribution to arm k row i is
        # row-scaled by svc_weight[[k]][i]:
        #   eta_i += svc_weight[[k]][i] * sigma * u[obs_idx_i].
        # Parallel to the areal / temporal SVC branches: a list of length n_arms,
        # each a numeric vector matching that arm's obs count (the per-arm obs_idx
        # length). Composes multiplicatively with copy `arm_scale`. An
        # uncorrelated slope (x || g) / (0 + x | g) is one weighted iid block per
        # coefficient, each with its own sigma axis (gcol33/tulpa#114).
        if (!is.null(p$svc_weight)) {
            out$svc_weight <- .multi_block_svc_weight(
                p$svc_weight, obs_idx, n_arms, block_index)
        }
        out
    } else if (type == "miid") {
        # Multivariate IID (gcol33/tulpa#114): p coupled per-group coefficient
        # fields sharing a free Sigma (the Q = I sibling of mcar). One block over
        # p * n_groups latent. `obs_idx` is the per-arm 1-based grouping index
        # (the mcar `spatial_idx` role); `field_weight` is the per-coefficient
        # design column (intercept all-ones, slope = covariate) -- outer length
        # n_fields, inner length n_arms. Expresses a correlated random slope
        # (1 + x | g): n_fields = 1 + n_slopes.
        if (is.null(p$obs_idx)) {
            stop("Block ", block_index, " (type 'miid'): `obs_idx` is ",
                 "required as a list of length n_arms.", call. = FALSE)
        }
        obs_idx <- .multi_block_per_arm_idx(p$obs_idx, n_arms,
                                              block_index, "obs_idx")
        n_fields <- as.integer(p$n_fields)
        if (is.null(p$field_weight) || length(p$field_weight) != n_fields) {
            stop("Block ", block_index, " (type 'miid'): `field_weight` must be ",
                 "a list of length n_fields (", n_fields, ").", call. = FALSE)
        }
        field_weight <- lapply(seq_len(n_fields), function(a) {
            .multi_block_svc_weight(p$field_weight[[a]], obs_idx, n_arms,
                                    block_index)
        })
        list(
            type         = "miid",
            obs_idx      = obs_idx,
            n_groups     = as.integer(p$n_groups),
            n_fields     = n_fields,
            field_weight = field_weight
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
        # A copied correlated field (mcar, gcol33/tulpaObs#64) has no scalar
        # `sigma` axis -- its amplitude lives in the free Sigma (the log-Cholesky
        # axes), and the copy applies alpha directly via arm_scale (NOT
        # sigma_occ = sigma / sigma_pos = alpha*sigma). The alpha column passes
        # through to the kernel unchanged; the log-Cholesky axes pass through too.
        if (is.na(i_sigma) && !is.na(i_alpha)) next
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
                                fn_sigma, fn_alpha, fn_phi = NULL) {
    if (is.null(fn_sigma) && is.null(fn_alpha) && is.null(fn_phi))
        return(log_marginal)
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
    # phi_<arm> dispersion axes are not block-prefixed; carry them straight
    # through so a single `fn_phi` re-weights each on the joint grid.
    phi_cols <- if (is.null(fn_phi)) character(0)
                else grep("^phi_", colnames(joint_grid), value = TRUE)
    if (length(view_map) == 0L && length(phi_cols) == 0L) return(log_marginal)
    view <- joint_grid[, c(view_map, match(phi_cols, colnames(joint_grid))),
                       drop = FALSE]
    colnames(view) <- c(names(view_map), phi_cols)
    hp <- .joint_hp_vec_for_grids(view, fn_sigma, fn_alpha, fn_phi)
    if (!is.null(hp) && length(hp) == length(log_marginal)) {
        log_marginal <- log_marginal + hp
    }
    log_marginal
}

# Multi-block joint outer Pareto-k-hat. Builds the re-evaluation closure
# (round-trips a sampled user-facing `joint_grid` through the SAME kernel via
# the shared cpp-grid / phi / hyperprior helpers) and defers to the
# block-type-aware driver `.joint_pareto_k`. The whole importance batch is
# re-solved in one kernel call across `n_threads_outer`, so the independent
# re-solves run concurrently across cores rather than one-at-a-time using all
# inner threads. No tiling on the re-eval path (the IS batch is not a per-axis
# alpha lattice), so no tile-partition reconstruction is needed.
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
                                         fn_sigma, fn_alpha, fn_phi = NULL,
                                         max_iter, tol, n_threads,
                                         force_sparse, cell_coupling,
                                         diagnose_k = TRUE, diagnose_draws = 500L,
                                         n_threads_outer = 1L, proposal = NULL,
                                         pareto_k_by_arm = FALSE,
                                         k_bootstrap = 1000L, k_tail_points = NULL,
                                         k_conf_bands = NULL) {
    res$pareto_k        <- NA_real_
    res$pareto_k_is_ess <- NA_real_
    res$pareto_k_scope  <- "outer (hyperparameter) Gaussian proposal"
    res$pareto_k_proposal_source <- NA_character_
    if (!isTRUE(diagnose_k)) return(res)

    warm_mode   <- .joint_modal_mode(res)
    modal_theta <- .joint_modal_theta(res)
    k_max_iter  <- min(as.integer(max_iter), .K_DIAG_MAX_ITER)
    n_to        <- as.integer(n_threads_outer)
    knobs       <- .kdiag_knobs()

    # One kernel call over the importance batch. Shamanskii reuse makes the
    # off-factor steps scatter grad-only (attacks the dominant per-iteration
    # Hessian fill); loosened inner tol cuts intrinsic Newton steps; per-cell
    # warm start (`x_init_per_cell`, an [S x n_x] matrix or NULL) starts each
    # draw from its nearest grid mode in BOTH serial and parallel (gcol33/tulpa#118).
    solve_fn <- function(theta_mat, x_init_per_cell = NULL) {
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
            tol          = max(knobs$tol, as.numeric(tol)),
            n_threads    = as.integer(n_threads),
            x_init_nullable = warm_mode,
            store_Q      = FALSE,
            phi_grid_per_arm = phi_ppa,
            n_threads_outer = n_to,
            tile_ids        = NULL,
            tile_pilot_cells = NULL,
            prune_tol       = 0.0,
            force_sparse    = isTRUE(force_sparse),
            cell_coupling_name = as.character(cell_coupling),
            inner_refresh   = knobs$refresh,
            x_init_per_cell = x_init_per_cell
        )
        .joint_multi_add_hp(r$log_marginal, theta_mat, axis_offsets, B,
                            fn_sigma, fn_alpha, fn_phi)
    }
    # Per-cell warm start (nearest grid mode, serial + parallel) when modes are
    # stored; else near-neighbour chain re-order; else plain broadcast
    # (gcol33/tulpa#118).
    refit <- .joint_make_diag_refit(res, solve_fn, modal_theta, knobs)
    arm_axes <- if (isTRUE(pareto_k_by_arm)) .joint_pareto_arm_axes(res) else NULL
    kd <- .joint_pareto_k(res, refit, diagnose_draws, proposal = proposal,
                          arm_axes = arm_axes, k_bootstrap = k_bootstrap,
                          k_tail_points = k_tail_points, k_conf_bands = k_conf_bands)
    res$pareto_k        <- kd$pareto_k
    res$pareto_k_is_ess <- kd$is_ess
    res$pareto_k_proposal_source <- kd$proposal_source
    res <- .joint_attach_pareto_k_uncertainty(res, kd)
    res <- .joint_attach_by_arm_k(res, kd)
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
                                  fn_phi = NULL,
                                  max_iter, tol, n_threads,
                                  x_init, verbose, store_Q,
                                  n_threads_outer = 1L,
                                  tile_warm = TRUE,
                                  prune_tol = 0.0,
                                  force_sparse = FALSE,
                                  cell_coupling = "separable",
                                  diagnose_k = TRUE,
                                  diagnose_draws = 500L,
                                  pareto_k_by_arm = FALSE,
                                  pareto_k_threads = NULL,
                                  k_bootstrap = 1000L,
                                  k_tail_points = NULL,
                                  k_conf_bands = NULL,
                                  inner_refresh = 1L,
                                  integration = "auto",
                                  local_ccd = NULL,
                                  adaptive_cutoff = 10,
                                  adaptive_stride = 2L,
                                  adaptive_max_frac = 0.75,
                                  timer = NULL) {
    tm <- timer %||% .tulpa_timer()                    # gcol33/tulpa#48
    integration <- match.arg(integration, c("auto", "ccd", "grid",
                                             "grid_adaptive"))
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

    # Per-latent-axis fine grid values (sorted unique), shared by the CCD and the
    # adaptive-lattice integrators to place / locate the outer nodes.
    block_of_axis      <- rep(seq_len(B), times = axis_counts)
    col_within         <- unlist(lapply(axis_counts, seq_len))
    latent_axis_values <- lapply(seq_len(d_axes), function(j) {
        sort(unique(as.numeric(block_grids[[block_of_axis[j]]][, col_within[j]])))
    })

    # Per-arm dispersion overrides on the outer grid. A phi axis varies
    # independently of the latent-block axes, so it forms a Cartesian (tensor)
    # product with the latent grid -- whether that latent grid is the dense
    # tensor or a CCD design (gcol33/tulpa#61). The cross is applied once,
    # below, to whichever latent grid was built.
    phi_axes <- .normalise_phi_grid(phi_grid, arm_names)
    has_phi  <- !is.null(phi_axes) &&
                any(vapply(phi_axes, length, integer(1)) > 0L)

    # Node placement for the LATENT axes. CCD (gcol33/tulpa#59) integrates on a
    # central composite design around the joint hyperparameter mode; it declines
    # (-> tensor grid) for too few axes, an unguessable axis (CAR_proper rho_car
    # / non-BYM2 rho), or a degenerate / ridged outer mode-find / Hessian
    # (gcol33/tulpa#62). An active phi axis no longer disables CCD
    # (gcol33/tulpa#61): the CCD rides the latent axes and the phi tensor
    # crosses on top, with the CCD design weights replicated across phi cells.
    # Axis threshold: "auto" (the default) engages CCD only at d >= 4, where the
    # tensor blow-up bites hardest, and keeps the cheaper, ridge-robust tensor
    # grid at d <= 3; explicit "ccd" lowers the threshold to d >= 3.
    dnode                 <- NULL
    tile_partition        <- NULL
    phi_grid_per_arm_list <- NULL
    integration_used      <- "grid"
    joint_grid            <- NULL
    use_adaptive          <- FALSE
    adaptive_info         <- NULL
    # Mode-Hessian outer proposal (gcol33/tulpa#116). Set only when the CCD
    # integrator engages: its Gaussian (centre `u_hat`, scale `L_scale` from the
    # analytic curvature at the outer mode) drives the outer Pareto-k so the
    # diagnostic stays defined when the hyperparameter posterior is sharp and the
    # grid concentrates on ~1 cell (where the grid-weighted covariance collapses).
    ccd_proposal          <- NULL

    ccd_requested <- .joint_ccd_engage(integration, d_axes)
    use_ccd <- ccd_requested
    if (use_ccd) {
        axis_values <- latent_axis_values

        # Warm latent mode for the mode-find / node solves: one solve at the
        # box-centre coordinate, broadcast as x_init so each subsequent inner
        # Newton converges in a few steps. Checkpoint/progress are stripped for
        # every CCD probe so they neither pollute the fit's checkpoint file nor
        # tick its progress bar.
        ccd_warm <- x_init
        with_quiet_opts <- function(expr) {
            op <- options(
                tulpa.nl_checkpoint = list(path = "", resume = TRUE),
                tulpa.nl_progress   = .nl_progress_args(list(progress = FALSE)))
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
            lp <- .joint_multi_add_hp(r$log_marginal, theta_mat, axis_offsets, B,
                                      fn_sigma, fn_alpha, fn_phi)
            # Carry the inner latent modes so the CCD mode-find can advance the
            # warm start per accepted point (gcol33/tulpa#62).
            if (is.matrix(r$modes)) attr(lp, "modes") <- r$modes
            lp
        }

        # Advance the inner warm start to the latent mode at each accepted
        # mode-find point, so subsequent probes solve in a few Newton steps
        # rather than cold from the box centre (gcol33/tulpa#62).
        set_warm <- function(mode) {
            mode <- as.numeric(mode)
            if (length(mode) && all(is.finite(mode))) ccd_warm <<- mode
        }

        ccd <- .joint_ccd_grid(axis_names, axis_offsets, prepared, axis_values,
                               eval_logpost, verbose = verbose,
                               set_warm = set_warm)
        if (is.null(ccd)) {
            use_ccd <- FALSE
        } else {
            joint_grid       <- ccd$grid
            dnode            <- ccd$dnode
            integration_used <- "ccd"
            # The CCD axes are exactly the leading latent-block columns of
            # joint_grid (axis_names, in order); phi crosses as a separate
            # tensor on top. Carry the mode-Hessian Gaussian over those axes for
            # the outer Pareto-k (gcol33/tulpa#116).
            ccd_proposal <- list(u_hat   = ccd$u_hat,
                                 L_scale = ccd$L_scale,
                                 tags    = ccd$tags,
                                 cols    = seq_len(d_axes))
        }
    }

    # Adaptive lattice (nested_laplace_joint_adaptive.R): the low-dimensional
    # companion to the CCD. It evaluates a coarse subsample of the full outer
    # lattice (latent block axes AND phi axes together) to locate the posterior
    # mass, then floods outward on the fine lattice keeping only the cells within
    # a log-density cutoff of the peak -- so the far tail, which every dense cell
    # still pays a full inner solve for, is never placed. The selected cells are
    # a strict subset of the dense tensor at uniform lattice spacing, so the main
    # kernel call below evaluates them with the full contract (store_Q, phi, tile
    # warm start) and the integration weight stays the plain softmax (dnode NULL);
    # the flood solves here are cheap grid selection (store_Q off, quiet). Declines
    # to the dense tensor on a degenerate lattice or a diffuse posterior whose kept
    # region rivals the dense grid.
    if (!use_ccd && .joint_adaptive_engage(integration, d_axes)) {
        if (has_phi) {
            active_ad     <- phi_axes[vapply(phi_axes, length, integer(1)) > 0L]
            phi_axis_vals <- lapply(active_ad, function(v)
                sort(unique(as.numeric(v))))
            phi_cols_ad   <- paste0("phi_", names(active_ad))
        } else {
            phi_axis_vals <- list()
            phi_cols_ad   <- character(0)
        }
        ad_axis_values <- c(latent_axis_values, phi_axis_vals)
        ad_col_names   <- c(axis_names, phi_cols_ad)

        # One warm centre solve, broadcast as x_init so each flood-selection solve
        # converges in a few Newton steps rather than cold from the box centre.
        ad_warm <- x_init
        centre_theta <- matrix(
            vapply(ad_axis_values, stats::median, numeric(1)), nrow = 1L,
            dimnames = list(NULL, ad_col_names))
        init_ad <- tryCatch(.joint_with_quiet_opts(.cpp_joint_multi(
            arms_list = arms, copy_arms = as.integer(cp$copy_arms_zero),
            copy_blocks = as.integer(cp$copy_blocks_zero), blocks_spec = blocks_spec,
            theta_grid = .joint_multi_cpp_grid(centre_theta, axis_offsets, B, cp),
            axis_offsets = axis_offsets, max_iter = as.integer(max_iter),
            tol = as.numeric(tol), n_threads = as.integer(n_threads),
            x_init_nullable = x_init, store_Q = FALSE,
            phi_grid_per_arm = .joint_multi_phi_per_arm(centre_theta, arm_names),
            n_threads_outer = 1L, tile_ids = NULL, tile_pilot_cells = NULL,
            prune_tol = 0.0, force_sparse = isTRUE(force_sparse),
            cell_coupling_name = as.character(cell_coupling),
            inner_refresh = as.integer(inner_refresh))),
            error = function(e) NULL)
        if (!is.null(init_ad) && is.matrix(init_ad$modes) &&
            nrow(init_ad$modes) >= 1L && is.finite(init_ad$log_marginal[1L])) {
            m1 <- as.numeric(init_ad$modes[1L, ])
            if (all(is.finite(m1))) ad_warm <- m1
        }

        # Grid-selection evaluator: physical theta (latent + phi columns) -> the
        # inner joint log-marginal with the hyperprior baked in. store_Q off and
        # quiet, since these solves only rank cells; the authoritative solve (with
        # store_Q) is the main kernel call over the selected grid.
        eval_theta_ad <- function(theta_mat) {
            r <- .joint_with_quiet_opts(.cpp_joint_multi(
                arms_list = arms, copy_arms = as.integer(cp$copy_arms_zero),
                copy_blocks = as.integer(cp$copy_blocks_zero),
                blocks_spec = blocks_spec,
                theta_grid = .joint_multi_cpp_grid(theta_mat, axis_offsets, B, cp),
                axis_offsets = axis_offsets, max_iter = as.integer(max_iter),
                tol = as.numeric(tol), n_threads = as.integer(n_threads),
                x_init_nullable = ad_warm, store_Q = FALSE,
                phi_grid_per_arm = .joint_multi_phi_per_arm(theta_mat, arm_names),
                n_threads_outer = as.integer(n_threads_outer),
                tile_ids = NULL, tile_pilot_cells = NULL, prune_tol = 0.0,
                force_sparse = isTRUE(force_sparse),
                cell_coupling_name = as.character(cell_coupling),
                inner_refresh = as.integer(inner_refresh)))
            list(log_marginal = .joint_multi_add_hp(
                     r$log_marginal, theta_mat, axis_offsets, B,
                     fn_sigma, fn_alpha, fn_phi),
                 modes = NULL)
        }

        ad <- .joint_adaptive_grid(ad_axis_values, ad_col_names, eval_theta_ad,
                                   cutoff   = adaptive_cutoff,
                                   stride   = adaptive_stride,
                                   max_frac = adaptive_max_frac,
                                   verbose  = verbose)
        if (!is.null(ad)) {
            joint_grid            <- ad$grid
            integration_used      <- "grid_adaptive"
            use_adaptive          <- TRUE
            adaptive_info         <- ad$info
            # phi columns are already part of the adaptive grid; extract them for
            # the C++ call and skip the phi tensor cross below.
            phi_grid_per_arm_list <- .joint_multi_phi_per_arm(joint_grid, arm_names)
        }
    }

    if (!use_ccd && !use_adaptive) {
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
    }

    # Latent node / cell count, before the phi tensor crosses on top -- the CCD
    # node count or the tensor latent-cell count, for the verbose announcement.
    n_latent_cells <- nrow(joint_grid)

    # phi_grid (per-arm dispersion overrides on the outer grid): each phi axis
    # varies independently of the latent-block grid, so it forms a Cartesian
    # product with the latent grid -- the dense tensor or the CCD design alike
    # (gcol33/tulpa#61). On the CCD path the per-node design weights `dnode`
    # are replicated across the phi cells so each (latent node, phi cell) keeps
    # the latent node's quadrature weight; phi rides as a uniform tensor axis,
    # exactly as on the dense path. A phi axis tied to a latent-block axis is
    # not expressible here; fold it into the relevant block's own axis grid.
    if (has_phi && !use_adaptive) {
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
        # phi columns don't belong to any latent block; the C++ entry consumes
        # them via phi_grid_per_arm, not the block axes.
        phi_grid_per_arm_list <- .joint_multi_phi_per_arm(joint_grid, arm_names)
        # Replicate the CCD design weights across the crossed phi cells (NULL on
        # the dense path, where weights are the uniform-cell softmax).
        if (!is.null(dnode)) dnode <- dnode[joint_idx$joint]
    }

    # Announce the engaged outer integrator at selection time (gcol33/tulpa#63):
    # the single authoritative line, tied to integration_used, before the heavy
    # inner solves -- so a CCD engaged silently by axis count (or declined back
    # to the tensor on a ridge) is visible without reading fit$...$integration.
    if (isTRUE(verbose) && !use_adaptive) {
        n_phi_cells <- as.integer(round(nrow(joint_grid) / n_latent_cells))
        .joint_announce_integration(
            integration_used, d_axes, n_latent_cells, n_phi_cells,
            nrow(joint_grid),
            declined = ccd_requested && !identical(integration_used, "ccd"))
    }

    # Tile partition for the three-tier warm-start (Phase 2 of
    # dev_notes/speedup.md). Tensor-grid-only: the scattered CCD design is not a
    # per-axis lattice the alpha-tile reuse relies on. Tile axis = every
    # joint_grid column EXCEPT the copy block's alpha column. Built from the
    # *user-facing* (sigma, alpha) grid (before sigma_pos materialisation) so
    # the partition reflects what is constant across alpha at fixed (sigma, rho,
    # ... other-block axes).
    if (!use_ccd && isTRUE(tile_warm) && cp$has_copy &&
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
                                            axis_offsets, B, fn_sigma, fn_alpha,
                                            fn_phi)

    # Local CCD refinement (gcol33/tulpa#64): replace a few high-weight, mutually
    # non-adjacent tensor cells with small curvature-aware node clouds so a coarse
    # base grid resolves the sharply-peaked directions without the k^d tensor
    # blow-up. Engages only on the tensor path (the finite-difference curvature
    # stencil needs axis neighbours), at >= 4 transformable latent axes, with no
    # active phi grid. The node solves reuse the kernel through a quiet evaluator
    # (checkpoint / progress stripped) warm-started per cell; refined cells carry
    # partition-of-unity design weights so the total integration weight is
    # conserved (no double-count).
    local_ccd_info <- NULL
    if (!is.null(local_ccd) && !use_ccd && !use_adaptive && !has_phi &&
        .joint_local_ccd_engage(d_axes)) {
        lc <- if (is.list(local_ccd)) local_ccd else list()
        lc_max_cells <- as.integer(lc$max_cells %||% 8L)
        lc_f0        <- as.numeric(lc$f0 %||% 1.1)
        tags_lc <- .joint_ccd_axis_tags(axis_names, axis_offsets, prepared)
        if (!is.null(tags_lc)) {
            with_quiet_opts <- function(expr) {
                op <- options(
                    tulpa.nl_checkpoint = list(path = "", resume = TRUE),
                    tulpa.nl_progress   = .nl_progress_args(list(progress = FALSE)))
                on.exit(options(op), add = TRUE)
                force(expr)
            }
            eval_nodes <- function(theta_mat, warm) {
                xi <- if (!is.null(warm) && length(warm) && all(is.finite(warm)))
                    as.numeric(warm) else x_init
                r <- with_quiet_opts(.cpp_joint_multi(
                    arms_list = arms, copy_arms = as.integer(cp$copy_arms_zero),
                    copy_blocks = as.integer(cp$copy_blocks_zero),
                    blocks_spec = blocks_spec,
                    theta_grid = .joint_multi_cpp_grid(theta_mat, axis_offsets, B, cp),
                    axis_offsets = axis_offsets, max_iter = as.integer(max_iter),
                    tol = as.numeric(tol), n_threads = as.integer(n_threads),
                    x_init_nullable = xi, store_Q = FALSE, phi_grid_per_arm = NULL,
                    n_threads_outer = as.integer(n_threads_outer),
                    tile_ids = NULL, tile_pilot_cells = NULL, prune_tol = 0.0,
                    force_sparse = isTRUE(force_sparse),
                    cell_coupling_name = as.character(cell_coupling),
                    inner_refresh = as.integer(inner_refresh)))
                list(log_marginal = .joint_multi_add_hp(
                         r$log_marginal, theta_mat, axis_offsets, B,
                         fn_sigma, fn_alpha, fn_phi),
                     modes = if (is.matrix(r$modes)) r$modes else NULL)
            }
            ref <- .joint_local_ccd_refine(
                joint_grid = joint_grid, log_marginal = res$log_marginal,
                modes = res$modes, dnode = dnode, latent_axes = axis_names,
                tags = tags_lc, eval_nodes = eval_nodes,
                max_cells = lc_max_cells, f0 = lc_f0, verbose = verbose)
            if (!is.null(ref)) {
                joint_grid       <- ref$joint_grid
                res$log_marginal <- ref$log_marginal
                res$modes        <- ref$modes
                dnode            <- ref$dnode
                local_ccd_info   <- ref$info
            }
        }
        tm$mark("grid")
    }

    res$theta_grid   <- joint_grid
    res$theta_names  <- colnames(joint_grid)
    res$axis_offsets <- axis_offsets
    res$integration  <- integration_used
    # Integration weights fold in the CCD design weights (`dnode`); for the
    # tensor grid `dnode` is NULL and this is the plain log-marginal softmax.
    res$weights      <- .joint_integration_weights(res$log_marginal, dnode)
    is_ccd <- identical(integration_used, "ccd")
    # Local CCD refinement also leaves a design-weighted (scattered) grid, so it
    # takes the same moment / SD path as the global CCD: weighted moments over the
    # design, no per-axis lattice SD refit. The adaptive lattice is a mass-
    # concentrated SUBSET of the tensor (gaps in the far tail), so its per-axis
    # levels are not the regular full lattice the 3-point profile SD refit needs;
    # and it is dense near the peak, so the weighted var-of-means over the kept
    # cells is already the calibrated SD. It therefore takes the same weighted-
    # moment path, not the lattice refit.
    is_design_weighted <- is_ccd || !is.null(local_ccd_info) || use_adaptive
    res <- .joint_posterior_moments_multi(
        res, prepared, axis_offsets, joint_grid, cp,
        int_weights = if (is_design_weighted) res$weights else NULL)
    if (!is_design_weighted) {
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
    res$local_ccd_info <- local_ccd_info
    res$adaptive_grid_info <- adaptive_info
    tm$mark("postproc")
    # Outer Pareto-k-hat: re-evaluate the inner joint marginal at sampled
    # hyperparameters via the shared cpp-grid / phi / hyperprior helpers and
    # PSIS-smooth. Declines (NA -> quad-ESS) when a block carries an axis with
    # unguessable support (CAR_proper's rho_car, a non-BYM2 rho, ...).
    # The diagnostic's importance batch runs across `pareto_k_threads` (resolved
    # by the top-level caller from control$k_threads; NULL only on a direct call,
    # where it follows this fit's own thread grant). gcol33/tulpa#117.
    k_to <- pareto_k_threads %||%
        .tulpa_pareto_k_threads(n_threads_outer, n_threads, diagnose_draws, NULL)
    res <- .joint_attach_pareto_k_multi(res, arms, cp, blocks_spec,
                                        axis_offsets, B, arm_names,
                                        fn_sigma, fn_alpha, fn_phi,
                                        max_iter, tol, n_threads,
                                        force_sparse, cell_coupling,
                                        diagnose_k = diagnose_k,
                                        diagnose_draws = diagnose_draws,
                                        n_threads_outer = k_to,
                                        proposal = ccd_proposal,
                                        pareto_k_by_arm = pareto_k_by_arm,
                                        k_bootstrap = k_bootstrap,
                                        k_tail_points = k_tail_points,
                                        k_conf_bands = k_conf_bands)
    tm$mark("diagnostics")
    res$timing <- tm$timing()
    res <- .joint_attach_diagnose_cost(res, diagnose_k, diagnose_draws)
    .finalize_fit(res, backend = "nested_laplace_joint",
                  extra_class = c("tulpa_nested_laplace_joint_multi",
                    "tulpa_nested_laplace_joint",
                                  "tulpa_nested_laplace", "list"))
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
                   prepared[[b]]$n_groups        %||%
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
        } else if (type == "mcar") {
            # Separable MCAR stores p coupled fields field-major: field a at
            # offset a * n_units within the block (size p * n_units).
            block_size[b]     <- as.integer(prepared[[b]]$n_fields) * n_units
            field_starts      <- c(field_starts, cur)
            field_block_types <- c(field_block_types, "mcar")
            if (is.null(phi_start)) phi_start <- cur
        } else if (type == "miid") {
            # Multivariate IID stores p coupled per-group coefficient fields
            # field-major: field a at offset a * n_groups within the block
            # (size p * n_groups). A per-group RE, not a spatial field, so it
            # does not register a field_start / phi_start alias.
            block_size[b] <- as.integer(prepared[[b]]$n_fields) * n_units
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
