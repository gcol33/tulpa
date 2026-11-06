.hsgp_sum_indices <- function(blocks) {
  ix <- which(vapply(blocks, function(p)
    identical(tolower(p$type %||% ""), "hsgp") &&
      !identical(p$observation_space, FALSE), logical(1)))
  if (length(ix) < 2L) return(integer(0))
  rows <- vapply(blocks[ix], function(p) {
    phi <- p[["training_phi"]] %||% p[["phi"]] %||% list(p[["phi_basis"]])
    sum(vapply(phi, nrow, integer(1)))
  }, integer(1))
  width <- sum(vapply(blocks[ix], function(p)
    ncol((p[["phi"]] %||% list(p[["phi_basis"]]))[[1L]]), integer(1)))
  if (length(unique(rows)) != 1L) stop("HSGP components must have the same pooled training rows.")
  if (width <= rows[1L]) integer(0) else ix
}

.field_hsgp_inputs <- function(object, src, newdata, comp, cells) {
  sp <- comp$spec
  ct <- .coords_2col(.coords_plain(sp$coords_matrix), "hsgp() prediction")
  cn <- .coords_2col(.field_coords_at(object, newdata), "hsgp() prediction")
  bs <- .hsgp_block_basis(sp, comp$block$name)
  dom <- cpp_hsgp_domain_2d(ct, bs$c)
  outside <- abs(cn[, 1L] - dom[1L]) > dom[3L] |
    abs(cn[, 2L] - dom[2L]) > dom[4L]
  if (any(outside)) {
    stop(sum(outside), " prediction coordinate(s) lie outside the HSGP domain ",
         "(the fitted extent widened by c = ", bs$c, "). Refit with a larger ",
         "`c` in spatial_gp(approx = 'hsgp').", call. = FALSE)
  }
  basis <- cpp_hsgp_basis_2d_domain(cn, bs$m, dom)
  train <- comp$block$phi_basis %||% if (is.list(comp$block$phi))
    do.call(rbind, comp$block$phi) else
    cpp_hsgp_basis_2d_domain(ct, bs$m, dom)$phi_basis
  if (!is.null(comp$block$svc_weight)) train <- train * comp$block$svc_weight
  s2 <- .field_theta(src$theta, "sigma2", comp$prefix)
  ls <- .field_theta(src$theta, "lengthscale", comp$prefix)
  sq <- lapply(unique(cells), function(k) as.numeric(
    cpp_hsgp_sqrt_spectral(basis$lambda_eig, s2[k], ls[k],
                           .hsgp_kernel_code(bs$kernel))))
  list(training = train, prediction = basis$phi_basis, spectral = sq)
}

.field_hsgp_sum_at <- function(object, src, newdata, L, cells, draw, fields) {
  inp <- lapply(fields, function(comp)
    .field_hsgp_inputs(object, src, newdata, comp, cells))
  out <- matrix(0, nrow(L), nrow(inp[[1L]]$prediction))
  u <- unique(cells)
  for (j in seq_along(u)) {
    ix <- which(cells == u[j])
    projection <- cpp_hsgp_sum_projection(
      lapply(inp, `[[`, "training"), lapply(inp, `[[`, "prediction"),
      lapply(inp, function(p) p$spectral[[j]]))
    basis <- Reduce(`+`, lapply(projection, `[[`, "basis"))
    out[ix, ] <- L[ix, fields[[1L]]$cols[[1L]], drop = FALSE] %*% t(basis)
    if (draw) {
      residual <- Reduce(`+`, lapply(projection, `[[`, "residual_basis"))
      out[ix, ] <- out[ix, , drop = FALSE] +
        matrix(stats::rnorm(length(ix) * ncol(residual)), length(ix)) %*% t(residual)
    }
  }
  out
}
