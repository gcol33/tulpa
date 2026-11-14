device_likelihood_benchmark <- function(append_row, weights = FALSE,
                                        detail_file = "likelihood-details.csv", pending = NULL) {
  source("tests/testthat/helper-device-likelihood.R", local = TRUE)
  families <- c("gaussian", "binomial", "poisson", "neg_binomial_2", "gamma", "beta",
                "beta_binomial", "truncated_poisson", "truncated_neg_binomial_2", "neg_binomial_1")
  discrete <- setdiff(families, c("gaussian", "gamma", "beta"))
  ladder <- as.integer(c(1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7))
  selected <- Sys.getenv("TULPA_BENCH_FAMILIES", "")
  if (nzchar(selected)) families <- intersect(families, strsplit(selected, ",", fixed = TRUE)[[1L]])
  cap <- as.double(Sys.getenv("TULPA_BENCH_MAX_N", "1e7"))
  ladder <- ladder[ladder <= cap]
  operation <- if (weights) "likelihood_weights" else "likelihood"
  details <- list()
  add_detail <- function(row) {
    details[[length(details) + 1L]] <<- row
    utils::write.csv(do.call(rbind, details), detail_file, row.names = FALSE)
  }
  for (family in families) for (zi in c(FALSE, if (family %in% discrete) TRUE)) {
    family_id <- match(family, c("gaussian", "binomial", "poisson", "neg_binomial_2", "gamma", "beta",
                                "beta_binomial", "truncated_poisson", "truncated_neg_binomial_2", "neg_binomial_1")) - 1L
    variant <- 2L * family_id + as.integer(zi)
    small <- fused_fixture(family, n = 127L, zi = zi)
    probe <- fused_run(small, "arena", return_weights = FALSE)
    nodes_per_row <- probe$arena_nodes / 127
    for (n in ladder) {
      if (!is.null(pending) && !any(pending$family == family & pending$zi == zi & pending$n == n)) next
      gc(FALSE)
      resource <- tulpa:::cpp_device_likelihood_resources()
      np <- 1L + zi
      host_bound <- n * (160 + 100 * np)
      device_bound <- n * (80 + 32 * np + if (weights) 8 * np * (np + 1) else 0)
      reason <- ""
      if (host_bound > resource$host_available * .5) reason <- "observation buffers exceed half the currently available host RAM"
      if (device_bound > resource$device_available * .8) reason <- "observation buffers exceed 80% of currently available device RAM"
      row <- data.frame(operation = operation, family = family, zi = zi, variant = variant, n = n,
        h_seconds = NA_real_, arena_seconds = NA_real_, cuda_seconds = NA_real_,
        batch4_seconds = NA_real_, streams4_seconds = NA_real_, cuda_setup_seconds = NA_real_,
        value_scale = NA_real_, value_error = NA_real_, gradient_scale = NA_real_, gradient_error = NA_real_,
        host_available = resource$host_available, device_available = resource$device_available,
        arena_projected_peak = NA_real_, arena_nodes = NA_real_, arena_capacity = NA_real_,
        reason = reason, arena_reason = "", chains_reason = "")
      if (nzchar(reason)) { add_detail(row); next }
      f <- fused_fixture(family, n = n, zi = zi)
      stopifnot(length(f$y) == n, nrow(f$X) == n, length(f$groups) == n,
                ncol(f$coefficients) == 1L, max(f$groups) == 5L)
      cat(sprintf("loaded %d observations: %s zi=%s weights=%s\n", n, family, zi, weights))
      cpu <- fused_run(f, "cpu", return_weights = weights)
      cuda <- fused_run(f, "cuda", return_weights = weights)
      stopifnot(cuda$used_cuda, !cpu$used_cuda)
      for (q in c("value", "gradient")) {
        scale <- max(1, abs(cpu[[q]]))
        gap <- max(abs(cpu[[q]] - cuda[[q]]))
        cat(sprintf("%s scale=%.17g error=%.17g tolerance=%.3g\n", q, scale, gap, 2e-8 * scale))
        stopifnot(is.finite(gap), gap <= 2e-8 * scale)
        row[[paste0(q, "_scale")]] <- scale
        row[[paste0(q, "_error")]] <- gap
      }
      repetitions <- vapply(c(cpu$seconds, cuda$seconds), function(seconds)
        max(1L, min(20L, as.integer(ceiling(.05 / seconds)))), 1L)
      names(repetitions) <- c("cpu", "cuda")
      samples <- matrix(NA_real_, 3L, 2L, dimnames = list(NULL, c("cpu", "cuda")))
      for (trial in 1:3) {
        order <- if (trial %% 2L) c("cpu", "cuda") else c("cuda", "cpu")
        for (choice in order) samples[trial, choice] <-
          fused_run(f, choice, repetitions = repetitions[[choice]], return_weights = weights)$seconds
        append_row(operation, variant, n, samples[trial, "cpu"], samples[trial, "cuda"])
      }
      row$h_seconds <- stats::median(samples[, "cpu"])
      row$cuda_seconds <- stats::median(samples[, "cuda"])
      row$cuda_setup_seconds <- cuda$setup_seconds
      required_nodes <- ceiling(nodes_per_row * n)
      capacity <- max(131072, 2^ceiling(log2(required_nodes)))
      row$arena_projected_peak <- capacity * 40 * 1.5 + host_bound
      if (!weights && row$arena_projected_peak <= tulpa:::cpp_device_likelihood_resources()$host_available * .5) {
        arena <- fused_run(f, "arena", return_weights = FALSE)
        row$arena_seconds <- arena$seconds
        row$arena_nodes <- arena$arena_nodes
        row$arena_capacity <- arena$arena_capacity
        stopifnot(max(abs(arena$gradient - cuda$gradient)) <= 2e-8 * max(1, abs(arena$gradient)))
        rm(arena)
      } else row$arena_reason <- if (weights) "arena comparison belongs to the gradient operation" else "measured arena nodes predict a peak above half the currently available host RAM"
      if (!weights && device_bound + 3 * n * np * 8 <= resource$device_available * .8) {
        chains <- f
        chains$coefficients <- f$coefficients[, rep(1L, 4L), drop = FALSE]
        stopifnot(ncol(chains$coefficients) == 4L)
        batch <- fused_run(chains, "cuda", batched = TRUE, repetitions = repetitions[["cuda"]], return_weights = FALSE)
        streams <- fused_run(chains, "cuda", batched = FALSE, repetitions = repetitions[["cuda"]], return_weights = FALSE)
        stopifnot(max(abs(batch$gradient - streams$gradient)) <= 1e-12 * max(1, abs(batch$gradient)))
        row$batch4_seconds <- batch$seconds
        row$streams4_seconds <- streams$seconds
        rm(chains, batch, streams)
      } else row$chains_reason <- if (weights) "chain scheduling comparison belongs to the gradient operation" else "four chain score buffers exceed the available device budget"
      add_detail(row)
      rm(f, cpu, cuda)
      gc(FALSE)
    }
  }
  invisible(do.call(rbind, details))
}
