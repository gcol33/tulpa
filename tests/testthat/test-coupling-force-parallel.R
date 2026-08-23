# TULPA_COUPLING_FORCE_PARALLEL takes the chunked parallel reduce on EVERY
# coupled cell instead of only where the cell count pays for the per-chunk
# partial buffers, so a small grid exercises the parallel path. The reduce runs
# in fixed chunk order, so the answer must be the one the serial pass gives --
# bit for bit, not to tolerance.
#
# gcol33/tulpa#466: the switch shipped with no test, so the reduce at
# nested_laplace_joint_multi.h was exercised only when a real grid happened to
# be large enough to take the worksteal branch.
#
# It is read at namespace scope, i.e. once when the DLL loads, so it cannot be
# flipped inside a running session: each arm is a fresh R process with the
# variable set or unset in its environment. That is also the mechanism a user
# has, so the test drives the switch the way it is actually used.

.cfp_script <- function(path) {
    writeLines(c(
        "suppressPackageStartupMessages(library(tulpa))",
        "tulpa:::cpp_register_test_occupancy_mixture_coupling()",
        "set.seed(466)",
        "n_cells <- 40L; n_visits <- 4L",
        "z <- rbinom(n_cells, 1L, plogis(0.3))",
        "y <- as.numeric(rbinom(n_cells * n_visits, 1L,",
        "                       plogis(-0.4) * rep(z, each = n_visits)))",
        "cell <- rep(seq_len(n_cells), each = n_visits)",
        "arm <- function(yy, N, map) list(",
        "    y = yy, n_trials = rep(1L, N), X = matrix(1, N, 1),",
        "    family = 'binomial', phi = 1, coupled = TRUE, cell_obs_map = map,",
        "    beta_prior_prec = 0.25)",
        "prior <- list(list(type = 'iid', n_units = 1L,",
        "                   sigma_grid = c(0.5, 0.8, 1.1, 1.4),",
        "                   obs_idx = list(rep(0L, n_cells),",
        "                                  rep(0L, n_cells * n_visits))))",
        "fit <- tulpa_nested_laplace_joint(",
        "    responses = list(occ = arm(rep(0, n_cells), n_cells,",
        "                               seq_len(n_cells)),",
        "                     det = arm(y, n_cells * n_visits, cell)),",
        "    prior = prior, cell_coupling = 'test_occupancy_mixture',",
        "    control = list(max_iter = 300L, tol = 1e-12, diagnose_k = FALSE,",
        "                   n_threads = 4L, n_threads_outer = 4L))",
        "saveRDS(list(lm = as.numeric(fit$log_marginal),",
        "             modes = as.matrix(fit$modes)),",
        "        commandArgs(trailingOnly = TRUE)[1])"
    ), path)
}

# system2()'s `env` argument is a no-op on Windows, so the switch is set on THIS
# process's environment for the duration of the call and inherited by the child.
.cfp_run <- function(script, out, force_parallel) {
    log <- paste0(out, ".log")
    vars <- list(
        R_LIBS = paste(.libPaths(), collapse = .Platform$path.sep),
        NOT_CRAN = "true",
        TULPA_COUPLING_FORCE_PARALLEL = if (force_parallel) "1" else NA
    )
    status <- withr::with_envvar(vars, suppressWarnings(system2(
        file.path(R.home("bin"), "Rscript"),
        args = c("--vanilla", shQuote(script), shQuote(out)),
        stdout = log, stderr = log)))
    list(status = status,
         log = if (file.exists(log)) {
             paste(readLines(log, warn = FALSE), collapse = "; ")
         } else "",
         value = if (file.exists(out)) readRDS(out) else NULL)
}

test_that("forcing the chunked coupling reduce reproduces the serial answer", {
    skip_on_cran()
    skip_on_os("solaris")

    dir <- withr::local_tempdir()
    script <- file.path(dir, "coupled_fit.R")
    .cfp_script(script)

    serial   <- .cfp_run(script, file.path(dir, "serial.rds"),   FALSE)
    parallel <- .cfp_run(script, file.path(dir, "parallel.rds"), TRUE)

    expect_equal(serial$status, 0L, info = serial$log)
    expect_equal(parallel$status, 0L, info = parallel$log)
    skip_if(is.null(serial$value) || is.null(parallel$value),
            "coupled subprocess fit did not produce a result")

    # The chunk partition is pinned by the grid geometry, not by how many
    # workers happen to be idle, so the reduction ORDER is the same on both
    # routes and the two answers are identical, not merely close.
    expect_identical(serial$value$lm, parallel$value$lm)
    expect_identical(serial$value$modes, parallel$value$modes)
    expect_true(all(is.finite(serial$value$lm)))
})
