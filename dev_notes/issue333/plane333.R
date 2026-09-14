# gcol33/tulpa#333: the descriptor-plane sweep, run in parallel.
#
# Each unit is `.dp_sweep_unit()` from
# tests/testthat/test-nested-laplace-joint-descriptor-plane.R -- one seed of the
# ogd fixture (three crossed iid blocks, sd 0.8 / 0.5 / 0.3) at one grid spread,
# a four- and a five-level base intervened on cell by cell against a
# twelve-level reference of the same data, three fits per unit.
#
#   Rscript plane333.R <lib> <repo> <hyperprior> <family> <out.rds> \
#                      [seeds=1:8] [spreads=2,3,4] [workers=8] [within_cell]
#
# `hyperprior` is the outer prior every fit states ("flat" or "proper"),
# `family` the arm ("gaussian" or "binomial"), `within_cell` the read every fit
# states (the ogd fixture's default when omitted). The output holds the units, the identity of the
# build that produced them, and a stamp column on every row.
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1L]; repo <- args[2L]; hyperprior <- args[3L]; family <- args[4L]
out <- args[5L]
seeds   <- if (length(args) >= 6L) eval(parse(text = args[6L])) else 1:8
spreads <- if (length(args) >= 7L) as.numeric(strsplit(args[7L], ",")[[1L]]) else c(2, 3, 4)
workers <- if (length(args) >= 8L) as.integer(args[8L]) else 8L
within_arg <- if (length(args) >= 9L) args[9L] else NULL
stopifnot(hyperprior %in% c("flat", "proper"), family %in% c("gaussian", "binomial"))

source(file.path(repo, "dev_notes", "harness", "load_harness.R"))
DP_FILE <- "test-nested-laplace-joint-descriptor-plane.R"
env <- harness_load(lib, repo, DP_FILE)
within_cell <- if (is.null(within_arg)) {
  as.character(formals(env$ogd_fixture_fit)$within_cell)
} else within_arg
id <- harness_identity(lib, hyperprior, within_cell)
id$family <- family; id$seeds <- seeds; id$spreads <- spreads
harness_print_identity(id)

grid <- expand.grid(seed = seeds, spread = spreads)
cl <- parallel::makePSOCKcluster(workers)
cat("worker pids:", paste(unlist(parallel::clusterEvalQ(cl, Sys.getpid())), collapse = " "), "\n")
parallel::clusterExport(cl, c("lib", "repo", "DP_FILE"))
parallel::clusterEvalQ(cl, {
  source(file.path(repo, "dev_notes", "harness", "load_harness.R"))
  assign(".dp_env", harness_load(lib, repo, DP_FILE), envir = globalenv())
  NULL
})
t0 <- Sys.time()
units <- parallel::parLapplyLB(cl, seq_len(nrow(grid)), function(i, grid, family,
                                                               hyperprior, within_cell) {
  get(".dp_env", envir = globalenv())$.dp_sweep_unit(
    grid$seed[i], grid$spread[i], family, hyperprior, within_cell)
}, grid = grid, family = family, hyperprior = hyperprior, within_cell = within_cell)
parallel::stopCluster(cl)
tag <- harness_build_tag(id)
units <- lapply(units, function(u) lapply(u, function(lv) {
  lapply(lv, function(df) {
    df$hyperprior <- hyperprior; df$within_cell <- within_cell; df$build <- tag
    df
  })
}))
saveRDS(list(identity = id, units = units, n_fits = 3L * nrow(grid)), out)
cat(sprintf("%d units (%d fits) in %.1f min -> %s\n", nrow(grid), 3L * nrow(grid),
            as.numeric(difftime(Sys.time(), t0, units = "mins")), out))
writeLines(format(Sys.time()), paste0(out, ".done"))
