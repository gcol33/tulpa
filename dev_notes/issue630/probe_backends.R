# gcol33/tulpa#630: the realized change on real fits, backend by backend.
#
# `probe_recov.R` measures `tulpa_re_cov_nested()`. This one measures the other
# two backends #630 wired -- `tulpa_nested_laplace()` (a grid-moment proposal
# WITH integration nodes, so all four candidates are available) and `fit_spde()`
# in both its methods -- by reading each fit's `pareto_k_first_pass` (the
# proposal exactly as the backend placed it, which is what these paths reported
# before #630) against its `pareto_k` (the best candidate).

suppressMessages(library(tulpa))
`%||%` <- function(a, b) if (is.null(a)) b else a

rook <- function(nr, nc) {
    n <- nr * nc; W <- matrix(0, n, n); id <- function(r, c) (c - 1) * nr + r
    for (r in seq_len(nr)) for (c in seq_len(nc)) {
        if (r < nr) { W[id(r, c), id(r + 1, c)] <- 1; W[id(r + 1, c), id(r, c)] <- 1 }
        if (c < nc) { W[id(r, c), id(r, c + 1)] <- 1; W[id(r, c + 1), id(r, c)] <- 1 }
    }
    W
}

# --------------------------------------------------------------------------- #
# tulpa_nested_laplace(): single positive-scale axis, ICAR block               #
# --------------------------------------------------------------------------- #
nl_fit <- function(seed, reps = 4L, nr = 5L, nc = 5L) {
    set.seed(seed)
    S <- nr * nc; W <- rook(nr, nc)
    unit <- rep(seq_len(S), each = reps); N <- length(unit)
    x <- rnorm(N); ntr <- rep(3L, N)
    y <- rbinom(N, ntr, plogis(-0.3 + 0.6 * x))
    idx <- tulpa:::.resolve_unit_index(factor(unit), "region", S)
    csr <- tulpa:::adjacency_to_csr_tulpa(W)
    prior <- list(type = "icar", spatial_idx = idx, n_spatial_units = S,
                  adj_row_ptr = csr$row_ptr, adj_col_idx = csr$col_idx,
                  n_neighbors = csr$n_neighbors)
    tulpa_nested_laplace(y = y, n_trials = ntr, X = cbind(1, x), prior = prior,
                         family = "binomial")
}

rows <- list()
for (reps in c(2L, 4L, 10L)) for (sd_i in 1:5) {
    f <- nl_fit(100L * reps + sd_i, reps = reps)
    rows[[length(rows) + 1L]] <- data.frame(
        backend = "tulpa_nested_laplace", config = sprintf("icar_5x5_reps%d", reps),
        seed = sd_i,
        first_pass = f$pareto_k_first_pass %||% NA_real_,
        reported   = f$pareto_k %||% NA_real_,
        src        = f$pareto_k_proposal_source %||% NA_character_,
        stringsAsFactors = FALSE)
}

# --------------------------------------------------------------------------- #
# fit_spde(): both integration methods on one field                            #
# --------------------------------------------------------------------------- #
spde_ok <- requireNamespace("tulpaMesh", quietly = TRUE)
if (spde_ok) for (meth in c("grid", "ccd")) for (sd_i in 1:3) {
    f <- tryCatch({
        set.seed(400L + sd_i)
        n <- 150L
        co <- cbind(runif(n), runif(n))
        y  <- rbinom(n, 1L, plogis(-0.2 + 1.0 * co[, 1]))
        fit_spde(y, matrix(1, n, 1L), spatial_spde(co), family = "binomial",
                 n_trials = rep(1L, n),
                 control = list(method = meth, n_grid = 5L, k_samples = 300L))
    }, error = function(e) NULL)
    if (is.null(f)) next
    rows[[length(rows) + 1L]] <- data.frame(
        backend = "fit_spde", config = paste0("method=", meth), seed = sd_i,
        first_pass = f$pareto_k_first_pass %||% NA_real_,
        reported   = f$pareto_k %||% NA_real_,
        src        = f$pareto_k_proposal_source %||% NA_character_,
        stringsAsFactors = FALSE)
}

d <- do.call(rbind, rows)
utils::write.csv(d, file.path("dev_notes", "issue630", "backends630.csv"),
                 row.names = FALSE)
print(d, digits = 4, row.names = FALSE)

band <- function(k) cut(k, c(-Inf, 0.5, 0.7, Inf),
                        labels = c("good", "ok", "unreliable"))
cat("\n== band, as placed vs as reported ==\n")
print(table(as_placed = band(d$first_pass), as_reported = band(d$reported)))
cat("\n== medians by config ==\n")
print(aggregate(cbind(first_pass, reported) ~ backend + config, d,
                function(x) round(median(x, na.rm = TRUE), 3),
                na.action = na.pass))
cat("\nadopted source:\n"); print(table(d$config, d$src))
