# gcol33/tulpa#853 -- what the per-axis within-cell continuization does to a
# TWO-axis grid, against an exact posterior.
#
# The issue reports new family-wise SBC failures in `sigma_pos_field` (an AXIS)
# and in `alpha` (a RATIO of two axes). A ratio-of-independent-jitters story
# cannot explain an axis's own marginal, so the two have to be measured apart.
#
# Fixture (exact outer posterior, nothing about an inner Laplace under test):
#   y_j ~ N(0, s1^2)            j = 1..n1        informs s1
#   z_j ~ N(0, s1^2 + s2^2)     j = 1..n2        informs s1 + s2, correlated
#   log s1, log s2 ~ N(0, p^2)
# so the outer log-posterior is closed form and the reference posterior is a
# fine 2-D quadrature over it. `alpha = s2 / s1` is the derived quantity.

# The probe is pure R, so it runs against the INSTALLED compiled package with
# the working tree's own `R/` sourced over it -- no rebuild, and the R half is
# the tree's.
suppressMessages(library(tulpa))
local({
  ns <- asNamespace("tulpa")
  tmp <- new.env(parent = ns)
  for (f in list.files("R", pattern = "[.][Rr]$", full.names = TRUE)) {
    try(sys.source(f, envir = tmp, keep.source = FALSE), silent = TRUE)
  }
  for (nm in ls(tmp, all.names = TRUE)) {
    v <- get(nm, envir = tmp)
    if (is.function(v)) environment(v) <- ns
    if (exists(nm, envir = ns, inherits = FALSE) &&
        bindingIsLocked(nm, ns)) unlockBinding(nm, ns)
    assign(nm, v, envir = ns)
  }
  cat("sourced", length(ls(tmp, all.names = TRUE)), "objects over the install\n")
})

lpost2 <- function(l1, l2, Sy, Sz, n1, n2, p) {
  v1 <- exp(2 * l1)
  vz <- v1 + exp(2 * l2)
  -n1 * l1 - Sy / (2 * v1) -
    (n2 / 2) * log(vz) - Sz / (2 * vz) -
    0.5 * (l1 / p)^2 - 0.5 * (l2 / p)^2
}

# Reference posterior: normalized weights on a fine tensor quadrature.
exact_ref <- function(Sy, Sz, n1, n2, p, m = 401L, span = 8) {
  g <- seq(-span, span, length.out = m)
  G <- expand.grid(l1 = g, l2 = g)
  lp <- lpost2(G$l1, G$l2, Sy, Sz, n1, n2, p)
  w <- exp(lp - max(lp)); w <- w / sum(w)
  list(s1 = exp(G$l1), s2 = exp(G$l2), w = w)
}
ref_cdf <- function(ref, q, t) sum(ref$w[q <= t])
ref_sd  <- function(ref, q) {
  m <- sum(ref$w * q); sqrt(max(sum(ref$w * (q - m)^2), 0))
}

# The engine-shaped tensor grid: mode plus per-axis marginal sd from the
# Hessian diagonal, n nodes over +/- 2.5 sd, in the log coordinate.
# `fixed`: the same nodes for every replicate, sized on the PRIOR -- which is
# what an SBC arm with a user-supplied hyperparameter grid runs, and the regime
# an adapted grid never enters: a replicate whose data pin the hyperparameter
# gets a posterior narrower than one cell.
grid_fit2 <- function(Sy, Sz, n1, n2, p, n_nodes, fixed = FALSE) {
  if (fixed) {
    ax <- rep(list(exp(p * seq(-2.5, 2.5, length.out = n_nodes))), 2)
  } else {
    f <- function(v) -lpost2(v[1], v[2], Sy, Sz, n1, n2, p)
    o <- stats::optim(c(0, 0), f, method = "BFGS", hessian = TRUE)
    sdj <- sqrt(pmax(diag(solve(o$hessian)), 1e-10))
    ax <- lapply(1:2, function(j)
      exp(o$par[j] + sdj[j] * seq(-2.5, 2.5, length.out = n_nodes)))
  }
  tg <- as.matrix(expand.grid(sigma = ax[[1]], sigma_pos = ax[[2]]))
  lm <- lpost2(log(tg[, 1]), log(tg[, 2]), Sy, Sz, n1, n2, p)
  specs <- tulpa:::.joint_axis_specs_from_grid(tg)
  lq <- tulpa:::.hyper_log_quad_weights(tg, specs)
  res <- list(theta_grid = tg, log_marginal = lm, log_quad = lq,
              integration = "grid", prior = list(type = "bym2"),
              blocks = NULL, axis_offsets = NULL)
  res$weights <- tulpa:::.nl_normalise_weights_safe(lm, "outer grid",
                                                    log_quad = lq)
  res <- tulpa:::.nl_posterior_moments(res, "bym2", within = "box_uniform")
  res$within_cell_requested <- "box_uniform"
  structure(res, class = c("tulpa_nested_laplace", "list", "tulpa_fit"))
}

# Two COUPLED alternatives to the shipped per-axis independent jitter, each
# keeping every axis's own within-cell marginal exactly (the uniform's inverse
# CDF is affine, so any uniform input reproduces the box) and differing only in
# how the axes' uniforms are tied:
#   `boxc` -- one uniform shared by every axis (comonotone, the extreme).
#   `boxr` -- a Gaussian copula at the grid's OWN weighted correlation of the
#             log coordinates, which is the local correlation the fit knows.
coupled_box <- function(fit, cells, rho = NULL) {
  tg <- fit$theta_grid
  d <- ncol(tg)
  w <- fit$weights
  doms <- tryCatch(tulpa:::.nl_axis_domains(fit), error = function(e) NULL)
  geo <- lapply(seq_len(d), function(j)
    tulpa:::.nl_hyper_axis_geometry(as.numeric(tg[, j]), w,
                                    if (length(doms) < j) NA_character_ else doms[[j]],
                                    "box_uniform", "extend"))
  n <- length(cells)
  U <- if (is.null(rho)) {
    matrix(stats::runif(n), n, d)                       # comonotone
  } else {
    R <- matrix(rho, d, d); diag(R) <- 1
    Z <- matrix(stats::rnorm(n * d), n, d) %*% chol(R)
    stats::pnorm(Z)
  }
  out <- matrix(NA_real_, n, d, dimnames = list(NULL, colnames(tg)))
  for (j in seq_len(d)) {
    g <- geo[[j]]
    k <- match(as.numeric(tg[cells, j]), g$values)
    out[, j] <- g$lo[k] + U[, j] * (g$hi[k] - g$lo[k])
  }
  out
}

# The grid's own weighted correlation between the two log axes.
grid_rho <- function(fit) {
  tg <- fit$theta_grid; w <- fit$weights; w <- w / sum(w)
  L <- log(tg)
  m <- colSums(w * L)
  C <- crossprod(sqrt(w) * sweep(L, 2L, m))
  s <- sqrt(diag(C))
  if (any(!is.finite(s)) || any(s <= 0)) return(0)
  max(min(C[1, 2] / (s[1] * s[2]), 0.999), -0.999)
}

run_arm <- function(n1, n2, p, n_nodes, n_sim, n_draw = 4000L, seed = 853L,
                    fixed = FALSE) {
  set.seed(seed)
  t1 <- exp(stats::rnorm(n_sim, 0, p))
  t2 <- exp(stats::rnorm(n_sim, 0, p))
  qn <- c("s1", "s2", "alpha")
  arms <- c("atom", "box", "boxc", "boxr", "exact")
  pit <- lapply(arms, function(.)
    matrix(NA_real_, n_sim, 3, dimnames = list(NULL, qn)))
  names(pit) <- arms
  rhos <- numeric(n_sim)
  # sd of the exact posterior against the width of the cell the draw sits in,
  # per axis -- the quantity the "smear is wider than the posterior" reading
  # turns on.
  ratio <- matrix(NA_real_, n_sim, 2)

  for (s in seq_len(n_sim)) {
    y <- stats::rnorm(n1, 0, t1[s])
    z <- stats::rnorm(n2, 0, sqrt(t1[s]^2 + t2[s]^2))
    Sy <- sum(y^2); Sz <- sum(z^2)
    fit <- grid_fit2(Sy, Sz, n1, n2, p, n_nodes, fixed = fixed)
    tg <- fit$theta_grid
    cells <- tulpa:::.nl_mixture_cells(fit$weights, seq_along(fit$weights),
                                       n_draw)$row_cells
    A <- tg[cells, , drop = FALSE]
    B <- tulpa_hyper_draws(fit, cells = cells)
    ref <- exact_ref(Sy, Sz, n1, n2, p)

    rhos[s] <- grid_rho(fit)
    Cm <- coupled_box(fit, cells, rho = NULL)
    Rp <- coupled_box(fit, cells, rho = rhos[s])

    tv <- c(s1 = t1[s], s2 = t2[s], alpha = t2[s] / t1[s])
    pit_of <- function(M) c(mean(M[, 1] <= tv[1]), mean(M[, 2] <= tv[2]),
                            mean(M[, 2] / M[, 1] <= tv[3]))
    pit$atom[s, ] <- pit_of(A)
    pit$box[s, ]  <- pit_of(B)
    pit$boxc[s, ] <- pit_of(Cm)
    pit$boxr[s, ] <- pit_of(Rp)
    pit$exact[s, ] <- c(ref_cdf(ref, ref$s1, tv[1]),
                        ref_cdf(ref, ref$s2, tv[2]),
                        ref_cdf(ref, ref$s2 / ref$s1, tv[3]))

    for (j in 1:2) {
      uv <- sort(unique(tg[, j]))
      w <- diff(range(uv)) / (length(uv) - 1)
      ratio[s, j] <- ref_sd(ref, if (j == 1) ref$s1 else ref$s2) / w
    }
  }
  list(pit = pit, ratio = ratio, rho = rhos)
}

ks <- function(v) suppressWarnings(stats::ks.test(v, "punif")$statistic)
kp <- function(v) suppressWarnings(stats::ks.test(v, "punif")$p.value)

report <- function(tag, r) {
  cat("\n== ", tag, "  (posterior sd / cell width: s1 ",
      sprintf("%.2f", median(r$ratio[, 1])), ", s2 ",
      sprintf("%.2f", median(r$ratio[, 2])), "; grid corr ",
      sprintf("%.2f", median(r$rho)), ")\n", sep = "")
  cat(sprintf("%-8s %-8s %8s %8s %8s %8s\n",
              "quantity", "arm", "KS", "p", "pinned", "sd"))
  for (q in c("s1", "s2", "alpha")) {
    for (a in c("atom", "box", "boxc", "boxr", "exact")) {
      v <- r$pit[[a]][, q]
      cat(sprintf("%-8s %-8s %8.4f %8.4f %8.3f %8.3f\n", q, a, ks(v), kp(v),
                  mean(v <= 1e-3 | v >= 1 - 1e-3), stats::sd(v)))
    }
  }
}

cfgs <- list(
  list(tag = "adapt weak  n1=8   nodes=5", n1 = 8L,   n2 = 8L,   nodes = 5L,
       fixed = FALSE),
  list(tag = "adapt sharp n1=400 nodes=5", n1 = 400L, n2 = 400L, nodes = 5L,
       fixed = FALSE),
  list(tag = "adapt sharp n1=400 nodes=9", n1 = 400L, n2 = 400L, nodes = 9L,
       fixed = FALSE),
  list(tag = "fixed weak  n1=8   nodes=5", n1 = 8L,   n2 = 8L,   nodes = 5L,
       fixed = TRUE),
  list(tag = "fixed sharp n1=400 nodes=5", n1 = 400L, n2 = 400L, nodes = 5L,
       fixed = TRUE),
  list(tag = "fixed sharp n1=400 nodes=9", n1 = 400L, n2 = 400L, nodes = 9L,
       fixed = TRUE)
)

n_sim <- as.integer(Sys.getenv("PROBE_NSIM", "300"))
for (cf in cfgs) {
  r <- run_arm(cf$n1, cf$n2, p = 0.5, n_nodes = cf$nodes, n_sim = n_sim,
               fixed = cf$fixed)
  report(cf$tag, r)
  saveRDS(r, file.path("dev_notes/issue853",
                       paste0("probe_", gsub("[^0-9a-z]+", "_",
                                             tolower(cf$tag)), ".rds")))
}
