# Simulation-based calibration of the tulpaObs nested-Laplace engine on the
# coupled occu_cover hurdle (Talts et al. 2018; Sailynoja et al. 2026, Alg 2).
#
# This is the citable method validation for the cover-glaser deliverable: draw a
# truth from the fit's own posterior, simulate a replicate on FRESH cells, refit
# the pooled data, and rank the truth in the augmented posterior. Under correct
# inference the ranks are uniform. The wide/narrow control arms report the same
# draws deliberately mis-scaled and must leave the band -- a calibration read
# nothing can fail is not evidence.

suppressMessages(library(tulpaObs))

OUT <- Sys.getenv("SBC_OUT",
  "C:/Users/GillesC/Documents/TaskGoblin/colabs/cover-glaser/validation/results/sbc")
if (!dir.exists(OUT)) dir.create(OUT, recursive = TRUE)

N_SIM   <- as.integer(Sys.getenv("SBC_N_SIM", "300"))
SEED    <- as.integer(Sys.getenv("SBC_SEED", "20260820"))
GRID    <- 8L                       # 8 x 8 rook-adjacency areal grid, N = 64
J       <- as.integer(Sys.getenv("SBC_J", "3"))

# Each of the 11 reported quantities gets its own simultaneous ECDF band, so
# "all 11 inside" at a per-quantity 0.95 is not a 0.95 statement about the fit:
# under exact calibration it holds with probability 0.95^11 = 0.57. LEVEL_FW is
# the Sidak level at which the JOINT read is 0.95. It is a function of the number
# of quantities alone, fixed before the run, and both reads are written.
LEVEL <- 0.95
# LEVEL_FW is derived from the number of quantities the run ACTUALLY scored, at
# the point that number is known, rather than from a literal 11. The count is
# not a constant of the fixture: the deliverable-placement arm at J = 10 scored
# 10, omitting `sigma`, on the same code path that scores 11 at J = 3. A
# hardcoded 11 then computes the Sidak level for a family the run does not have,
# which widens the band and reads permissive on exactly the arm that degraded
# most. It is also why a raw count is only comparable across arms once the
# denominator is checked.

# Engine identity, stamped rather than assumed: a number nobody can trace to a
# build is not a result. tulpa 0.3.0 / tulpaObs 0.2.0 retire copy() for share().
engine_stamp <- function() {
  # On a machine that carries the source trees, read the commit from them. On one
  # that was handed a built engine (LiSC), the shipping side states the commits
  # in SBC_ENGINE_SHA -- the identity has to travel WITH the data rather than be
  # re-derived at the far end, where the repo does not exist.
  env <- Sys.getenv("SBC_ENGINE_SHA", "")
  if (nzchar(env))
    return(sprintf("tulpa %s | tulpaObs %s | commits %s",
                   packageVersion("tulpa"), packageVersion("tulpaObs"), env))
  sha <- function(p) {
    r <- suppressWarnings(try(system2("git", c("-C", p, "rev-parse", "--short", "HEAD"),
                                      stdout = TRUE, stderr = FALSE), silent = TRUE))
    if (inherits(r, "try-error") || !length(r)) NA_character_ else r[[1L]]
  }
  d <- "C:/Users/GillesC/Documents/dev"
  sprintf("tulpa %s (%s) | tulpaObs %s (%s)",
          packageVersion("tulpa"), sha(file.path(d, "tulpa")),
          packageVersion("tulpaObs"), sha(file.path(d, "tulpaObs")))
}

# WHICH COVER LIKELIHOOD IS BEING VALIDATED. The engine offers a beta and a
# lognormal positive arm, and they are different likelihoods rather than two
# parameterizations of one: beta puts cover on (0, 1) with a precision, lognormal
# puts log-cover on the real line with a residual SD. A calibration result on one
# is not a calibration result on the other.
#
# The script Michael runs fits the BETA arm (BioDiv_State_GC_shared25.R:768,
# `occu_cover(response = "beta")`), so a lognormal fixture scores a likelihood the
# deliverable does not use.
#
# The generating dispersion is left at the simulator's own default in both cases
# -- `sigma_pos = 0.4` for lognormal, `phi = 30` for beta -- so the lognormal path
# through this driver is unchanged and the arms already run against it stay
# comparable.
FAMILY <- Sys.getenv("SBC_FAMILY", "lognormal")
if (!FAMILY %in% c("lognormal", "beta"))
  stop("SBC_FAMILY must be 'lognormal' or 'beta', got ", sQuote(FAMILY))

# WHICH POSTERIOR REPRESENTATION IS BEING VALIDATED, and it is the reason this
# switch exists. On `nested_laplace` the hyperparameters are not drawn from a
# continuous posterior: the joint substrate reads `sigma`, `alpha` and `disp`
# off `theta_grid[cells, ]`, so their whole posterior is the outer grid's node
# values. Measured on the arms on disk, `sigma` carries 6.6 to 9.1 effective
# atoms and `alpha` only 2.4 to 3.9 (validation/scripts/atom_count.R), and every
# SBC failure to date sits on those quantities while no continuous one has ever
# failed.
#
# `nuts` samples the same model with no grid anywhere, so it scores the same 11
# quantities on a continuous posterior. It is the comparator that separates "the
# outer layer's representation" from "the fit is wrong": if `sigma` clears on
# the sampled route and not on the grid one, the representation owns it.
#
# Both routes go through the SAME registry entry
# (tulpaObs `.tobs_sbc_draws_occu_cover`) and report on the same scales --
# `sigma` is the amplitude against the unscaled intrinsic precision either way,
# `disp` is cover()'s own surface either way -- so the two are comparable.
METHOD <- Sys.getenv("SBC_METHOD", "nested_laplace")
if (!METHOD %in% c("nested_laplace", "nuts"))
  stop("SBC_METHOD must be 'nested_laplace' or 'nuts', got ", sQuote(METHOD))
IS_NUTS <- identical(METHOD, "nuts")

# Sampler budget, used only on the nuts route. Stated rather than defaulted: the
# number of draws behind a rank is part of what a rank means.
N_ITER   <- as.integer(Sys.getenv("SBC_NUTS_ITER", "2000"))
N_WARMUP <- as.integer(Sys.getenv("SBC_NUTS_WARMUP", "1000"))
N_CHAINS <- as.integer(Sys.getenv("SBC_NUTS_CHAINS", "1"))

cat(sprintf("%s | n.sim=%d | seed=%d | N=%d J=%d | cover %s | method %s\n",
            engine_stamp(), N_SIM, SEED, GRID * GRID, J, FAMILY, METHOD))

# --- Areal fixture: 8 x 8 rook grid, coupled occupancy + cover hurdle --------
rook_adj <- function(g) {
  N <- g * g
  A <- matrix(0L, N, N)
  idx <- function(r, c) (r - 1L) * g + c
  for (r in seq_len(g)) for (c in seq_len(g)) {
    s <- idx(r, c)
    if (r > 1L) A[s, idx(r - 1L, c)] <- 1L
    if (r < g) A[s, idx(r + 1L, c)] <- 1L
    if (c > 1L) A[s, idx(r, c - 1L)] <- 1L
    if (c < g) A[s, idx(r, c + 1L)] <- 1L
  }
  A
}

adj <- rook_adj(GRID)
N   <- nrow(adj)

sim <- simulate_occu_cover(N = N, J = J, positive = FAMILY,
                           adj = adj, sigma = 0.7, alpha = 1, seed = 1L)
long <- data.frame(site_id = rep(seq_len(N), each = J),
                   visit = rep(seq_len(J), times = N),
                   y = as.vector(t(sim$y)),
                   det_cov1 = sim$visit_data$det_cov1,
                   pos_cov1 = sim$visit_data$pos_cov1)
od <- tobs_data(long, y = "y", site = "site_id", visit = "visit",
                det.covs = c("det_cov1", "pos_cov1"))
y_pos <- sim$y_pos; y_pos[is.na(y_pos)] <- 0

# Two separate decisions, kept separate. The SPAN of each axis is a declared
# prior -- since 0.2.1 the outer grid is a quadrature rule for a prior flat in
# log over its span -- so it is set on prior grounds and not moved to fit the
# data. [0.1, 3] is the engine's own declared field-SD span. The NODE COUNT is a
# quadrature accuracy choice with no prior content, so it is set high enough to
# integrate that prior rather than left at the engine's default 5. The count is
# read from a declared grid-ESS target on the base fit (calib_grid_ess.R), never
# from any rank the run returns.
K_SIGMA <- as.integer(Sys.getenv("SBC_NODES_SIGMA", "13"))
K_PHI   <- as.integer(Sys.getenv("SBC_NODES_PHI", "11"))

# WHICH QUADRATURE IS BEING VALIDATED. `pinned` states the nodes, which is what
# every run so far has done. `auto` omits them and lets the engine place its own
# on the mode Hessian.
#
# This is not a detail. An explicit numeric grid DISABLES the engine's
# auto-recenter, and the script Michael runs defaults SIGMA_GRID to "auto"
# (BioDiv_State_GC_shared25.R:189). So a pinned fixture validates a quadrature
# scheme the deliverable does not use, which is the one thing a validation of
# the quadrature must not do.
#
# It also bounds what a pinned grid can reach. At J = 30 the per-axis peak mass
# barely moves with node count -- sigma 0.402 -> 0.352 and phi 0.688 -> 0.589
# going from 21 to 41 nodes -- because the posterior is narrower than the node
# spacing over the declared span, so extra nodes land in the tails. Placement,
# not count, is the lever there, and placement is exactly what `auto` supplies.
#
# The span stays the declared prior either way. Recentering moves where the
# nodes SIT, the same category of change as asking the alpha axis for more
# resolution, and does not restate the prior.
GRID_MODE <- Sys.getenv("SBC_GRID_MODE", "pinned")
if (!GRID_MODE %in% c("pinned", "auto"))
  stop("SBC_GRID_MODE must be 'pinned' or 'auto', got ", sQuote(GRID_MODE))

# The two outer axes are placed INDEPENDENTLY, because the deliverable places
# them independently: BioDiv_State_GC_shared25.R leaves SIGMA_GRID at "auto"
# (line 189) and pins PHI_GRID_POS to four nodes over [1, 60] (line 190). One
# switch covering both axes cannot express that combination, and "the grid the
# deliverable actually builds" is the configuration this step most needs to be
# able to ask about. SBC_GRID_MODE stays the shorthand that sets both, so every
# arm already on record reproduces from an unchanged environment.
SIGMA_PLACEMENT <- Sys.getenv("SBC_SIGMA_PLACEMENT", GRID_MODE)
PHI_PLACEMENT   <- Sys.getenv("SBC_PHI_PLACEMENT",   GRID_MODE)
if (!SIGMA_PLACEMENT %in% c("pinned", "auto"))
  stop("SBC_SIGMA_PLACEMENT must be 'pinned' or 'auto', got ", sQuote(SIGMA_PLACEMENT))
if (!PHI_PLACEMENT %in% c("pinned", "auto"))
  stop("SBC_PHI_PLACEMENT must be 'pinned' or 'auto', got ", sQuote(PHI_PLACEMENT))

ctl <- if (IS_NUTS) {
  list(n.iter = N_ITER, n.warmup = N_WARMUP, n.chains = N_CHAINS,
       verbose = FALSE)
} else {
  list(engine = "joint", verbose = FALSE)
}
if (!IS_NUTS && identical(SIGMA_PLACEMENT, "pinned")) {
  ctl$sigma.grid <- exp(seq(log(0.10), log(3.00), length.out = K_SIGMA))
}
if (!IS_NUTS && identical(PHI_PLACEMENT, "pinned")) {
  # The positive-arm dispersion axis spans a different parameter in each family,
  # so its span is family-specific while the node count is not. [0.20, 0.90] is
  # the declared lognormal residual-SD span; [1, 60] is the beta precision span
  # the deliverable declares (BioDiv_State_GC_shared25.R:190), and the generating
  # precision of 30 sits inside it. The deliverable reads that span on 4 nodes;
  # the count is a quadrature accuracy choice with no prior content, per the note
  # above, so it is set here from the accuracy target rather than copied across.
  ctl$phi.grid.pos <- if (identical(FAMILY, "beta")) {
    exp(seq(log(1.00), log(60.00), length.out = K_PHI))
  } else {
    exp(seq(log(0.20), log(0.90), length.out = K_PHI))
  }
}

# Printed where it is decided, and printed always: how the hyperparameters were
# represented is part of a run's identity, and a pinned grid, an engine-placed
# one and a sampled posterior are not the same validation.
#
# Under `nuts` the placement settings above are not applied -- `ctl` carries the
# sampler budget and neither grid is attached -- so naming them here would report
# a quadrature the run never built. What identifies a sampled run is the number
# of draws behind each rank.
if (IS_NUTS) {
  cat(sprintf("hyperparameters: sampled, no outer grid | %d iter, %d warmup, %d chain(s)\n",
              N_ITER, N_WARMUP, N_CHAINS))
} else {
  cat(sprintf("grid mode: sigma %s%s | phi %s%s\n",
              SIGMA_PLACEMENT,
              if (identical(SIGMA_PLACEMENT, "pinned")) sprintf(" (%d nodes)", K_SIGMA) else " (engine places)",
              PHI_PLACEMENT,
              if (identical(PHI_PLACEMENT, "pinned")) sprintf(" (%d nodes)", K_PHI) else " (engine places)"))
}

# Outer-grid OpenMP width. The default is 1L, so the 1433-to-5075 cell grid is
# solved SERIALLY unless this is asked for, and 300 sequential simulations of a
# serial grid is the whole runtime. It is part of the run's identity rather than
# a free speedup: tulpa is bit-for-bit reproducible at a GIVEN width and fits at
# different widths agree only to floating-point tolerance, so it is stamped with
# the result. The engine also derives its own cap from omp_get_max_threads(), so
# a job script must export OMP_NUM_THREADS or the width silently clamps to 1
# (gcol33/tulpa#651); `n_threads_outer_realised` is what actually ran.
#
# The name is `n.threads.outer`. tobs() takes dot-separated controls and
# translates them to tulpa's underscore ones at the boundary
# (occu_cover_joint.R:746), and its validator REJECTS the engine-side spelling
# rather than passing it through, so the two are not interchangeable here.
THREADS_OUTER <- as.integer(Sys.getenv("SBC_THREADS_OUTER", "1"))
if (THREADS_OUTER > 1L) ctl$n.threads.outer <- THREADS_OUTER

# WHETHER THE OUTER GRID STAYS A TENSOR. `on` leaves the engine's adaptive
# refinement and var-of-means consistency pass running, which append mode-tracked
# slice cells: a new level on one axis at one combination of the others.
# `.hyper_log_quad_weights()` weights every cell per level over each axis's unique
# values, a tensor measure, so a slice cell narrows its neighbours' widths in every
# row while carrying width only in its own. On this fixture at J = 3 that moves
# the `sigma` posterior mean from 0.872 to 0.924 on the same base nodes
# (validation/RESULT_evidence_slice_cells.md). `off` disables both passes and
# leaves the stated nodes as a pure tensor. `on` sets nothing, so every arm on
# record reproduces from an unchanged environment.
REFINE <- Sys.getenv("SBC_REFINE", "on")
if (!REFINE %in% c("on", "off"))
  stop("SBC_REFINE must be 'on' or 'off', got ", sQuote(REFINE))
if (IS_NUTS && identical(REFINE, "off"))
  stop("SBC_REFINE = 'off' names an outer-grid pass; the nuts route has no grid")
if (identical(REFINE, "off")) {
  ctl$adaptive.grid <- FALSE
  ctl$var.of.means.consistency <- FALSE
}
if (!IS_NUTS) cat(sprintf("grid refinement: %s\n", REFINE))

# The THIRD accuracy axis, and the one that used to bind. The copy amplitude
# carries a declared atom at 0 plus Exponential(1) with a CCD split around it, so
# STATING its nodes would state a different prior; `alpha = grid(n = k)` instead
# re-reads the engine's own declared axis at a higher resolution and leaves the
# prior alone (tulpa nested_laplace_joint.R:117-121). Under copy() there was no
# way to ask, and the axis saturated near 11-13 surviving nodes whatever the rest
# of the grid did -- which is why grid ESS fell as the data sharpened and why
# J = 30 previously reached no configuration meeting the accuracy target.
#
# Set from a declared grid-ESS target on the BASE fit, measured on the grid this
# driver builds (validation/results/sbc/driver_calib/), never from any rank the
# run returns.
#
# READ THE NEXT PARAGRAPH BEFORE RAISING THIS. Grid ESS is a measure of how
# widely the outer weight is spread, not of how accurately the grid integrates
# any particular marginal, and the two come apart here. At J = 10, raising alpha
# to 21 nodes lifts base ESS from 42.0 to 50.7 and makes the three field/copy
# hyperparameters much WORSE: KS on sigma, sigma_pos_field and alpha goes from
# 0.070 / 0.150 / 0.133 to 0.205 / 0.236 / 0.199, p about 1e-14. The
# configuration that calibrates best on this fixture is the one that resolves
# SIGMA (21 nodes, ESS 37.3, all KS <= 0.084) at a LOWER ESS. So an ESS target
# selects the wrong knob, and alpha resolution is not the accuracy lever it
# looked like.
#
# 0 asks for no alpha resolution at all, which is a bare share(spatial()) and
# the engine's own declared axis. That is what the runs predating share() did,
# and it is the control arm any statement about the alpha axis has to be read
# against.
ALPHA_N <- as.integer(Sys.getenv("SBC_ALPHA_N", "9"))

# Parse tree, never a pasted string.
#
# Braced deliberately. At top level R closes the statement as soon as the `if`
# branch is a complete expression, so an `else` starting the next line is a
# parse error rather than the other half of the same call -- which is how the
# first version of this died three seconds into both decomposition arms.
pos_form <- if (ALPHA_N > 0L && !IS_NUTS) {
  eval(bquote(~ pos_cov1 + share(spatial(), alpha = grid(n = .(ALPHA_N)))))
} else {
  # `grid(n =)` asks an outer axis for resolution. A sampled fit has no axis, so
  # under nuts the copy amplitude is simply a sampled parameter and the bare
  # share() is the whole spec.
  ~ pos_cov1 + share(spatial())
}

fit <- tobs(~ occ_cov1 + icar(graph = adj),
            data = cbind(data.frame(site_id = seq_len(N)), sim$data),
            family = occu_cover(FAMILY),
            detection = ~ det_cov1,
            positive = pos_form,
            y = od$y, y_pos = y_pos, visits = od$det.covs,
            method = METHOD, control = ctl)

if (IS_NUTS) {
  hd <- fit$hyper_draws
  cat(sprintf("sampled posterior: %d draws | %d divergent | hyper columns %s
",
              nrow(fit$draws), sum(fit$divergent %||% 0L),
              paste(colnames(hd), collapse = ", ")))
  # The comparator's whole point, printed where it can be checked: these are
  # continuous, so the atom counts that bound the grid route do not apply here.
  for (q in intersect(c("sigma", "alpha"), colnames(hd)))
    cat(sprintf("  %-8s mean %.4f sd %.4f | %d distinct draws
",
                q, mean(hd[, q]), stats::sd(hd[, q]), length(unique(hd[, q]))))
} else {

og <- fit$joint_fit
cat(sprintf("outer grid: %d cells | ess %.1f | outer threads requested %s realised %s
",
            nrow(og$theta_grid), 1 / sum(og$weights^2),
            og$n_threads_outer_requested %||% THREADS_OUTER,
            og$n_threads_outer_realised %||% "unreported"))
slice <- og$refining_axis %||% rep("", nrow(og$theta_grid))
slice[is.na(slice)] <- ""
w_og <- og$weights; w_og[is.na(w_og)] <- 0
cat(sprintf("slice cells: %d | slice mass %.4f | sigma posterior mean %.4f\n",
            sum(slice != ""), sum(w_og[slice != ""]) / sum(w_og),
            og$theta_mean[["sigma"]] %||% NA_real_))
if (identical(REFINE, "off") && any(slice != ""))
  stop("SBC_REFINE = 'off' but the base fit carries ", sum(slice != ""),
       " slice cells; the grid is not a tensor")
for (a in colnames(og$theta_grid)) {
  m <- tapply(og$weights, og$theta_grid[, a], sum); m <- m / sum(m)
  v <- as.numeric(names(m))
  cat(sprintf("  %-8s %2d nodes [%.3g, %.3g] end-node mass %.4f / %.4f
",
              a, length(m), min(v), max(v), m[[1]], m[[length(m)]]))
}
}

# --- Run SBC -----------------------------------------------------------------
t0 <- Sys.time()
res <- sbc(fit, n.sim = N_SIM, controls = c("wide", "narrow"),
           fit.control = ctl, seed = SEED, level = LEVEL,
           control = list(progress = TRUE))
cat(sprintf("SBC wall clock: %.1f min\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

print(res)

# --- Persist -----------------------------------------------------------------
saveRDS(res, file.path(OUT, "sbc_occu_cover.rds"))
write.csv(res$report, file.path(OUT, "sbc_report.csv"), row.names = FALSE)

# The same ranks read at the family-wise level. Only the `inside` columns move --
# `ks` and `p_unif` do not depend on the band -- so this re-reads the stored PIT
# values rather than refitting anything. `sbc_report()` is tulpa-internal and
# this is a validation script rather than package code, so it is reached
# directly; the alternative is a second 300-simulation run for one column.
n_q <- length(unique(res$report$quantity[res$report$arm == "posterior"]))
LEVEL_FW <- LEVEL^(1 / n_q)
cat(sprintf("family-wise: %d quantities scored, per-quantity level %.4f for a joint %.2f
",
            n_q, LEVEL_FW, LEVEL))
if (n_q != 11L)
  cat(sprintf("NOTE: %d quantities scored, not the fixture's usual 11; raw counts are not comparable to an 11-quantity arm without saying so.
", n_q))

rep_fw <- tulpa:::sbc_report(res$pit, level = LEVEL_FW)
write.csv(rep_fw, file.path(OUT, "sbc_report_familywise.csv"), row.names = FALSE)

is_control <- res$report$arm %in% c("wide", "narrow")

# Figures: build_sbc_figure.R (reads sbc_occu_cover.rds).

# --- One-line verdict for the reply / report ---------------------------------
rep <- res$report
real <- rep[!is_control, ]
ctrl <- rep[is_control, ]
cat(sprintf("\nREAL arms: %d/%d inside the %.0f%% per-quantity band (min p_unif %.3g).\n",
            sum(real$inside), nrow(real), 100 * res$level, min(real$p_unif)))
if (nrow(ctrl))
  cat(sprintf("CONTROL arms: %d/%d inside (want LOW -- they must fail).\n",
              sum(ctrl$inside), nrow(ctrl)))

real_fw <- rep_fw[!rep_fw$arm %in% c("wide", "narrow"), ]
ctrl_fw <- rep_fw[rep_fw$arm %in% c("wide", "narrow"), ]
cat(sprintf("REAL arms: %d/%d raw and %d/%d folded inside the family-wise band ",
            sum(real_fw$inside), nrow(real_fw),
            sum(real_fw$inside_folded), nrow(real_fw)))
cat(sprintf("(per-quantity %.4f, joint %.2f over %d quantities).\n",
            LEVEL_FW, LEVEL, nrow(real_fw)))
if (nrow(ctrl_fw))
  cat(sprintf("CONTROL arms at the family-wise band: %d/%d inside (want LOW).\n",
              sum(ctrl_fw$inside), nrow(ctrl_fw)))
if (nrow(real_fw) && sum(real_fw$inside) < nrow(real_fw))
  cat(sprintf("  still outside: %s\n",
              paste(real_fw$quantity[!real_fw$inside], collapse = ", ")))
cat("Saved:", normalizePath(OUT), "\n")
