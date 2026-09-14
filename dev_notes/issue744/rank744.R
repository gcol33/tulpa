# gcol33/tulpa#744: the #328 ranking comparison summarised the way
# R/nested_laplace_joint_ccd_local.R quotes it, off the CSVs run744.sh writes.
#
#   Rscript rank744.R <out_dir>
out <- commandArgs(trailingOnly = TRUE)[1L]
for (tag in c("flat", "proper", "flat_chord", "proper_chord")) {
  f <- file.path(out, paste0("fitrank_", tag, ".csv"))
  if (!file.exists(f)) { cat(tag, "missing\n"); next }
  s <- utils::read.csv(f)
  stopifnot(length(unique(s$build)) == 1L, length(unique(s$hyperprior)) == 1L,
            length(unique(s$within_cell)) == 1L)
  cat(sprintf("%-13s build %s prior %s read %s rows %d\n", tag, substr(s$build[1L], 1, 10),
              s$hyperprior[1L], s$within_cell[1L], nrow(s)))
  for (p in c("ep", "wd", "md")) {
    ew <- s[[paste0(p, "_weight")]]; em <- s[[paste0(p, "_move")]]
    fl <- s[[paste0("fl_", p)]]
    above <- abs(ew - em) > fl
    cat(sprintf("  %s |weight - moved| summed %.4f  floor summed %.4f  above floor %d of %d  weight %.4f  moved %.4f  moved nearer %d  moved nearer where resolved %d of %d\n",
                p, sum(abs(ew - em)), sum(fl), sum(above), nrow(s), sum(ew), sum(em),
                sum(em < ew), sum(em[above] < ew[above]), sum(above)))
  }
}
