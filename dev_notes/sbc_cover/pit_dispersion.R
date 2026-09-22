# Dispersion read of an SBC run's PIT values.
#
# sd(qnorm(PIT)) is 1 when the reported posterior has the right width, above 1
# when it is too narrow and below 1 when it is too wide. It answers "by how
# much", which a KS p-value does not, and it keeps its power at n_sim = 300
# where the folded KS test does not.
#
# Point it at a directory of result directories, each holding the driver's
# sbc_occu_cover.rds:
#
#   Rscript pit_dispersion.R <results_dir> [<tag> ...]
#
# The result directories live on LiSC under ~/cover_sbc/results/.

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("usage: Rscript pit_dispersion.R <results_dir> [<tag> ...]")
root <- args[1]
tags <- if (length(args) > 1) args[-1] else list.dirs(root, recursive = FALSE, full.names = FALSE)

read_pit <- function(tag) {
  f <- file.path(root, tag, "sbc_occu_cover.rds")
  if (!file.exists(f)) stop("no sbc_occu_cover.rds under ", tag)
  readRDS(f)$pit
}
P <- lapply(tags, read_pit)
names(P) <- tags

sd_z <- function(pit) sd(qnorm(pit[pit > 0 & pit < 1]))

quantities <- unique(P[[1]]$quantity[P[[1]]$arm == "posterior"])
n_sim <- sum(P[[1]]$arm == "posterior" & P[[1]]$quantity == quantities[1])
se <- 1 / sqrt(2 * (n_sim - 1))

cat(sprintf("posterior arm, n_sim = %d, se of sd(z) under calibration ~ %.3f\n", n_sim, se))
cat(sprintf("%-18s", "quantity"))
for (tag in tags) cat(sprintf("%14s", substr(tag, 1, 14)))
cat("\n")
for (q in quantities) {
  cat(sprintf("%-18s", q))
  for (tag in tags) {
    s <- P[[tag]][P[[tag]]$arm == "posterior" & P[[tag]]$quantity == q, ]
    cat(sprintf("%14.3f", sd_z(s$pit)))
  }
  cat("\n")
}
