# Quantiles of each arm against the reference sampler, with a Monte Carlo
# standard error for the reference median from batch means.
args <- commandArgs(trailingOnly = TRUE)
prefix <- args[1]
d <- "C:/GillesC/Documents/dev/tulpa/dev_notes/issue761"
arms <- c(ref = "ref", old = "old", new = "new")
dr <- lapply(arms, function(a) {
  f <- file.path(d, sprintf("%s_%s.rds", prefix, a))
  if (file.exists(f)) readRDS(f)
})
probs <- c(0.05, 0.25, 0.5, 0.75, 0.95)
bm_se <- function(x, q, nb = 50L) {
  b <- split(x, cut(seq_along(x), nb, labels = FALSE))
  sd(vapply(b, quantile, numeric(1), probs = q)) / sqrt(nb)
}
for (v in colnames(dr$ref)) {
  cat("\n==", v, "==\n")
  tab <- t(vapply(Filter(Negate(is.null), dr), function(m)
    c(quantile(m[, v], probs), mean = mean(m[, v])), numeric(6)))
  print(round(tab, 4))
  present <- Filter(Negate(is.null), dr)
  cat("median batch-means SE:",
      paste(names(present), vapply(present, function(m) signif(bm_se(m[, v], 0.5), 3),
                                   numeric(1)), collapse = "  "), "\n")
}
