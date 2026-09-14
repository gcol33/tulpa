args <- commandArgs(TRUE); F <- args[1]
strip <- function(x) {
  if (is.environment(x) || is.function(x) || inherits(x, "externalptr")) return(NULL)
  if (is.list(x)) {
    nm <- names(x)
    if (!is.null(nm)) x <- x[!grepl("tim|elapsed|seconds", nm)]
    x[] <- lapply(x, strip)
  }
  x
}
H <- readRDS(file.path(F, "id_head.rds"))
for (arm in c("hp_default", "hp_proper")) {
  A <- readRDS(file.path(F, paste0("id_", arm, ".rds")))
  for (k in names(H)) {
    a <- strip(A[[k]]); h <- strip(H[[k]])
    cat(sprintf("%-11s %-12s identical=%s  n_fields=%d  identical(log_marginal)=%s identical(summary coef)=%s\n",
                arm, k, identical(a, h), length(h), identical(A[[k]]$log_marginal, H[[k]]$log_marginal),
                identical(A[[k]]$weights, H[[k]]$weights)))
    if (!identical(a, h)) {
      d <- names(h)[!vapply(names(h), function(n) identical(a[[n]], h[[n]]), logical(1))]
      cat("   differing fields:", paste(d, collapse = ", "), "\n")
    }
  }
}
Fl <- readRDS(file.path(F, "id_hp_flat.rds"))
for (k in names(Fl)) {
  f <- Fl[[k]]
  cat(sprintf("flat %-12s cells %d  sum|log_hyperprior| %.3g  declined: %s  evidence %s (%s)\n", k,
              length(f$log_marginal), sum(abs(f$log_hyperprior %||% 0)),
              paste(unique(unlist(f$log_hyperprior_declined)), collapse = ","),
              format(f$log_evidence), paste(f$log_evidence_declined, collapse = ",")))
}
