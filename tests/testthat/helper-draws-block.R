# Draw columns of one parameter block of a sampler fit. `draws_block()` returns
# the scalar column named `stem` or the indexed columns `stem[k]`;
# `fixed_draws()` the leading fixed-effect block.
draws_block <- function(fit, stem) {
  cn <- colnames(fit$draws)
  fit$draws[, cn == stem | startsWith(cn, paste0(stem, "[")), drop = FALSE]
}

fixed_draws <- function(fit) {
  fit$draws[, seq_len(fit$n_fixed), drop = FALSE]
}
