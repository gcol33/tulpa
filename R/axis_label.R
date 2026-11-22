# Outer-axis labels and the roles they carry.
#
# An outer axis plays one of three roles: a hyperparameter of a latent block,
# the dispersion of a response arm, or the copy scale coupling a block (or the
# field of the single-block joint path) into a second arm. The label grammar is
# one rule and these helpers are the only code that writes or reads it:
#
#   b<k>.<hyper>   hyperparameter `hyper` of latent block k (1-based); every
#                  block axis carries its block, a single block included, so
#                  a label never depends on how many blocks the fit holds
#   phi_<arm>      the dispersion of response arm `arm`
#   <hyper>        an axis of the single-block joint path (`sigma`, the arm
#                  copy scale `alpha`), which holds no block list
#
# The copy scale is a role, never a name: it is the `alpha` axis the copy
# machinery lays on a copy block (or on the field of the single-block joint
# path). A block that declares its own axes on its own coordinate (a `tgmrf()`
# block, `.nl_linear_axes()`) may call one of them `alpha`, and that axis is the
# block's coordinate (`.axis_is_copy()`).

.AXIS_BLOCK_RE <- "^b([0-9]+)[.]"

# Label of hyperparameter `hyper` of block `block`; `NA` gives the block-free
# label.
.axis_label <- function(block, hyper) {
  block <- rep_len(as.integer(block), length(hyper))
  ifelse(is.na(block), as.character(hyper), paste0("b", block, ".", hyper))
}

# The prefix a label of block `block` starts with.
.axis_prefix <- function(block) paste0("b", as.integer(block), ".")

# Hyperparameter name of each label, without its block.
.axis_bare <- function(label) sub(.AXIS_BLOCK_RE, "", as.character(label %||% ""))

# Whether each label names a block axis.
.axis_has_block <- function(label) grepl(.AXIS_BLOCK_RE, as.character(label))

# Block index of each label, `NA` for a block-free one.
.axis_block <- function(label) {
  label <- as.character(label)
  out <- rep(NA_integer_, length(label))
  hit <- .axis_has_block(label)
  out[hit] <- as.integer(sub(paste0(.AXIS_BLOCK_RE, ".*"), "\\1", label[hit]))
  out
}

# The block prefix each label carries, `""` for a block-free one.
.axis_label_prefix <- function(label) {
  b <- .axis_block(label)
  ifelse(is.na(b), "", .axis_prefix(b))
}

# Whether each label is a response arm's dispersion.
.axis_is_dispersion <- function(label) {
  label <- as.character(label)
  !.axis_has_block(label) & startsWith(label, "phi_")
}

# Whether each label is a copy scale: an `alpha` axis that no block declares
# itself. `block_declared` are the labels of the axes blocks declare on their
# own coordinate (`.nl_linear_axes()`).
.axis_is_copy <- function(label, block_declared = character(0)) {
  label <- as.character(label)
  .axis_bare(label) == "alpha" & !label %in% block_declared
}
