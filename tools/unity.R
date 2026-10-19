# Unity build of src/: each src/unity_<group>.cpp #includes the translation
# units of one group, and the package compiles those files alone. The member
# lists in the unity files are the grouping; this script checks them and writes
# the OBJECTS line and the per-group prerequisites into src/Makevars and
# src/Makevars.win.
#
#   Rscript tools/unity.R          # check membership, rewrite the Makevars blocks
#   Rscript tools/unity.R --check  # check only; non-zero exit if a block is stale
#
# Run it from the package root after adding, removing or regrouping a .cpp.

args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args
src <- "src"

unity <- sort(list.files(src, "^unity_.*\\.cpp$"))
if (!length(unity)) stop("no src/unity_*.cpp files")
members <- lapply(unity, function(u) {
  x <- readLines(file.path(src, u))
  m <- regmatches(x, regexec('^#include "([^"/]+\\.cpp)"', x))
  vapply(Filter(length, m), `[[`, "", 2L)
})
names(members) <- unity

tus <- setdiff(list.files(src, "\\.cpp$"), unity)
grouped <- unlist(members, use.names = FALSE)
dup <- unique(grouped[duplicated(grouped)])
missing <- setdiff(tus, grouped)
unknown <- setdiff(grouped, tus)
if (length(dup)) stop("in more than one unity file: ", paste(dup, collapse = ", "))
if (length(missing)) stop("in no unity file: ", paste(missing, collapse = ", "))
if (length(unknown)) stop("named by a unity file but absent from src/: ", paste(unknown, collapse = ", "))

objs <- sub("\\.cpp$", ".o", unity)
# One make line per item after the head, continued with a backslash.
wrap <- function(head, items) {
  c(paste(head, "\\"), paste0("  ", items, c(rep(" \\", length(items) - 1L), "")))
}
block <- c(
  "# unity-build: begin (written by tools/unity.R)",
  wrap("OBJECTS =", objs),
  "all: $(SHLIB)",
  unlist(lapply(seq_along(unity), function(i) wrap(paste0(objs[i], ":"), c(unity[i], members[[i]])))),
  "# unity-build: end"
)

stale <- character()
for (mk in file.path(src, c("Makevars", "Makevars.win"))) {
  x <- readLines(mk)
  b <- grep("^# unity-build: begin", x); e <- grep("^# unity-build: end", x)
  new <- if (length(b) == 1L && length(e) == 1L && e > b) {
    c(x[seq_len(b - 1L)], block, x[-seq_len(e)])
  } else if (!length(b) && !length(e)) {
    c(x, "", block)
  } else stop(mk, ": unbalanced unity-build markers")
  if (!identical(new, x)) {
    stale <- c(stale, mk)
    if (!check_only) { con <- file(mk, "wb"); writeLines(new, con, sep = "\n"); close(con) }
  }
}
cat(sprintf("%d translation units in %d groups\n", length(tus), length(unity)))
if (length(stale)) {
  cat(if (check_only) "stale:" else "rewrote:", paste(stale, collapse = ", "), "\n")
  if (check_only) quit(status = 1L)
}
