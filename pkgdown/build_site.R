# Deploy entry point for the pkgdown site. Use this rather than calling
# `pkgdown::build_site()` directly.
#
# `pkgdown:::package_mds()` renders EVERY `*.md` at the package root into a page
# and into the site search index, minus a hard-coded exclusion list (README,
# LICENSE, NEWS, the GitHub templates, cran-comments). There is no config knob to
# extend that list, so `CLAUDE.md`, `AGENTS.md`, `api.md` and `todo.md` are
# published verbatim and become the first hits in site search.
#
# They are COPIED to a stash and deleted from the root for the duration of the
# build, then copied back. Copy-then-delete rather than move: a move leaves the
# only copy of an untracked file outside the repository, and if the restore does
# not run that file is gone. The stash lives beside the repository rather than in
# `tempdir()`, which R removes when the session exits.
#
# The restore runs from a function's `on.exit`, not at top level. `on.exit()`
# called at top level in `Rscript` registers against no frame and never fires.

INTERNAL_MD <- c("CLAUDE.md", "AGENTS.md", "api.md", "todo.md")

build_site_without_internal_md <- function(root = ".",
                                           internal = INTERNAL_MD) {
  root  <- normalizePath(root, winslash = "/")
  stash <- file.path(root, ".pkgdown-stash")
  dir.create(stash, showWarnings = FALSE, recursive = TRUE)

  present <- internal[file.exists(file.path(root, internal))]
  staged  <- character(0)

  restore <- function() {
    failed <- character(0)
    for (f in staged) {
      src <- file.path(stash, f)
      dst <- file.path(root, f)
      if (!file.exists(src)) {
        failed <- c(failed, f)
        next
      }
      if (!file.copy(src, dst, overwrite = TRUE)) {
        failed <- c(failed, f)
        next
      }
      unlink(src)
    }
    if (length(failed)) {
      stop("could not restore ", paste(failed, collapse = ", "),
           " -- the copies are in ", stash, call. = FALSE)
    }
    if (!length(list.files(stash, all.files = TRUE, no.. = TRUE))) {
      unlink(stash, recursive = TRUE)
    }
  }
  on.exit(restore(), add = TRUE)

  for (f in present) {
    if (!file.copy(file.path(root, f), file.path(stash, f), overwrite = TRUE)) {
      stop("could not stash ", f, call. = FALSE)
    }
    staged <- c(staged, f)
    # Only now is it safe to remove the original: a copy exists on disk, inside
    # the repository, and survives this process being killed.
    if (!file.remove(file.path(root, f))) {
      stop("could not remove ", f, " from the package root", call. = FALSE)
    }
  }
  message("held back from the site: ", paste(present, collapse = ", "))

  pkgdown::build_site(preview = FALSE, install = FALSE)

  # GitHub Pages runs Jekyll over `docs/` unless this file is present, and
  # Jekyll re-renders pkgdown's already-built markdown through Liquid. The
  # CITATION's BibTeX key reaches `authors.md` as `@Manual{tulpa,`, which Liquid
  # reads as an unterminated `{{tulpa}` variable and the whole deployment fails.
  # Nothing here needs Jekyll: the HTML is already built.
  nojekyll <- file.path(root, "docs", ".nojekyll")
  if (!file.exists(nojekyll)) file.create(nojekyll)

  html   <- sub("\\.md$", ".html", internal)
  leaked <- html[file.exists(file.path(root, "docs", html))]
  if (length(leaked)) {
    stop("internal pages reached docs/: ", paste(leaked, collapse = ", "),
         call. = FALSE)
  }
  idx <- file.path(root, "docs", "search.json")
  if (file.exists(idx)) {
    pat <- paste(gsub("\\.", "\\\\.", html), collapse = "|")
    if (any(grepl(pat, readLines(idx, warn = FALSE)))) {
      stop("internal pages reached the search index", call. = FALSE)
    }
  }
  invisible(present)
}

build_site_without_internal_md()
message("site built, internal pages absent from docs/ and search.json")
