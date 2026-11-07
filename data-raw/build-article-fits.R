args <- commandArgs(trailingOnly = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
root <- normalizePath(file.path(dirname(script), ".."), winslash = "/")
.libPaths(c(file.path(root, "lib"), .libPaths()))
rscript <- file.path(R.home("bin"),
                     if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
Sys.setenv(OMP_NUM_THREADS = "2", OPENBLAS_NUM_THREADS = "1")
if (length(args) >= 2L && args[1] == "--article") {
  input <- normalizePath(file.path(root, args[2]), winslash = "/")
  options(tulpa.fit_cache.build = TRUE, tulpa.fit_cache.checkpoint = TRUE)
  knitr::opts_chunk$set(error = FALSE)
  work <- tempfile("article-fits-")
  dir.create(work)
  knitr::opts_chunk$set(fig.path = paste0(work, "/fig/"))
  setwd(dirname(input))
  checkpoint <- tryCatch({
    knitr::knit(input, output = file.path(work, "article.md"),
                envir = new.env(parent = globalenv()), quiet = TRUE,
                encoding = "UTF-8")
    FALSE
  }, tulpa_fit_cache_checkpoint = function(e) TRUE)
  unlink(work, recursive = TRUE)
  if (checkpoint) quit(status = 10L)
} else {
  tracked <- system2("git", c("-C", shQuote(root), "ls-files", "vignettes"), stdout = TRUE)
  articles <- tracked[grepl("\\.Rmd$", tracked)]
  if (length(args)) {
    selected <- tools::file_path_sans_ext(basename(articles)) %in% args
    if (!all(args %in% tools::file_path_sans_ext(basename(articles)))) stop("Unknown article")
    articles <- articles[selected]
  }
  stopifnot(length(articles) > 0L)
  message("Building caches for ", length(articles), " tracked articles/vignettes")
  started <- proc.time()[["elapsed"]]
  for (article in articles) {
    repeat {
      status <- system2(rscript,
                        c("--vanilla", shQuote(script), "--article", shQuote(article)))
      if (status != 10L) break
    }
    if (status != 0L) stop("Cache build failed: ", article)
  }
  message(sprintf("Cache build: %.2f seconds", proc.time()[["elapsed"]] - started))
}
