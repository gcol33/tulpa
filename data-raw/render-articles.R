args <- commandArgs(trailingOnly = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
root <- normalizePath(file.path(dirname(script), ".."), winslash = "/")
.libPaths(c(file.path(root, "lib"), .libPaths()))
rscript <- file.path(R.home("bin"),
                     if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
Sys.unsetenv("NOT_CRAN")
Sys.setenv(OMP_NUM_THREADS = "2", OPENBLAS_NUM_THREADS = "1")
setwd(root)
normalize_html <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  writeLines(sub("[\t ]+$", "", lines), path, useBytes = TRUE)
}
if (length(args) >= 2L && args[1] %in% c("--article", "--vignette")) {
  library(tulpa)
  fitters <- c("tulpa", "tulpa_nested_laplace", "tulpa_nested_laplace_joint",
               "tulpa_em_laplace", "tulpa_em_mc", "fit_spde", "fit_st_nested",
               "tulpa_tgmrf", "tulpa_kfold", "tulpa_reloo", "sbc",
               "tulpa_laplace", "tulpa_laplace_beta", "tulpa_re_cov_nested",
               "tulpa_re_aghq", "tulpa_re_cov_gibbs", "tulpa_eb",
               "tulpa_multinomial", "tulpa_ordinal", "tulpa_gaussian",
               "tulpa_nuts_beta", "tulpa_nuts_spde", "mala", "pathfinder")
  .article_render_fit_calls <- 0L
  for (f in fitters) trace(f, tracer = quote({
    .GlobalEnv$.article_render_fit_calls <- .GlobalEnv$.article_render_fit_calls + 1L
    stop("Model fitting during render")
  }),
                          where = asNamespace("tulpa"), print = FALSE)
  knitr::opts_chunk$set(error = FALSE)
  started <- proc.time()[["elapsed"]]
  if (args[1] == "--article") {
    pkg <- pkgdown::as_pkgdown(root)
    pkgdown::build_article(args[2], pkg = pkg, new_process = FALSE, quiet = TRUE)
    normalize_html(file.path(pkg$dst_path,
      pkg$vignettes$file_out[match(args[2], pkg$vignettes$name)]))
  } else {
    rmarkdown::render(file.path(root, args[2]), output_dir = tempdir(),
                      envir = new.env(parent = globalenv()), quiet = TRUE)
  }
  stopifnot(.article_render_fit_calls == 0L)
  cat(sprintf("RENDER\t%s\t%.3f\n", args[2], proc.time()[["elapsed"]] - started))
} else {
  pkg <- pkgdown::as_pkgdown(root)
  stopifnot(nrow(pkg$vignettes) == length(system2(
    "git", c("-C", shQuote(root), "ls-files", "vignettes/*.Rmd"), stdout = TRUE)))
  pkgdown::init_site(pkg)
  started <- proc.time()[["elapsed"]]
  for (name in pkg$vignettes$name) {
    status <- system2(rscript,
                      c("--vanilla", shQuote(script), "--article", shQuote(name)))
    if (status != 0L) stop("Article render failed: ", name)
  }
  pkgdown::build_articles_index(pkg)
  normalize_html(file.path(pkg$dst_path, "articles/index.html"))
  cat(sprintf("RENDER_ALL\t%d\t%.3f\n", nrow(pkg$vignettes),
              proc.time()[["elapsed"]] - started))
  shipped <- pkg$vignettes$file_in[dirname(pkg$vignettes$file_in) == "vignettes"]
  started <- proc.time()[["elapsed"]]
  for (file in shipped) {
    status <- system2(rscript,
                      c("--vanilla", shQuote(script), "--vignette", shQuote(file)))
    if (status != 0L) stop("Vignette render failed: ", file)
  }
  cat(sprintf("RENDER_VIGNETTES\t%d\t%.3f\n", length(shipped),
              proc.time()[["elapsed"]] - started))
}
