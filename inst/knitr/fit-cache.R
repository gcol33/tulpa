local({
  input <- knitr::current_input(dir = TRUE)
  article <- tools::file_path_sans_ext(basename(input))
  cache_file <- file.path(dirname(input), "fits", paste0(article, ".rds"))
  building <- isTRUE(getOption("tulpa.fit_cache.build"))
  env <- knitr::knit_global()
  version <- as.character(utils::packageVersion("tulpa"))
  hash <- function(x) {
    path <- tempfile()
    on.exit(unlink(path))
    writeBin(serialize(x, NULL, version = 2), path)
    unname(tools::md5sum(path))
  }
  fail <- function(label, reason) {
    stop(sprintf("Fit cache for article '%s', chunk '%s': %s. Rebuild with data-raw/build-article-fits.R",
                 article, label, reason), call. = FALSE)
  }
  ref_read <- function(key) {
    if (!identical(key, "article_environment")) stop("Unknown cache environment")
    env
  }
  ref_write <- function(x) {
    if (identical(x, env)) "article_environment" else NULL
  }
  compact <- function(x, omit) {
    if (inherits(x, "tulpa_fit")) {
      x[intersect(names(x), omit)] <- NULL
    } else if (is.list(x) && !is.data.frame(x)) {
      for (i in seq_along(x)) x[i] <- list(compact(x[[i]], omit))
    }
    x
  }
  cache <- if (file.exists(cache_file)) readRDS(cache_file, refhook = ref_read) else NULL
  if (!building && is.null(cache)) fail("setup", "file is missing")
  if (!building && !identical(cache$version, version)) {
    fail("setup", sprintf("tulpa version is %s, cache was built with %s", version, cache$version))
  }
  if (is.null(cache) || !identical(cache$version, version)) {
    cache <- list(version = version, chunks = list())
  }
  chunks <- knitr::knit_code$get()
  source_hashes <- vapply(chunks, hash, character(1))
  evaluate <- knitr::knit_hooks$get("evaluate")
  if (!is.null(attr(evaluate, "tulpa.evaluate"))) {
    evaluate <- attr(evaluate, "tulpa.evaluate")
  }
  knitr::opts_hooks$set(fit = function(options) {
    if (!is.null(options$fit)) {
      options$cache <- FALSE
      if (!is.null(cache$chunks[[options$label]])) options$eval <- TRUE
    }
    options
  })
  cached_evaluate <- function(code, envir, ...) {
    opts <- knitr::opts_current$get()
    label <- opts$label
    prefix <- seq_len(match(label, names(source_hashes)))
    stamp <- hash(list(code = code, objects = opts$fit, upstream = source_hashes[prefix]))
    if (length(opts$fit_omit)) stamp <- hash(list(stamp, omit = opts$fit_omit))
    if (is.null(opts$fit)) return(evaluate(code, envir = envir, ...))
    names <- if (isTRUE(opts$fit)) character() else opts$fit
    if (!is.character(names) || anyNA(names) || anyDuplicated(names)) {
      fail(label, "fit must be TRUE or unique object names")
    }
    saved <- cache$chunks[[label]]
    valid <- identical(saved$hash, stamp) && identical(names(saved$objects), names)
    if (!valid && !building) fail(label, "code or upstream chunks have changed")
    if (building && isTRUE(opts$fit_stateful)) valid <- FALSE
    if (valid) {
      list2env(saved$objects, envir = envir)
      if (!is.null(saved$seed)) assign(".Random.seed", saved$seed, globalenv())
      return(saved$output)
    }
    message(sprintf("Caching %s / %s", article, label))
    started <- proc.time()[["elapsed"]]
    output <- evaluate(code, envir = envir, ...)
    objects <- mget(names, envir = envir, inherits = FALSE)
    if (length(opts$fit_omit)) objects <- compact(objects, opts$fit_omit)
    cache$chunks[[label]] <<- list(
      hash = stamp, objects = objects, output = output,
      seed = get0(".Random.seed", globalenv()),
      seconds = proc.time()[["elapsed"]] - started)
    dir.create(dirname(cache_file), recursive = TRUE, showWarnings = FALSE)
    tmp <- paste0(cache_file, ".tmp")
    saveRDS(cache, tmp, compress = "xz", refhook = ref_write)
    if (!file.copy(tmp, cache_file, overwrite = TRUE)) stop("Cannot write fit cache")
    unlink(tmp)
    message(sprintf("Cache %s: %.1f KiB; chunk %.2f seconds", article,
                    file.info(cache_file)$size / 1024,
                    cache$chunks[[label]]$seconds))
    if (isTRUE(getOption("tulpa.fit_cache.checkpoint")) && !isTRUE(opts$fit_stateful)) {
      stop(structure(list(message = "Fit cache checkpoint", call = NULL),
                     class = c("tulpa_fit_cache_checkpoint", "error", "condition")))
    }
    output
  }
  attr(cached_evaluate, "tulpa.evaluate") <- evaluate
  knitr::knit_hooks$set(evaluate = cached_evaluate)
})
