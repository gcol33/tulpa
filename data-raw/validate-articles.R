script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
root <- normalizePath(file.path(dirname(script), ".."), winslash = "/")
.libPaths(c(file.path(root, "lib"), .libPaths()))
tracked <- system2("git", c("-C", shQuote(root), "ls-files", "vignettes"), stdout = TRUE)
sources <- tracked[grepl("\\.Rmd$", tracked)]
rows <- lapply(sources, function(source) {
  input <- file.path(root, source)
  name <- tools::file_path_sans_ext(basename(input))
  lines <- readLines(input, warn = FALSE)
  headers <- grep("^```\\{r", lines, value = TRUE)
  fit_headers <- headers[grepl(", fit = ", headers, fixed = TRUE)]
  labels <- sub("[,}].*", "", sub("^```\\{r *", "", fit_headers))
  cache_file <- file.path(dirname(input), "fits", paste0(name, ".rds"))
  env <- new.env(parent = globalenv())
  cache <- readRDS(cache_file, refhook = function(key) env)
  stopifnot(identical(cache$version, as.character(utils::packageVersion("tulpa"))),
            setequal(labels, names(cache$chunks)))
  error_labels <- sub("[,}].*", "", sub("^```\\{r *", "",
    headers[grepl("error *= *TRUE", headers)]))
  errors <- vapply(cache$chunks, function(chunk)
    sum(vapply(chunk$output, inherits, logical(1), "error")), integer(1))
  stopifnot(all(names(errors)[errors > 0L] %in% error_labels))
  page <- file.path(root, "docs/articles", paste0(name, ".html"))
  html <- xml2::read_html(page)
  images <- xml2::xml_attr(xml2::xml_find_all(html, "//img[contains(@class, 'r-plt')]"), "src")
  stopifnot(all(file.exists(file.path(dirname(page), images))))
  figures <- sum(grepl("fig.alt", headers, fixed = TRUE))
  stopifnot(length(images) >= figures)
  text <- xml2::xml_text(html)
  normalize <- function(x) gsub("[[:space:]]+", " ", trimws(x))
  visible_text <- normalize(text)
  for (label in labels) {
    header <- fit_headers[match(label, labels)]
    if (grepl('results *= *"hide"|include *= *FALSE', header)) next
    output <- Filter(is.character, cache$chunks[[label]]$output)
    for (block in output) {
      lines <- trimws(strsplit(block, "\n", fixed = TRUE)[[1]])
      lines <- lines[nzchar(lines)]
      if (length(lines) && !grepl(normalize(lines[1]), visible_text, fixed = TRUE)) {
        stop("Missing fit output: ", name, " / ", label, " / ", lines[1])
      }
    }
  }
  outputs <- lengths(lapply(cache$chunks, function(chunk)
    Filter(is.character, chunk$output)))
  if (sum(outputs)) stopifnot(grepl("#>", text, fixed = TRUE))
  data.frame(article = name, cached_chunks = length(labels),
             bytes = file.info(cache_file)$size, figures = length(images),
             expected_errors = sum(errors))
})
result <- do.call(rbind, rows)
print(result, row.names = FALSE)
cat(sprintf("TOTAL\t%d pages\t%d fit chunks\t%.1f KiB\t%d figures\n",
            nrow(result), sum(result$cached_chunks), sum(result$bytes) / 1024,
            sum(result$figures)))
links <- unlist(lapply(sources[dirname(sources) == "vignettes"], function(source) {
  text <- paste(readLines(file.path(root, source), warn = FALSE), collapse = "\n")
  matches <- regmatches(text, gregexpr("https://gillescolling.com/tulpa/articles/[a-z0-9-]+\\.html", text))[[1]]
  unique(matches)
}))
stopifnot(all(file.exists(file.path(root, "docs/articles", basename(links)))))
cat("All", length(unique(links)), "article links in shipped vignettes resolve locally\n")
