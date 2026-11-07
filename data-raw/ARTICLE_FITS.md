The R Markdown sources contain the fitting code. A chunk option such as
`fit = c("fit", "comparison")` names the objects used after the chunk.
`fit = TRUE` records output from a chunk whose objects are only used inside
that chunk. Include calls through local helpers, replication studies,
cross-validation, comparison-package fits, and expected-error examples.

Each setup chunk sources `system.file("knitr/fit-cache.R", package = "tulpa")`.
The hook restores the named objects, the random-number state, and the
recorded output. Other chunks evaluate normally. The RDS files contain the
tulpa version and hashes of the fit code and preceding chunk code. Rendering
refuses missing or stale caches with the article and chunk names.

Install the package, then run `Rscript data-raw/build-article-fits.R` from
the repository. Optional arguments select article basenames. Each tracked
article or vignette runs against the installed package. Each newly computed
fit chunk ends its R session; the next session restores the cached prefix.
For an article whose fitting examples share files, set the chunk default
`fit_stateful = TRUE` in setup. Its cache is built from the start in one
session so file creation, interruption and resume form one computation.
`fit_omit` lists fit fields that the article does not read. Use it for large
stored matrices only after checking the article's readers against the full
fit. The omission list is part of the cache hash.
The builder prints elapsed time and compressed cache size after each chunk,
and reuses unchanged cached prefixes when resumed. Delete an article's RDS
to recompute all its fits. Article caches live in `vignettes/articles/fits/`;
the three shipped vignettes use `vignettes/fits/`, included in source builds.

The builder selects sources with `git ls-files`. To adopt a new article,
add the setup source line, mark every fitting chunk with its required object
names, track the source, and run the builder with its basename. Keep plotting
and summaries in ordinary chunks so they execute during every render.
