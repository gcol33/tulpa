# Census of the functions the engine compiles (gcol33/tulpa#1193).
#
#   Rscript tools/instantiations.R [path/to/tulpa.dll]
#
# Reads the unity objects in src/ (build them first, e.g. with
# pkgbuild::compile_dll()) and reports the unique function symbols by owner,
# the functions compiled in more than one unity group, the std::function
# handlers by enclosing function, and the template families with the most
# instantiations. Given a DLL it also reports the size of its .pdata table: one
# 12-byte entry per function, which is what `checking compiled code` reads back
# through `objdump -x`.

bin_tool <- function(name) {
  path <- Sys.which(name)
  if (!nzchar(path)) {
    rtools <- Sys.getenv("RTOOLS45_HOME", "C:/rtools45")
    path <- file.path(rtools, "x86_64-w64-mingw32.static.posix", "bin", paste0(name, ".exe"))
  }
  if (!file.exists(path)) stop("cannot find ", name)
  path
}

nm <- bin_tool("nm")
objdump <- bin_tool("objdump")

objects <- Sys.glob(file.path("src", "unity_*.o"))
if (!length(objects)) stop("no src/unity_*.o; build the package first")

rows <- do.call(rbind, lapply(objects, function(o) {
  out <- system2(nm, c("-C", "--defined-only", shQuote(o)), stdout = TRUE)
  m <- regmatches(out, regexec("^[0-9a-f]+ ([TtWw]) (.*)$", out))
  m <- m[lengths(m) == 3L]
  sym <- vapply(m, `[[`, "", 3L)
  sym <- sym[!startsWith(sym, ".")]
  if (!length(sym)) return(NULL)
  data.frame(unit = basename(o), sym = sym, stringsAsFactors = FALSE)
}))

head_of <- function(s) substr(sub("\\(.*$", "", s), 1L, 300L)
owner_of <- function(s) {
  h <- head_of(s)
  out <- rep("other", length(h))
  for (k in rev(c("tulpa", "Eigen::", "Rcpp::", "std::", "__gnu_cxx", "tinyformat", "arena::")))
    out[grepl(k, h, fixed = TRUE)] <- k
  out
}

units_per_sym <- tapply(rows$unit, rows$sym, function(u) length(unique(u)))
syms <- names(units_per_sym)
owner <- owner_of(syms)

cat(sprintf("compiled function bodies: %d, unique: %d, compiled in >1 unity group: %d (%d redundant compiles)\n\n",
            nrow(rows), length(syms), sum(units_per_sym > 1L), sum(units_per_sym - 1L)))
tab <- data.frame(
  owner = names(table(owner)),
  unique = as.integer(table(owner)),
  in_several_groups = as.integer(tapply(units_per_sym > 1L, owner, sum)),
  redundant_compiles = as.integer(tapply(units_per_sym - 1L, owner, sum))
)
print(tab[order(-tab$unique), ], row.names = FALSE)

# Family = qualified name with template arguments removed.
family_of <- function(s) {
  h <- head_of(s)
  repeat {
    h2 <- gsub("<[^<>]*>", "", h)
    if (identical(h2, h)) break
    h <- h2
  }
  sub("^.* ", "", h)
}
fam <- table(family_of(syms))
cat("\ntemplate families with the most instantiations:\n")
top <- head(sort(fam[fam > 1L], decreasing = TRUE), 30L)
print(data.frame(n = as.integer(top), family = names(top)), row.names = FALSE)

inv <- syms[startsWith(syms, "std::_Function_handler<") & grepl("_M_invoke", syms, fixed = TRUE)]
if (length(inv)) {
  encl <- sub("^.*?((?:\\w+::)*[\\w~]+)(?:<[^{]*>)?\\([^{]*?\\)(?: const)?::\\{lambda.*$", "\\1", inv, perl = TRUE)
  encl[encl == inv] <- "?"
  cat(sprintf("\nstd::function lambda types: %d\n", length(inv)))
  e <- head(sort(table(encl), decreasing = TRUE), 15L)
  print(data.frame(n = as.integer(e), enclosing = names(e)), row.names = FALSE)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args)) {
  h <- system2(objdump, c("-h", shQuote(args[1L])), stdout = TRUE)
  p <- grep(" \\.pdata ", h, value = TRUE)
  if (length(p)) {
    size <- strtoi(strsplit(trimws(p), "\\s+")[[1L]][3L], 16L)
    lines <- length(system2(objdump, c("-x", shQuote(args[1L])), stdout = TRUE))
    cat(sprintf("\n%s: %d functions in .pdata, %d lines of objdump -x\n", basename(args[1L]), size %/% 12L, lines))
  }
}
