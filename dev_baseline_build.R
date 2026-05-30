cat("compileAttributes...\n"); Rcpp::compileAttributes(".")
cat("install...\n")
ok <- tryCatch({ install.packages(".", repos=NULL, type="source", quiet=TRUE); TRUE },
              error=function(e){ cat("INSTALL ERROR:", conditionMessage(e), "\n"); FALSE })
cat("INSTALL_OK=", ok, "\n")
