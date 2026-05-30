Rcpp::compileAttributes(".")
res <- tools:::.install_packages(c("--no-multiarch", "--no-docs", "."))
