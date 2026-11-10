# Stamp of an article fit-cache input. Bytes 7-10 of an XDR serialization
# header record the version of R that wrote it, so they are zeroed: the stamp
# then depends on the object alone and a cache built under one R release is
# read under another.
.fit_cache_hash <- function(x) {
  bytes <- serialize(x, NULL, xdr = TRUE, version = 2L)
  bytes[7:10] <- as.raw(0L)
  path <- tempfile()
  on.exit(unlink(path))
  writeBin(bytes, path)
  unname(tools::md5sum(path))
}
