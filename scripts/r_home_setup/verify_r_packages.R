lib <- path.expand("~/R/x86_64-pc-linux-gnu-library/4.6")
.libPaths(c(lib, .libPaths()))
wanted <- readLines("scripts/r_home_setup/requested_packages.txt")
ip <- installed.packages()
results <- lapply(wanted, function(p) {
    error <- ""
    ok <- tryCatch({ loadNamespace(p); TRUE }, error = function(e) { error <<- conditionMessage(e); FALSE })
    present <- p %in% rownames(ip)
    data.frame(package = p, installed = present, loads = ok,
               version = if (present) ip[p, "Version"] else "",
               library = if (present) ip[p, "LibPath"] else "", error = error)
})
result <- do.call(rbind, results)
write.table(result, "logs/r_home_setup/verification.tsv", sep = "\t", quote = TRUE, row.names = FALSE)
write.table(ip[, c("Package", "Version", "LibPath", "Built")],
            "logs/r_home_setup/after.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(capture.output(sessionInfo()), "logs/r_home_setup/sessionInfo.txt")
print(result[!result$loads, ], row.names = FALSE)
cat("VERIFIED:", sum(result$loads), "/", nrow(result), "packages load\n")
if (requireNamespace("cmdstanr", quietly = TRUE)) {
    cat("CmdStan path:", cmdstanr::cmdstan_path(), "\n")
    cat("CmdStan version:", as.character(cmdstanr::cmdstan_version()), "\n")
}
pins <- c(SeuratData = "3e51f44303069b64f5dc4d68e6a3d4a343f55c39",
          SeuratDisk = "877d4e18ab38c686f5db54f8cd290274ccdbe295",
          BCalm = "d5164ba4b62107290085de376822fff07c7d40aa",
          gghalves = "e5c3c79e79f13a00795a2a7774794252d1980d86")
pin_ok <- vapply(names(pins), function(p) {
    actual <- if (p %in% rownames(ip)) packageDescription(p)$RemoteSha else NULL
    cat(p, "RemoteSha:", actual, "\n")
    identical(actual, pins[[p]])
}, logical(1))
stopifnot(all(result$loads))
stopifnot(all(pin_ok))
probe <- tempfile(fileext = ".feather")
arrow::write_feather(data.frame(x = 1:3), probe)
stopifnot(identical(arrow::read_feather(probe)$x, 1:3))
unlink(probe)
probe <- tempfile(fileext = ".h5")
h <- hdf5r::H5File$new(probe, mode = "w")
h[["x"]] <- 1:3
stopifnot(identical(h[["x"]][], 1:3))
h$close_all()
unlink(probe)
stopifnot(magick::image_info(magick::image_blank(2, 2))$width == 2)
cat("Arrow/HDF5 roundtrips and ImageMagick image creation passed.\n")
# Attach BCalm's Depends, as in normal interactive use. mpra's validity method
# resolves Biobase::validMsg from the search path rather than importing it.
suppressPackageStartupMessages(library(BCalm))
counts <- matrix(as.integer(11:34), nrow = 4,
                 dimnames = list(paste0("bc", 1:4), paste0("s", 1:6)))
mpra_probe <- BCalm::MPRASet(DNA = counts, RNA = counts + 1L,
                           barcode = rownames(counts), eid = c("e1", "e1", "e2", "e2"),
                           label = c(e1 = "x", e2 = "y"))
stopifnot(methods::validObject(mpra_probe), identical(BCalm::getRNA(mpra_probe), counts + 1L))
cat("BCalm MPRASet construction and count retrieval passed.\n")
cat("All requested packages passed namespace loading.\n")
