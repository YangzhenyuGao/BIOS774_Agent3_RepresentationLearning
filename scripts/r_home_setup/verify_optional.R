.libPaths(c(path.expand("~/R/x86_64-pc-linux-gnu-library/4.6"), .libPaths()))
optional <- readLines("scripts/r_home_setup/optional_packages.txt")
main <- readLines("scripts/r_home_setup/requested_packages.txt")
ip <- installed.packages()
result <- do.call(rbind, lapply(unique(c(main, optional)), function(p) {
    error <- ""
    ok <- tryCatch({ loadNamespace(p); TRUE }, error = function(e) {
        error <<- conditionMessage(e); FALSE
    })
    present <- p %in% rownames(ip)
    data.frame(package = p, scope = if (p %in% main) "main" else "appendix_B",
               installed = present, loads = ok,
               version = if (present) ip[p, "Version"] else "",
               library = if (present) ip[p, "LibPath"] else "", error = error)
}))
write.table(result, "logs/r_home_setup/optional_verification.tsv",
            sep = "\t", quote = TRUE, row.names = FALSE)
write.table(ip[, c("Package", "Version", "LibPath", "Built")],
            "logs/r_home_setup/optional_after.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(capture.output(sessionInfo()), "logs/r_home_setup/optional_sessionInfo.txt")
print(result[!result$loads, ], row.names = FALSE)
cat("VERIFIED main:", sum(result$loads[result$scope == "main"]), "/", length(main), "\n")
cat("VERIFIED appendix B:", sum(result$loads[result$package %in% optional]), "/", length(optional), "\n")
stopifnot(all(result$loads))
# Recheck the original pinned sources and compiled-library smoke tests as well.
source("scripts/r_home_setup/verify_r_packages.R")
