# Extend the personal Sycamore R library with all of install.md Appendix B.
options(timeout = 3600, Ncpus = as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "4")),
        repos = c(CRAN = "https://cloud.r-project.org"))
lib <- path.expand("~/R/x86_64-pc-linux-gnu-library/4.6")
stopifnot(getRversion() >= "4.6.0", getRversion() < "4.7.0", dir.exists(lib))
.libPaths(c(lib, .libPaths()))
Sys.setenv(R_LIBS_USER = lib)
wanted <- readLines("scripts/r_home_setup/optional_packages.txt")
stopifnot(length(wanted) == 169L, !anyDuplicated(wanted))
ip <- installed.packages()
write.table(ip[, c("Package", "Version", "LibPath", "Built")],
            "logs/r_home_setup/optional_before.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
todo <- setdiff(wanted, rownames(ip))
cat("Started:", format(Sys.time()), "\nLibrary:", lib,
    "\nBioconductor:", as.character(BiocManager::version()),
    "\nAlready installed:", length(wanted) - length(todo), "/", length(wanted),
    "\nTo install:", paste(todo, collapse = ", "), "\n")
if (length(todo)) tryCatch(
    BiocManager::install(todo, lib = lib, ask = FALSE, update = FALSE,
                         Ncpus = getOption("Ncpus")),
    error = function(e) cat("INSTALL ERROR:", conditionMessage(e), "\n"))
source("scripts/r_home_setup/verify_optional.R")
