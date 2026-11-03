# One-time personal R library migration; independent of Agent3's Python pipeline.
options(timeout = 1800, Ncpus = as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "4")),
        repos = c(CRAN = "https://cloud.r-project.org"))
lib <- path.expand("~/R/x86_64-pc-linux-gnu-library/4.6")
stopifnot(getRversion() >= "4.6.0", getRversion() < "4.7.0")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
Sys.setenv(R_LIBS_USER = lib)
wanted <- readLines("scripts/r_home_setup/requested_packages.txt")
stopifnot(length(wanted) == 139L, !anyDuplicated(wanted))
logdir <- "logs/r_home_setup"
cat("Started:", format(Sys.time()), "\nLibrary:", lib, "\n")
print(.libPaths())
safe <- function(label, expr) {
    cat("\nSTEP:", label, "\n")
    tryCatch(force(expr), error = function(e) {
        cat("STEP ERROR:", label, conditionMessage(e), "\n")
        FALSE
    })
}
for (p in c("BiocManager", "remotes")) {
    if (!requireNamespace(p, quietly = TRUE)) install.packages(p, lib = lib)
}
cat("Bioconductor:", as.character(BiocManager::version()), "\n")
main <- setdiff(wanted, c("BCalm", "gghalves", "cmdstanr", "SeuratData", "SeuratDisk"))
todo <- setdiff(main, rownames(installed.packages()))
cat("Already installed:", length(main) - length(todo), "of", length(main), "CRAN/Bioc packages\n")
cat("To install:", paste(todo, collapse = ", "), "\n")
if (length(todo)) safe("CRAN and Bioconductor", BiocManager::install(
    todo, lib = lib, ask = FALSE, update = FALSE, Ncpus = getOption("Ncpus")))
for (p in "cmdstanr") {
    if (!requireNamespace(p, quietly = TRUE)) {
        repo <- "https://stan-dev.r-universe.dev"
        safe(p, install.packages(p, lib = lib, repos = c(repo, getOption("repos"))))
    }
}
pins <- c(SeuratData = "satijalab/seurat-data@3e51f44303069b64f5dc4d68e6a3d4a343f55c39",
          SeuratDisk = "mojaveazure/seurat-disk@877d4e18ab38c686f5db54f8cd290274ccdbe295",
          BCalm = "kircherlab/BCalm@d5164ba4b62107290085de376822fff07c7d40aa",
          gghalves = "erocoar/gghalves@e5c3c79e79f13a00795a2a7774794252d1980d86")
for (p in names(pins)) {
    sha <- sub(".*@", "", pins[[p]])
    current <- if (requireNamespace(p, quietly = TRUE)) packageDescription(p)$RemoteSha else NULL
    if (!identical(current, sha)) safe(p, remotes::install_github(
        pins[[p]], lib = lib, upgrade = "never", Ncpus = getOption("Ncpus"), build_vignettes = FALSE))
}
source("scripts/r_home_setup/verify_r_packages.R")
