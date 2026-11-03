# Canonical Seurat/Signac training-only WNN graph reference. No shared input writes.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 1L)
destination <- normalizePath(args[[1]], mustWork=TRUE)
project <- "/vast/som/yunligrp/users/ygao/BIOS774Agent/Agent3_RepresentationLearning/"
stopifnot(startsWith(paste0(destination, "/"), project))
suppressPackageStartupMessages({ library(Seurat); library(Signac); library(Matrix) })
set.seed(774)
rna <- as(readMM(file.path(destination, "rna.mtx")), "CsparseMatrix")
atac <- as(readMM(file.path(destination, "atac.mtx")), "CsparseMatrix")
ids <- readLines(file.path(destination, "barcodes.txt"))
genes <- readLines(file.path(destination, "genes.txt"))
peaks <- readLines(file.path(destination, "peaks.txt"))
rownames(rna) <- make.unique(genes)
rownames(atac) <- peaks
colnames(rna) <- colnames(atac) <- ids
object <- CreateSeuratObject(counts=rna)
object[["ATAC"]] <- CreateChromatinAssay(counts=atac, sep=c(":", "-"), min.cells=0, min.features=0)
object <- NormalizeData(object, verbose=FALSE)
object <- FindVariableFeatures(object, nfeatures=min(3000L, nrow(rna)), verbose=FALSE)
object <- ScaleData(object, verbose=FALSE)
object <- RunPCA(object, npcs=32, verbose=FALSE, seed.use=774)
DefaultAssay(object) <- "ATAC"
object <- RunTFIDF(object, verbose=FALSE)
object <- FindTopFeatures(object, min.cutoff=0, verbose=FALSE)
object <- RunSVD(object, n=33, verbose=FALSE, seed.use=774)
depth <- log1p(Matrix::colSums(atac))
correlation <- cor(Embeddings(object, "lsi")[,1], depth)
dims <- if (abs(correlation) > 0.5) 2:33 else 1:32
object <- FindMultiModalNeighbors(object, reduction.list=list("pca", "lsi"), dims.list=list(1:32,dims),
                                  modality.weight.name=c("RNA.weight", "ATAC.weight"), verbose=FALSE)
writeMM(object[["wsnn"]], file.path(destination, "seurat_wsnn.mtx"))
write.csv(data.frame(barcode=ids, RNA.weight=object$RNA.weight), file.path(destination, "seurat_weights.csv"), row.names=FALSE)
writeLines(capture.output(sessionInfo()), file.path(destination, "sessionInfo.txt"))
write.csv(data.frame(training_cells=length(ids), first_lsi_depth_correlation=correlation,
                     dropped_first=abs(correlation)>0.5, labels_used=FALSE),
          file.path(destination, "reference_metadata.csv"), row.names=FALSE)
