# Input provenance and storage boundaries

Both input datasets and the four approved local model folders are read-only.
Agent3 writes only within its project root, except for explicitly authorized
package installation into its existing Conda prefix and the shared personal R
library. Acquisition scripts are preserved but not rerun during execution.

## PBMC Multiome

Source is 10x Genomics "PBMC from a Healthy Donor, Granulocytes Removed Through
Cell Sorting (10k)", Cell Ranger ARC 1.0.0, GRCh38, CC BY 4.0. Local root:
`/vast/som/yunligrp/users/ygao/BIOS774Agent/data/pbmc_granulocyte_sorted_10k`.
The filtered HDF5 has 11,909 cells, 36,601 Gene Expression features and 108,377
Peaks; modalities share barcode columns. Background cells from raw HDF5 do not
enter the representation experiment.

Labels come from the Satija lab's FixFlip CSV, with 19 cell types for 10,412 cells
and 1,497 `filtered` entries. The local LABEL_PROVENANCE.json documents that the
released SeuratData `seurat_annotations` swaps Naive/Memory B; that field is not
used. Expert labels derive from WNN clusters of these same cells, creating a
WNN structural advantage. Single-donor results cannot imply donor generalization.

Data and annotation SHA256SUMS were rechecked in Slurm job 5003331 and passed.
Training-only Agent3 QC retained 10,243 cells: 7,171 train, 1,532 validation and
1,540 test. Among retained cells, 10,078 have curated labels and 165 were labeled
`filtered` by the vignette; 334 vignette-labeled cells failed Agent3 QC. These
are pre-modeling cohort counts, not performance results.

## TCGA-BRCA

Local root: `/vast/som/yunligrp/users/ygao/imaging_fi/data/TCGA-BRCA`.
The other project's full directory, including processed/annotations/provenance,
must remain unmodified. Agent3 builds its own table from GDC raw clinical JSON
and UCSC Xena BRCA_clinicalMatrix.tsv; only aggregate audit cross-checks use the
other project's tables. Its cohorts and splits are not adopted.

The manifest has 1,133 slides; excluding one non-primary/metastatic slide leaves
1,132 slides for 1,061 patients. Primary-sample Xena matching gives 825 patients
with PAM50 including Normal-like; exclude 23 Normal-like from primary
classification, leaving 802 labeled patients: LumA 410, LumB 190, Basal 137,
Her2 65. There are 872 patients with all declared clinical inputs and 665 with
both complete inputs and a four-class label. Missingness indicators and
imputation preserve incomplete clinical records.

Across all 1,098 GDC clinical cases, PAM50 counts match the reference audit
(LumA 421, LumB 192, Basal 140, Her2 67, Normal 23). Agent3's endpoint policy
uses primary-diagnosis/GDC follow-up values plus primary-sample Xena fallback.
It finds 1,081 positive-duration OS-eligible cases versus the reference's 1,075:
six additional censored cases have reference durations missing and Agent3
durations 394–677 days. Deaths remain 149. Per-case differences stay locally in
outputs/tcga_brca/preprocess/os_discrepancies.json. This audit difference is
reported rather than forcing agreement with another project's endpoint policy.

## Local pretrained models

Exactly UNI2-h, Virchow2, CONCH-Z and MUSK are approved. Every file was rechecked
against DOWNLOAD_MANIFEST.json in job 5003331. No second pathology-weight copy
was downloaded. Remote CONCH-Z code was read before execution and its hashes
are recorded. CPU strict-loading/random-forward diagnostic job 5003358 passed
for all four, including text/image dimensions for CONCH-Z and MUSK. H100 bf16
acceptance job 5003345 remains queued at the time of this entry.

| Model | Pinned revision | Single-scale image dimension |
|---|---|---|
| UNI2-h | d517a8dd47902dd7c308b3c36f63bce47e7b9a43 | 1536 |
| Virchow2 | 3158645804b69e3f3bc4439d4116edddf0840a72 | 2560 (CLS + mean non-register patch tokens) |
| CONCH-Z | 60211fced9c25a014494e248eedb6a6be80109f7 | 768 |
| MUSK | de1ffed28608c197d2903f6fa42b491a3fbf0fb8 | 1024 |

All four use CC-BY-NC-ND-4.0 academic/noncommercial terms. Weights and fine-tuned
derivatives are excluded from Git/submission. Public-FM pretraining overlap
with TCGA cannot be excluded from these manifests and must remain a limitation.

ImageNet ResNet18 is a separate non-pathology reference; its official torchvision
weight URL, hash and cache path are recorded in logs/preflight/imagenet_weights.json.

### Production verification update, November 3, 2026

The earlier queued H100 acceptance **5003345** subsequently passed on an actual
H100 80GB with torch 2.8.0+cu128, including all four native model forwards.
The authoritative receipt is `logs/preflight/gpu_runtime.json`. Production uses
UNI2-h, CONCH-Z and the separate ImageNet ResNet18 reference. At pinned revision
60211fce..., CONCH-Z is the aligned text tower paired with the CONCH v1.5 vision
encoder, as documented in its local model card. Model/data references are listed
in `docs/REFERENCES.json`; model-card references identify exact revisions rather
than implying weights from a different release were used.

All 1,132 primary slides now have three complete feature stores. One explicit
thumbnail-artifact repair was necessary for slide ba1d2e38-fd12-478a-976a-e6701ed784e2;
the raw image remained untouched and no patient was removed. Original derived
coordinates, failed extraction receipt, before/after diagnostics and repair
source/hash are retained in Agent3. Region QC, cap and model rules did not change;
see the dated amendment in METHOD_NOTES.md.
