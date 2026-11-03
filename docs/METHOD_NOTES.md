# Pre-registered protocol — version 1

Written November 2, 2026 before any real-data representation pilot or baseline
training. Input inspection and train-fitted preprocessing precede this protocol;
no validation model scores or test predictions informed these choices. Changes
after empirical results require a dated amendment and preserved prior evidence.

## Objective-dependent planning

Rules branch on modality types, pairing, sample size, supervision and missingness,
never dataset names. Every registry family receives an applicability status and
a rule/evidence-based reason. Evaluation-only labels cannot fit representations.
They may stratify split assignment and evaluate validation embeddings; they never
select genes/peaks, QC cutoffs or train encoders. Classification probes are
separate supervised downstream analyses with fixed representation parameters.

Raw paired count modalities admit MultiVI, muon WNN, Gaussian MOFA+ and dual-view
contrastive methods. MOFA+ acts on train-fitted PCA/LSI scores; it is a Gaussian
factor model and is not described as a count model. Muon WNN learns a training
graph; a declared barycentric nearest-neighbor extension maps unseen cells into
training WNN diffusion coordinates. It is an inductive extension of WNN, not
a claim that muon supplies a native out-of-sample encoder. A Seurat/Signac R
training-only graph cross-check will assess the extension/reference limitations.

Gigapixel images insert metadata, tissue QC, tiling and encoder-feature stages.
Images paired with small clinical tables admit frozen-FM/MIL baselines,
supervised early/late/gated fusion and image-tabular InfoNCE, with a small-pair
caution. Fine-tuning arms must update actual pretrained tower parameters and
remain separately named from frozen-feature probes. Clinical templated text uses
exactly the same allowed fields as the table; it introduces no new information.

## Splits and preprocessing

- Seed 774; disjoint 70/15/15 train/validation/test, balanced within target strata.
  All eligible patients, including unlabeled patients, are assigned before any
  modeling. All slides/tiles of a patient retain that patient's split.
- Every transform fits training samples only, recording fitted IDs and split
  hashes. Test data are unavailable to model workers until a final plan is
  frozen and hash-verified. Held-out embeddings use existing encoders.
- RNA: raw counts preserved; training-only log-normalized Seurat HVGs (up to
  3,000), gene scaling/clipping at ±10 and PCA (32 dimensions). RNA library-size
  normalization is per-cell; no cross-split parameters are learned.
- ATAC: training-frequency minimum max(3, 0.1% of training cells), up to 30,000
  peaks chosen by binary variance, binary TF-IDF with training IDF and log1p
  scaling, LSI (32 dimensions). Drop the first component only when its absolute
  training correlation with log depth exceeds 0.5; report the actual value.
- Count QC: median ±4 scaled MAD of log1p UMI/gene/fragment counts; upper
  mitochondrial cutoff max(5%, min(20%, median+4 MAD)); lower MAD gates for
  available peak/TSS fragment fractions. Parameters are training-fitted.
- Scrublet: training-only doublet diagnostic, expected rate 6%. No automatic
  exclusion without a validated out-of-sample detector. Residual-doublet caveat
  is mandatory. TSS fragment fraction is explicitly a proxy, not conventional
  flanking-normalized enrichment. Nucleosome signal is not computed unless a
  separately recorded fragment analysis is performed.
- Clinical allowlist: age, primary AJCC stage group, histological type and
  ER/PR/HER2 status. Train median numeric imputation/scaling, train categorical
  vocabulary/one-hot, and one missingness indicator per original variable.
  Unknown test categories map to all-zero one-hot plus the original missingness
  indicator. Missing, not evaluated, equivocal and indeterminate remain explicit.
- Clinical sensitivity set removes receptors. No PAM50/molecular-derived,
  follow-up, outcome or treatment fields enter tabular or text inputs.

## WSI and encoder budget policy

Use primary diagnostic slides only; read metadata and regions directly from the
read-only shared store. At most two concurrent slide readers. Tissue detection
uses thumbnail saturation thresholding and morphology. Tiles have approximately
0.5 µm/px physical resolution, with native encoder resizing (224/384/448 px).
Record missing MPP fallback, tissue/pen/blur decisions, failures and retained
coordinates; never silently drop failed slides. Stain normalization is disabled
unless the training-only throughput/QC pilot demonstrates a specific need.

A seeded training-only pilot of about 20 slides measures each encoder's tiles/s
and reader throughput. Exactly four local pathology FMs are runtime-tested;
production initially considers one vision-only FM, one image-text FM and an
ImageNet ResNet18 baseline. Encoder selection uses verified accessibility,
native input size, measured speed and the GPU budget, before held-out labels.
All other encoder statuses/reasons remain recorded. Fix the tile cap from
64/128/256 using measured throughput and a safety factor of 2. Store coordinates
and features, not production tile images. A separately bounded train/validation
tile cache is allowed only for actual tower fine-tuning and a small QC gallery.

Deadline: November 10, 2026, Eastern date; exact cutoff hour unspecified. Hard
limits: 48 aggregate GPU-hours and two concurrent GPUs. Initial planning envelope
(not a promise to consume it): 4 h environment/smoke, 4 h throughput/micro-pilots,
8 h extraction, 16 h training/tuning, 6 h sensitivities, 6 h contingency; 4 h
unreserved. Reservations and actual elapsed GPU-hours are recorded/reconciled.
If projected cost exceeds the cap, reduce declared tile counts or candidate
capacity before starting; never exceed the cap without user approval.

## Fair pilots and downstream probes

All eligible candidates are attempted, with failures retained. Seeds 774/775/776;
one heuristic configuration first. Default per-seed pilot: at most 600 fit
seconds, 50 epochs, validation early stopping patience 10. MOFA epochs are
variational iterations. Family-specific throughput and convergence are reported;
equal epoch counts are not claimed to be equivalent work. Final training: at
most 150 epochs, retaining training-only fitting and validation early stopping.
Train subsets used by expensive fine-tuning arms are explicitly recorded and
never treated as comparable data access without a caveat.

Baseline representations: RNA PCA, ATAC LSI, block-scaled concatenation and WNN;
clinical-only, mean-pooled frozen histology, early concatenation and stronger
histology-only gated ABMIL. Every representation uses identical probe selection:
logistic regression C in [0.1, 1, 10] and kNN k in [5, 15, 30], validation
macro-F1, fixed tie preference for the smaller C/k. Standardization fits train
embeddings only. End-to-end supervised heads are reported separately.

Unsupervised clustering uses Leiden at predeclared resolution 1.0. Label ARI/NMI
are descriptive, and not used to tune the resolution against held-out labels.
UMAPs are visualizations, not selection evidence. All cell labels are curated
WNN-derived references, with 19-type primary and versioned coarse-level analyses.

## Selection and tuning

Reject non-finite or collapsed representations: effective rank <1.5, all
dimensions variance <1e-8, or >95% near-duplicate vectors. Flag disconnected
graphs and non-convergence; assess them explicitly before accepting a model.
Validation evidence separates structure, alignment, utility, modality gain,
seed/subsample stability, QC/site confounding and compute. Runtime is fit-only.

Primary selection uses an epsilon-Pareto set over objective primary metrics.
Practical equivalence margin is 0.01 absolute for macro-F1/balanced accuracy or
recall@10, and 0.02 for C-index. These are project decision thresholds, not
external scientific standards. A method is dominated only when another is no
worse within the margin on every primary metric and better beyond the margin
on at least one. Prefer fewer trainable parameters, then lower fit cost within
equivalence; never select by raw training loss or import time. Keep the best
single modality and simple fusion as references even if not selected.

Advance at most two candidates per dataset; tune a small family-specific grid
(learning rate, latent dimension, contrastive temperature, dropout or MIL width).
Keep the default unless improvement exceeds the margin. Freeze chosen configs,
seeds, training scopes, checkpoints, code/config/preprocessing/weight hashes and
probe hyperparameters before test access. A baseline may win.

At most one scientifically valid retry: OOM → half batch size; measured numerical
divergence → halve learning rate with gradient clipping; API incompatibility →
record a verified compatible package/API fix and rerun the same genuine method.
Never silently rename another method as the failed one.

## Evaluation, uncertainty and verdict

Compute structure (Leiden ARI/NMI, label silhouette, train→test kNN/logistic,
per-class graph connectivity); global pairwise-distance correlations; modality
neighbor overlap; per-view alignment (FOSCTTM, symmetric recall@1/5/10 and mixing);
utility; modality ablation; effective rank/variance/duplicates; QC/depth or site
predictability; seed and matched subsample stability; fit time/memory/GPU-hours.

Retrieval/FOSCTTM candidate pool is all eligible held-out samples; explicitly
report pool size and direction. A training-fitted cross-view ridge/kNN map is
a separate baseline/probe when an adapter has no native shared-space encoder.
RNA→ATAC targets are binary train-selected peaks; evaluate per-cell/per-peak
AUROC/AUPRC against peak-frequency and kNN-in-RNA-PCA baselines. MultiVI uses
its native decoder with target modality masked. No target modality is supplied
to a cross-modal prediction call.

Bootstrap 500 paired resamples, seed 774, for held-out primary metrics and paired
differences. Sampling unit is cell for PBMC and patient for TCGA; the former does
not establish donor generalization. Report missing-class resamples and rare-class
uncertainty. TCGA includes class recall, worst-class recall, OVR AUROC, confusion,
LumA/LumB distinction, receptor-removal and missing-modality sensitivities, site
prediction with chance references, and a fresh site-held-out refit. Exploratory
survival uses training-fitted penalized Cox, train-defined risk groups, held-out
C-index and KM curves; report event counts and endpoint lineage/disagreements.

Verdict rules read saved validated evidence. `integrate`: joint beats both best
single and simple fusion beyond margin, paired CI lower bounds >0, no major
validity/confounding flags. `integrate_with_caution`: positive gain but uncertain
CI or conflicting/confounded evidence. Otherwise `single_modality_sufficient`.
For multiple primary metrics, claims identify which improve and which trade off;
no unexplained weighted aggregate is permitted. Approved/prohibited uses are
concrete and retain single-donor, annotation-bias and site caveats.

## Reporting and evidence

Only validated saved evidence supplies report numbers. Each claim maps to its
artifact and field. Generated reports lead with objective, decisions, findings
and verdict. Deterministic templates and real figures; no invented values,
placeholder conclusions or implied LLM use. Project report ≤4 body pages.
Inspect every rendered page and verify text extraction before packaging.

## Pre-pilot implementation clarification (November 2, 2026)

No real baseline/model pilot has run. Synthetic acceptance tests exposed a need
to make the tie policy explicit: after epsilon-Pareto filtering, keep candidates
within the first primary metric's margin, then apply the next primary metric's
margin to that constrained group. Thus PBMC structure must be equivalent before
alignment distinguishes candidates. Only after these quality constraints prefer
fewer trainable parameters, then lower embedding dimension, then fit time rounded
to whole seconds, then a fixed model-ID tie. Sub-second timings cannot select a
winner. The final verdict checks paired differences for every primary metric,
not only macro-F1; an alignment gain with equivalent structure can support a
cautious integration verdict.

Macro-F1 uncertainty uses paired unstratified sample bootstrap as above. Balanced
accuracy and TCGA OVR AUROC additionally use paired label-stratified bootstrap to
preserve rare-class support. Retrieval uncertainty resamples paired queries with
the gallery fixed; pool sizes and this conditional interpretation are recorded.
Selected-model seed sensitivity reports held-out metrics for all predeclared
final seeds. Both deterministic references and stochastic selected models get
80%-training adapter-refit neighbor stability conditional on the fixed train-only
preprocessing. This is not a claim of full preprocessing-refit stability.

Actual tower-finetuned finalists are genuinely refit for receptor removal and
site-held-out sensitivity, initialized from the original local pretrained weights.
Clinical text obeys the receptor-removal allowlist. MOFA+ external samples use a
declared fixed-loading/noise weighted-ridge factor extension; loading uncertainty
is ignored, so this is not the full variational posterior. The canonical R
Seurat/Signac WNN training-only graph cross-check is diagnostic and never selects
models. Per-run R session information and the R inventory accompany it.

When the selected count embedding predicts a QC/depth variable with held-out
R² above 0.5, add an explicit confounding caution; biological and technical
associations are not automatically separable. This is a project flag, not an
external standard. Shortlist LR changes are replicated across the same declared
seeds before acceptance and reranking. MOFA tunes factor dimension rather than
an unused learning rate. Selected/required-reference final checkpoints are fixed
before any held-out metrics are read.

### Diagnostic definition clarification, 2026-11-03, before real model fitting

Constant QC targets do not yield an informative R-squared and are explicitly
marked not applicable, as are variables with fewer than 11 finite observations
in either scope or a non-finite score. No finite replacement value is used.
Site predictability requires at least two training sites and at least two known
evaluation sites. Its declared chance reference is uniform independent prediction
over the training-site classes (1/K), including when the held-out pool contains a
subset of those classes. Unseen sites are counted separately. These numerical
guards were motivated by synthetic report QA, without observing real model
performance; primary metrics, pilot budgets and decision margins are unchanged.

### Measured queue constraint, 2026-11-03, before TCGA model fitting

The real throughput pilot completed on 20 training slides (job 5003454). Its
budget-only 256-tile projection was 7.881 aggregate GPU-hours with twofold safety.
The live priority association permits one GPU, so production uses the authorized
two-hour short partition. The largest predeclared cap satisfying two shards and
90 seconds initialization allowance per task is 128: 3.941 aggregate hours,
1.970 hours per task plus startup. All encoders and downstream methods share this
fixed cap; no TCGA model performance informed this operational change. The exact
raw measurement is retained as throughput_measurement.json, and the derived
choice/alternatives are recorded in encoder_selection.json and decision_log.jsonl.
The feature array is throttled to one GPU while PBMC is active, then may expand
to two only through a locked ledger check. Aggregate hours always sum both tasks.

### Explicit coordinate QC repair, 2026-11-03, before TCGA model fitting

The first feature array completed 1,131 slides and stopped with one region-QC
failure. Slide ba1d2e38-fd12-478a-976a-e6701ed784e2 contains a large uniform red
edge artifact. All 256 original thumbnail-selected coordinates fell in that
artifact; none passed the unchanged blur gate. This was diagnosed using image
quality alone, without inspecting PAM50 or model validation/test results.

The explicit coordinate postprocessor identifies saturated (>0.95), nearly
constant (mean channel local variance <1 in a 5-pixel window), edge-connected
components occupying >=1% of the thumbnail, and masks them to white before the
existing thumbnail Otsu/morphology/seeded-sampling routine. It does not modify
the raw slide or pixel-level encoder input. The original files and failure
receipt are archived. The base signature still identifies original inputs and
base tiler; a separate coordinate_repair receipt identifies postprocessor source,
original/new coordinate hashes and the masked fraction. The normal tile CLI
refreshes actual artifact checksums; no completion flag is manually edited.

After repair, 120/128 production candidates passed the same region QC (244/256
in the diagnostic pool). All encoders use those same fixed candidates; the cap,
blur/tissue/pen gates, cohort and split are unchanged. Synthetic checks verify
removal of the uniform edge region while retaining textured content, and refusal
to repair an image without a qualifying artifact. The Slurm feature wrapper
preserves any failure receipt and permits one such explicit input repair; other
errors still halt. No PBMC core, checkpoint, metric or held-out decision changed.

### Frozen-export IO compatibility, 2026-11-03

TCGA completed all candidate fits and froze 11 final runs before embedding export.
The Arrow writer then rejected an object column mixing NumPy float32 and float64
vectors; no TCGA held-out metric had been calculated. The same genuine frozen
encoders had already produced seven NPZ representation files.

The scoped production CLI serialization adapter converts only floating vector
cells in the embedding column to Python lists for Arrow list<double> inference.
It preserves float64 precision and verifies exact values after a Parquet round
trip. It does not alter the model, preprocessing, selection, or frozen-plan
implementation, and records its own component source hash separately. Two
regression checks include float64 values that would be lost by narrowing to
float32. Real isolated QA checks all 7,427 exported rows plus IDs/splits, the
original frozen plan and all 11 checkpoints. Normal stage execution and artifact
checksums still determine completion; no completion record is edited manually.

The patient table's fix exposed a second Arrow restriction: a nested empty
remote_code_sha256 dictionary in a slide encoder receipt cannot be stored as an
empty Parquet struct. Encoder-receipt dictionaries are therefore stored as
canonical JSON strings; parsing must exactly reproduce the original dictionary,
including empty fields. No dummy metadata is inserted. Full real-data QA covers
all patient vectors and all 1,132 slide vectors plus complete scalar and encoder
metadata, not only a minimal dtype example. The original frozen fingerprints
remain unchanged. This is the same IO component, not a model or analysis change.
