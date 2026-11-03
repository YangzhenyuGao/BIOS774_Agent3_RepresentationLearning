#!/usr/bin/env bash
# Download original 10x outputs without extracting or transforming them.
set -euo pipefail
DATA_ROOT=${AGENT3_DATA_ROOT:-/vast/som/yunligrp/users/ygao/BIOS774Agent/data}
SAMPLE=pbmc_granulocyte_sorted_10k
DEST="$DATA_ROOT/$SAMPLE"
BASE="https://cf.10xgenomics.com/samples/cell-arc/1.0.0/$SAMPLE"
# Seurat team's curated cell-type annotations for this sample (WNN vignette), shipped as an R data package.
ANNOTATION_URL="https://seurat.nygenome.org/src/contrib/pbmcMultiome.SeuratData_0.1.4.tar.gz"
mkdir -p "$DEST/download_metadata"
cd "$DEST"
files=(
  filtered_feature_bc_matrix.h5 raw_feature_bc_matrix.h5
  atac_fragments.tsv.gz atac_fragments.tsv.gz.tbi
  atac_peaks.bed atac_peak_annotation.tsv per_barcode_metrics.csv
  summary.csv analysis.tar.gz web_summary.html
)
for suffix in "${files[@]}"; do
  name="${SAMPLE}_${suffix}"
  url="$BASE/$name"
  headers="download_metadata/$name.headers"
  curl --fail --silent --show-error --location --head --retry 4 \
    --connect-timeout 30 --max-time 120 "$url" > "$headers"
  expected=$(awk 'tolower($1)=="content-length:" {gsub("\r", "", $2); n=$2} END {print n}' "$headers")
  if [[ -f "$name" ]]; then
    if [[ -n "$expected" && $(stat -c %s "$name") != "$expected" ]]; then
      echo "Existing file does not match remote size: $name" >&2
      exit 1
    fi
    recorded=""
    if [[ -f SHA256SUMS ]]; then
      recorded=$(awk -v name="$name" '$2==name {print}' SHA256SUMS)
    fi
    if [[ -n "$recorded" ]]; then
      printf '%s\n' "$recorded" | sha256sum --check --status
    elif [[ -z "$expected" ]]; then
      echo "Cannot verify existing file: no remote size or saved checksum: $name" >&2
      exit 1
    fi
    echo "Existing file verified: $name"
    continue
  fi
  echo "Downloading: $name (expected bytes: ${expected:-not supplied})"
  curl --fail --silent --show-error --location --retry 6 --retry-delay 5 \
    --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
    --continue-at - --dump-header "download_metadata/$name.transfer_headers" \
    --output "$name.part" "$url"
  actual=$(stat -c %s "$name.part")
  if [[ -n "$expected" && "$actual" != "$expected" ]]; then
    echo "Size mismatch: $name ($actual != $expected)" >&2
    exit 1
  fi
  mv "$name.part" "$name"
  echo "Complete: $name ($actual bytes)"
done
mkdir -p annotations
annotation="annotations/$(basename "$ANNOTATION_URL")"
if [[ ! -f "$annotation" ]]; then
  curl --fail --silent --show-error --location --retry 6 --retry-delay 5 \
    --connect-timeout 30 --dump-header "download_metadata/$(basename "$ANNOTATION_URL").transfer_headers" \
    --output "$annotation.part" "$ANNOTATION_URL"
  mv "$annotation.part" "$annotation"
  echo "Complete: $annotation ($(stat -c %s "$annotation") bytes)"
fi
printf 'file\tbytes\turl\n' > download_metadata/sources.tsv
for suffix in "${files[@]}"; do
  name="${SAMPLE}_${suffix}"
  printf '%s\t%s\t%s/%s\n' "$name" "$(stat -c %s "$name")" "$BASE" "$name" >> download_metadata/sources.tsv
done
printf '%s\t%s\t%s\n' "$annotation" "$(stat -c %s "$annotation")" "$ANNOTATION_URL" >> download_metadata/sources.tsv
sha256sum "${files[@]/#/${SAMPLE}_}" "$annotation" > SHA256SUMS
date -u '+%Y-%m-%dT%H:%M:%SZ' > download_metadata/completed_utc.txt
hostname > download_metadata/download_host.txt
echo 'All downloads complete; SHA256SUMS and source metadata saved.'
