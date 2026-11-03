"""Inspect the original sparse 10x matrices without fitting or modifying data."""
import argparse
import collections
import json
from pathlib import Path

import h5py
import numpy as np


def inspect(path):
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        shape = tuple(int(x) for x in matrix["shape"][:])
        barcodes = matrix["barcodes"].asstr()[:]
        feature_types = matrix["features/feature_type"].asstr()[:]
        feature_ids = matrix["features/id"].asstr()[:]
        nnz = len(matrix["data"])
        assert len(barcodes) == shape[1]
        assert len(set(barcodes)) == len(barcodes)
        assert len(feature_ids) == len(feature_types) == shape[0]
        assert len(matrix["indices"]) == nnz
        pointers = matrix["indptr"][:]
        assert len(pointers) == shape[1] + 1
        assert pointers[0] == 0 and pointers[-1] == nnz
        assert np.all(pointers[1:] >= pointers[:-1])
        counts = dict(collections.Counter(feature_types))
        assert counts.get("Gene Expression", 0) > 0 and counts.get("Peaks", 0) > 0
        result = {
            "file": path.name,
            "shape_features_by_barcodes": shape,
            "barcodes": len(barcodes),
            "features_by_type": counts,
            "nonzero_entries": nnz,
            "count_dtype": str(matrix["data"].dtype),
            "genomes": sorted(set(matrix["features/genome"].asstr()[:])),
            "unique_barcodes": True,
            "sparse_structure_checks_passed": True,
        }
        return result, set(barcodes), feature_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("--sample", default="pbmc_granulocyte_sorted_10k")
    args = parser.parse_args()
    root = args.data_directory
    sample = args.sample
    filtered, filtered_barcodes, filtered_ids = inspect(root / f"{sample}_filtered_feature_bc_matrix.h5")
    raw, raw_barcodes, raw_ids = inspect(root / f"{sample}_raw_feature_bc_matrix.h5")
    assert filtered_barcodes <= raw_barcodes
    assert np.array_equal(filtered_ids, raw_ids)
    with (root / f"{sample}_atac_peaks.bed").open() as handle:
        peak_count = sum(1 for line in handle if line.strip() and not line.startswith("#"))
    assert peak_count == filtered["features_by_type"]["Peaks"]
    result = {
        "filtered": filtered,
        "raw": raw,
        "filtered_barcodes_subset_of_raw": True,
        "same_feature_order_in_raw_and_filtered": True,
        "bed_peaks_match_matrix_feature_count": True,
        "paired_modalities": "Gene Expression and Peaks share the same barcode columns in each matrix.",
        "scope": "Structural inspection only; no preprocessing, model training or biological validation.",
    }
    text = json.dumps(result, indent=2) + "\n"
    (root / "download_metadata/matrix_inspection.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
