"""Explicit, receipted coordinate postprocessing for a failed uniform-color artifact.

The base tiler/input signature stays attached to its original inputs. A separate
repair receipt identifies the postprocessor and the old/new coordinate hashes.
No stage manifest or completion flag is edited by this tool; the normal CLI must
revalidate tiling and perform genuine encoder extraction afterwards.
"""
import argparse
import inspect
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import openslide
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2hsv

from agent3_replearn.config import load_config
from agent3_replearn.preprocessing.wsi import seed_for, tissue_coordinates
from agent3_replearn.utils import checksum, read_json, require_compute, write_json


def remove_flat_edge_artifact(image):
    """Remove only large, nearly constant, saturated edge-connected components."""
    array = np.asarray(image.convert('RGB')).copy()
    hsv = rgb2hsv(array)
    channels = array.astype(np.float64)
    variance = np.maximum(ndimage.uniform_filter(channels * channels, size=(5, 5, 1)) -
                          ndimage.uniform_filter(channels, size=(5, 5, 1)) ** 2, 0).mean(2)
    labels, count = ndimage.label((hsv[:, :, 1] > .95) & (variance < 1))
    sizes = np.bincount(labels.ravel())
    edges = set(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])) - {0}
    keep = [label for label in edges if sizes[label] >= .01 * labels.size]
    mask = np.isin(labels, keep)
    if not mask.any():
        raise ValueError('No qualifying uniform-color edge artifact; no automatic QC relaxation allowed')
    mask = ndimage.binary_fill_holes(ndimage.binary_dilation(mask, iterations=3))
    array[mask] = 255
    return Image.fromarray(array), mask


def repair(config, slide_id):
    source = config.out / 'features/coordinates' / (slide_id + '.json')
    store = source.with_suffix('.npz')
    previous = read_json(source)
    tool_hash = checksum(Path(__file__))
    if previous.get('coordinate_repair', {}).get('tool_sha256') == tool_hash:
        assert previous['coordinate_sha256'] == checksum(store)
        return False
    diagnosis = config.out / 'features/qc_diagnosis' / slide_id
    diagnosis.mkdir(parents=True, exist_ok=True)
    for original in [source, store]:
        target = diagnosis / ('before_repair' + original.suffix)
        if not target.exists():
            shutil.copy2(original, target)
    original_open = openslide.OpenSlide
    recorded = {}

    class SegmentationView:
        def __init__(self, path):
            self.slide = original_open(path)

        def __getattr__(self, name):
            return getattr(self.slide, name)

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            self.slide.close()

        def get_thumbnail(self, size):
            original = self.slide.get_thumbnail(size)
            corrected, artifact = remove_flat_edge_artifact(original)
            original.save(diagnosis / 'segmentation_thumbnail_before.png')
            corrected.save(diagnosis / 'segmentation_thumbnail_after.png')
            Image.fromarray(artifact.astype(np.uint8) * 255).save(diagnosis / 'artifact_mask.png')
            recorded['artifact_fraction'] = float(artifact.mean())
            return corrected

    params = config.preprocessing
    with patch.object(openslide, 'OpenSlide', SegmentationView):
        coordinates, metadata, _, _ = tissue_coordinates(
            previous['path'], seed_for(slide_id, config.seed), params.get('tile_size', 224),
            params.get('target_mpp', .5), max(params.get('tile_cap_options', [256])),
            params.get('tissue_fraction_min', .6))
    if not len(coordinates):
        raise ValueError('No real tissue candidates after artifact removal')
    np.savez_compressed(store, coordinates=coordinates)
    receipt = dict(method='uniform_saturated_edge_component_excluded_before_thumbnail_Otsu',
        trigger='No candidate passed unchanged region QC',
        tool_sha256=tool_hash, postprocessor_source=inspect.getsource(remove_flat_edge_artifact),
        original_coordinate_sha256=previous['coordinate_sha256'],
        original_metadata_sha256=checksum(diagnosis / 'before_repair.json'),
        repaired_coordinate_sha256=checksum(store), **recorded,
        labels_or_model_metrics_used=False,
        region_qc_and_tile_cap_changed=False,
        base_signature_definition='Original input/config/base-tiler identity; repair is recorded separately here')
    write_json(source, {**previous, **metadata, 'coordinate_sha256': checksum(store), 'coordinate_repair': receipt})
    write_json(diagnosis / 'coordinate_repair.json', receipt)
    print('Repaired derived coordinates:', slide_id, recorded, flush=True)
    return True


def main():
    require_compute()
    parser = argparse.ArgumentParser()
    parser.add_argument('slide_id', nargs='?')
    parser.add_argument('--failures')
    args = parser.parse_args()
    config = load_config('config/tcga_brca.yaml')
    failures = read_json(args.failures)['failures'] if args.failures else [dict(slide_id=args.slide_id)]
    if not failures or any(not r.get('slide_id') for r in failures):
        raise ValueError('A failed slide must be identified explicitly')
    for row in failures:
        if 'error' in row and row['error'] != 'No tiles passed region QC; slide is not silently dropped':
            raise ValueError('This repair does not apply to other failure types')
        repair(config, row['slide_id'])


if __name__ == '__main__':
    main()
