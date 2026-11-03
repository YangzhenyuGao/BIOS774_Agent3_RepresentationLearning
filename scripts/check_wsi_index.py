"""Bounded training-slide gallery for physical-coordinate and tissue-mask QA."""

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd

from agent3_replearn.config import load_config
from agent3_replearn.preprocessing.wsi import read_tile, seed_for, tile_quality, tissue_coordinates
from agent3_replearn.utils import checksum, read_json, require_compute, write_json

require_compute()
cfg = load_config('config/tcga_brca.yaml')
train = set(read_json(cfg.out / 'splits/split_manifest.json')['groups']['train'])
inventory = pd.read_parquet(cfg.out / 'preprocess/slide_inventory.parquet')
inventory = inventory[inventory.primary_eligible & inventory.patient_id.isin(train)].sort_values('slide_id')
coordinates = cfg.out / 'features/coordinates'
destination = cfg.out / 'features/coordinate_qc'
destination.mkdir(exist_ok=True)
selected, strata = [], set()
for row in inventory.itertuples():
    meta = read_json(coordinates / (row.slide_id + '.json'))
    key = (meta['objective_power'], round(meta['mpp_x'], 2))
    if key not in strata:
        selected.append(row)
        strata.add(key)
    if len(selected) == 3:
        break
receipts = []
for row in selected:
    _, meta, thumbnail, mask = tissue_coordinates(row.path, seed_for(row.slide_id, cfg.seed),
        cfg.preprocessing['tile_size'], cfg.preprocessing['target_mpp'], max(cfg.preprocessing['tile_cap_options']),
        cfg.preprocessing['tissue_fraction_min'])
    path = coordinates / (row.slide_id + '.npz')
    points = np.load(path)['coordinates']
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), gridspec_kw={'width_ratios': [2, 1, 1, 1]})
    axes[0].imshow(thumbnail)
    axes[0].imshow(np.ma.masked_where(~mask, mask), cmap='autumn', alpha=.25, vmin=0, vmax=1)
    axes[0].scatter(points[:, 0] * thumbnail.shape[1] / meta['dimensions'][0],
                    points[:, 1] * thumbnail.shape[0] / meta['dimensions'][1], s=2, c='cyan')
    axes[0].set_title(f"{row.patient_id}; {meta['mpp_x']:.3f} µm/px\nTissue mask and sampled coordinates")
    quality = []
    with openslide.OpenSlide(row.path) as slide:
        for index, ax in enumerate(axes[1:]):
            point = points[index % len(points)]
            tile = read_tile(slide, point, meta)
            passed, metrics = tile_quality(tile, cfg.preprocessing['blur_variance_min'])
            quality.append(dict(coordinate=point.tolist(), passed=passed, **metrics))
            ax.imshow(tile)
            ax.set_title(f"QC {'pass' if passed else 'reject'}\n{meta['physical_tile_width_um']:.1f} µm field")
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.subplots_adjust(top=.78)
    image = destination / (row.slide_id + '.png')
    fig.savefig(image, dpi=140)
    plt.close(fig)
    receipts.append(dict(slide_id=row.slide_id, patient_id=row.patient_id, split='train',
        coordinate_sha256=checksum(path), image=str(image), image_sha256=checksum(image), tile_quality=quality))
write_json(destination / 'manifest.json', dict(selection='First training slide per objective-power/rounded-MPP stratum, up to three; no labels used',
    scope='Visual QC only; rejected example tiles are not exceptions to production QC', slides=receipts))
print(f'Saved {len(receipts)} bounded training-slide coordinate-QC panels.')
