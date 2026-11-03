"""Inspect a failed slide without changing coordinates, QC gates or its cohort."""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd

from agent3_replearn.config import load_config
from agent3_replearn.preprocessing.wsi import read_tile, tile_quality
from agent3_replearn.utils import checksum, read_json, require_compute, write_json

require_compute()
parser = argparse.ArgumentParser()
parser.add_argument('slide_id')
args = parser.parse_args()
config = load_config('config/tcga_brca.yaml')
source = config.out / 'features/coordinates' / (args.slide_id + '.json')
metadata = read_json(source)
store = source.with_suffix('.npz')
coordinates = np.load(store)['coordinates']
out = config.out / 'features/qc_diagnosis' / args.slide_id
out.mkdir(parents=True, exist_ok=True)
records = []
with openslide.OpenSlide(metadata['path']) as slide:
    thumbnail = slide.get_thumbnail((1200, 1200)).convert('RGB')
    thumbnail.save(out / 'thumbnail.png')
    properties = dict(slide.properties)
    for index, coordinate in enumerate(coordinates):
        image = read_tile(slide, coordinate, metadata)
        passed, qc = tile_quality(image, config.preprocessing.get('blur_variance_min', 30))
        records.append(dict(index=index, x=int(coordinate[0]), y=int(coordinate[1]), passed=passed, **qc))
    frame = pd.DataFrame(records)
    frame.to_csv(out / 'candidate_qc.csv', index=False)
    chosen = list(range(min(8, len(frame)))) + frame.nlargest(8, 'blur_variance')['index'].tolist()
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax, index in zip(axes.flat, chosen):
        ax.imshow(read_tile(slide, coordinates[index], metadata))
        q = records[index]
        ax.set_title(f"#{index}: blur {q['blur_variance']:.1f}\ntissue {q['tissue_fraction']:.2f}, pen {q['pen_fraction']:.2f}", fontsize=8)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out / 'candidate_gallery.png', dpi=130)
    plt.close(fig)
write_json(out / 'diagnosis.json', dict(slide_id=args.slide_id, properties=properties,
    coordinate_source_sha256=checksum(source), coordinate_store_sha256=checksum(store),
    inspected_candidates=len(frame), passed_first_128=int(frame.iloc[:128].passed.sum()),
    passed_all=int(frame.passed.sum()), qc_summary=frame.describe().to_dict(),
    action='Diagnosis only. No original, cohort, feature or coordinate file modified.'))
print(frame.describe().to_string())
print('QC pass counts:', int(frame.iloc[:128].passed.sum()), int(frame.passed.sum()))
