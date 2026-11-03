import numpy as np
import tifffile

from agent3_replearn.preprocessing.wsi import read_tile, tile_quality, tissue_coordinates


def test_real_tiled_tiff_opening_tissue_and_region_qc(tmp_path):
    import openslide
    rng = np.random.default_rng(774)
    image = np.full((1024, 1024, 3), 245, dtype=np.uint8)
    texture = rng.integers(0, 80, (768, 768, 3), dtype=np.uint8)
    image[128:896, 128:896] = np.clip(texture.astype(int) + [120, 30, 100], 0, 255).astype(np.uint8)
    path = tmp_path / 'synthetic_tiled.tiff'
    tifffile.imwrite(path, image, tile=(128, 128), photometric='rgb', compression='deflate')
    assert openslide.OpenSlide.detect_format(str(path))
    coordinates, metadata, _, _ = tissue_coordinates(path, seed=774, tile_size=128, target_mpp=.5, cap=8)
    assert len(coordinates) > 0
    with openslide.OpenSlide(str(path)) as slide:
        tile = read_tile(slide, coordinates[0], metadata)
    ok, qc = tile_quality(tile)
    assert ok and qc['tissue_fraction'] > .4


def test_coordinate_resume_repairs_corrupted_and_missing_files(monkeypatch):
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    import pandas as pd

    from agent3_replearn.preprocessing import wsi
    from agent3_replearn.utils import ROOT

    calls = []

    def synthetic_coordinates(*args):
        calls.append(args)
        return np.asarray([[0, 0]], dtype=np.int64), {}, None, None

    monkeypatch.setattr(wsi, 'require_compute', lambda: None)
    monkeypatch.setattr(wsi, 'tissue_coordinates', synthetic_coordinates)
    with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as directory:
        root = Path(directory)
        source = root / 'slide.tiff'
        source.write_bytes(b'input fixture')
        inventory = pd.DataFrame([dict(slide_id='slide', patient_id='patient', path=str(source), primary_eligible=True)])
        monkeypatch.setattr(wsi.pd, 'read_parquet', lambda _: inventory)
        cfg = SimpleNamespace(out=root / 'output', preprocessing={}, seed=774)
        artifacts = wsi.tile_stage(cfg)
        coordinate = cfg.out / 'features/coordinates/slide.npz'
        assert coordinate in artifacts
        wsi.tile_stage(cfg)
        assert len(calls) == 1
        coordinate.write_bytes(b'corrupted')
        wsi.tile_stage(cfg)
        assert len(calls) == 2
        coordinate.unlink()
        wsi.tile_stage(cfg)
        assert len(calls) == 3
        assert np.load(coordinate)['coordinates'].tolist() == [[0, 0]]

