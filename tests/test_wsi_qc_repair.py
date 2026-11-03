import importlib.util

import numpy as np
import pytest
from PIL import Image

from agent3_replearn.utils import ROOT


def repair_module():
    spec = importlib.util.spec_from_file_location('repair_wsi_coordinates', ROOT / 'scripts/repair_wsi_coordinates.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uniform_edge_artifact_removed_while_textured_region_is_preserved():
    pixels = np.random.default_rng(774).integers(70, 245, size=(192, 128, 3), dtype=np.uint8)
    pixels[128:] = [195, 0, 0]
    corrected, mask = repair_module().remove_flat_edge_artifact(Image.fromarray(pixels))
    actual = np.asarray(corrected)
    assert np.array_equal(actual[:110], pixels[:110])
    assert (actual[140:] == 255).all()
    assert mask[140:].all() and not mask[:110].any()


def test_texture_without_large_uniform_edge_component_is_not_repaired():
    pixels = np.random.default_rng(774).integers(70, 245, size=(192, 128, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match='No qualifying'):
        repair_module().remove_flat_edge_artifact(Image.fromarray(pixels))
