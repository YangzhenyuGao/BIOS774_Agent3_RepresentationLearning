"""Typed dataset configuration; planner rules do not inspect dataset names."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from .schemas import ComputeBudget, StrictModel
from .utils import ROOT, assert_project_output, digest


class ModalityConfig(StrictModel):
    modality_id: str
    modality_type: str
    path: str
    format: str
    pairing_key: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class Config(StrictModel):
    dataset_id: str
    modalities: list[ModalityConfig]
    output: str
    report_name: str
    seed: int = 774
    split_fractions: list[float] = Field(default_factory=lambda: [0.7, 0.15, 0.15])
    labels: dict[str, Any] = Field(default_factory=dict)
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    model_store: str = "/vast/som/yunligrp/users/ygao/models"
    encoders: list[str] = Field(default_factory=lambda: ["UNI2-h", "CONCH-Z", "imagenet_resnet18"])
    parallel_slide_readers: int = Field(default=2, ge=1, le=4)
    bootstrap_replicates: int = Field(default=500, ge=50)
    equivalence_margin: float = Field(default=0.01, gt=0)
    smoke: bool = False
    reporting_mode: Literal["deterministic"] = "deterministic"

    @model_validator(mode="after")
    def validate_paths_and_splits(self):
        assert_project_output(self.out)
        if len(self.split_fractions) != 3 or abs(sum(self.split_fractions) - 1) > 1e-8:
            raise ValueError("Three split fractions must sum to one")
        if any(x <= 0 for x in self.split_fractions):
            raise ValueError("All splits require a positive fraction")
        if len({m.modality_id for m in self.modalities}) != len(self.modalities):
            raise ValueError("Modality IDs must be unique")
        allowed = {"UNI2-h", "Virchow2", "CONCH-Z", "MUSK", "imagenet_resnet18"}
        if set(self.encoders) - allowed:
            raise ValueError("Unapproved pathology encoder")
        if self.smoke and any(not Path(m.path).resolve().is_relative_to(ROOT.resolve()) for m in self.modalities):
            raise ValueError("Smoke runs may use only synthetic fixtures stored inside Agent3")
        return self

    @property
    def out(self):
        return ROOT / self.output

    @property
    def config_hash(self):
        return digest(self.model_dump())


def load_config(path):
    path = Path(path)
    value = yaml.safe_load(path.read_text())
    base = value.pop("base", None)
    if base:
        defaults = yaml.safe_load((path.parent / base).read_text())
        value = {**defaults, **value}
    return Config.model_validate(value)
