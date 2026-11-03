"""Explicit allowlist and train-fitted mixed clinical encoding."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ALLOWED = {"age", "stage", "histological_type", "er", "pr", "her2"}


class ClinicalTransform:
    def __init__(self, columns):
        if not set(columns) <= ALLOWED or not columns:
            raise ValueError("Clinical input allowlist violation")
        self.columns = columns

    def fit(self, frame):
        frame = self.clean_frame(frame)
        numeric = [c for c in self.columns if c == "age"]
        categorical = [c for c in self.columns if c != "age"]
        self.transform = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical,
                ),
            ],
            remainder="drop",
        )
        self.transform.fit(frame)
        self.fit_ids = frame.index.astype(str).tolist()
        return self

    def clean_frame(self, frame):
        frame = frame[self.columns].copy()
        for c in self.columns:
            if c == "age":
                frame[c] = pd.to_numeric(frame[c], errors="coerce")
            else:
                frame[c] = frame[c].replace({"missing": np.nan, None: np.nan})
        return frame

    def encode(self, frame):
        frame = self.clean_frame(frame)
        # One indicator per original variable, regardless of training missingness.
        indicators = frame.isna().to_numpy(dtype=np.float32)
        return np.concatenate([self.transform.transform(frame), indicators], axis=1).astype(np.float32)


def clinical_text(frame, columns):
    if not set(columns) <= ALLOWED:
        raise ValueError("Clinical-text input allowlist violation")
    # Version 1 includes exactly the same fields as the tabular arm, in fixed order.
    lines = []
    for _, row in frame[columns].iterrows():
        parts = []
        for c in columns:
            value = row[c]
            if pd.isna(value) or str(value) == "missing":
                value = "not recorded"
            elif c == "age":
                value = f"{float(value):.1f} years"
            parts.append(f"{c.replace('_', ' ')}: {value}")
        lines.append(". ".join(parts) + ".")
    return lines
