#!/usr/bin/env python3
"""Build a competition-ready submission CSV from prediction probabilities.

This script matches the notebook's final submission logic and validates the output
against the competition sample submission before writing the final CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def _coerce_prediction_frame(predictions, label_cols):
    if isinstance(predictions, Mapping):
        return pd.DataFrame.from_dict(predictions, orient="index", columns=label_cols)

    if hasattr(predictions, "to_pandas"):
        predictions = predictions.to_pandas()

    if isinstance(predictions, pd.DataFrame):
        df = predictions.copy()
    elif isinstance(predictions, str):
        df = pd.read_csv(predictions, index_col=0)
    else:
        raise TypeError(
            "Unsupported predictions object. Pass a dict-like mapping or a CSV path."
        )

    if df.index.name is None:
        df.index = df.index.astype(str)

    if list(df.columns) != label_cols:
        missing = [column for column in label_cols if column not in df.columns]
        unexpected = [column for column in df.columns if column not in label_cols]
        if missing:
            raise ValueError(f"Prediction columns missing labels: {missing}")
        if unexpected:
            raise ValueError(f"Prediction columns include unexpected labels: {unexpected}")
        df = df[label_cols]

    return df


def build_submission(
    prediction_frame,
    sample_submission_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    sample = pd.read_csv(sample_submission_path)
    if "StudyInstanceUID" not in sample.columns:
        raise ValueError("Sample submission does not contain StudyInstanceUID")

    label_cols = [column for column in sample.columns if column != "StudyInstanceUID"]
    if not label_cols:
        raise ValueError("Sample submission has no target label columns")

    df = _coerce_prediction_frame(prediction_frame, label_cols)
    submission_ids = sample["StudyInstanceUID"].astype(str)

    if len(df) == 0:
        raise ValueError("Prediction frame is empty")

    if set(submission_ids) != set(df.index.astype(str)):
        missing_predictions = sorted(set(submission_ids) - set(df.index.astype(str)))
        unexpected_predictions = sorted(set(df.index.astype(str)) - set(submission_ids))
        raise ValueError(
            "Test prediction IDs do not match sample submission IDs. "
            f"Missing predictions: {missing_predictions[:5]}; "
            f"unexpected predictions: {unexpected_predictions[:5]}"
        )

    aligned = df.reindex(submission_ids.tolist())
    values = aligned.to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinite values")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1")

    submission = sample.copy()
    submission.loc[:, label_cols] = values

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)

    saved = pd.read_csv(output)
    if saved.shape != submission.shape or saved.columns.tolist() != submission.columns.tolist():
        raise ValueError("Saved submission does not match the sample submission schema")

    print(f"Saved {len(saved)} rows to {output}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Kaggle submission CSV from probability predictions.")
    parser.add_argument(
        "--sample",
        default="/kaggle/input/competitions/rsna-knee-abnormality-detection/sample_submission.csv",
        help="Path to the competition sample submission CSV.",
    )
    parser.add_argument(
        "--predictions",
        default="predictions.csv",
        help="CSV file containing StudyInstanceUID in the index and class probabilities as columns.",
    )
    parser.add_argument(
        "--output",
        default="/kaggle/working/submission.csv",
        help="Output submission path.",
    )
    args = parser.parse_args()

    if not Path(args.predictions).exists():
        raise FileNotFoundError(
            f"Prediction CSV not found: {args.predictions}. "
            "Run this script after generating test_probabilities or provide a valid predictions.csv file."
        )

    prediction_frame = pd.read_csv(args.predictions, index_col=0)
    build_submission(prediction_frame, args.sample, args.output)
